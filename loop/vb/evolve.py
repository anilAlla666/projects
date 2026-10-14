"""The self-evolving step: the loop learns rules from its own mistakes on tasks whose answers are known.

The loop never sees answer labels while it works. Afterwards, on a task with a labelled pool, this module:

1. post-mortem: reruns the loop's final grader on every pool attempt and keeps the ones it got wrong (a correct
   attempt rejected, or a wrong one accepted), each with the grader's own output, so the misfiring check is named;
2. propose: a model (Claude Opus through Claude Code on the operator's subscription) turns those mistakes and the
   judge's own report into at most two general rules for the judge, and a lint drops any rule that names the task,
   a path or a specific value, so a rule is about a class of mistake, never about one task;
3. rulebook: proposed rules go into a versioned rulebook as candidates (configs/rulebook.json); the judge reads only
   adopted rules (vb/verify_loop.rulebook_section);
4. gate: a candidate rule is adopted only if the loop with it scores higher on the hidden pools of other tasks than
   the loop without it, paired on the same starting grader, and no task drops. Otherwise it is rejected;
5. ledger: every post-mortem, proposal and gate decision is appended to runs/evolve/ledger.jsonl.

Nothing here runs inside a loop round, so a grader takes as long to verify as before.

    VB_HOST=root@<docker server> VB_DATA=/root/vbench python -m vb.evolve postmortem TASK REMOTE_TESTS_DIR [JUDGEMENT_MD]
    python -m vb.evolve propose TASK POSTMORTEM_JSON [JUDGEMENT_MD]
    python -m vb.evolve gate RULE_ID TASK:GRADER_DIR ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import host

ROOT = Path(__file__).resolve().parents[1]
RULEBOOK = ROOT / "configs" / "rulebook.json"
LEDGER = ROOT / "runs" / "evolve" / "ledger.jsonl"
MAX_RULES_PER_PROPOSAL = 2

# Runs inside the task's hidden-grader image: stage each labelled attempt the way the hidden grader does (flat mode),
# run the written grader on it, and report the attempts it got wrong with the tail of its own output.
_STAGE = r'''
import json, shutil, subprocess
from pathlib import Path
G = Path("/grader"); cfg = json.load(open(G / "config.json"))
man = json.load(open(G / "pool" / "manifest.json")); entries = man["entries"] if isinstance(man, dict) else man
out = []
for e in entries:
    for p in cfg["artifacts"]:
        t = Path(p)
        if t.is_dir(): shutil.rmtree(t, ignore_errors=True)
        elif t.exists(): t.unlink()
    for path, (sha, mode) in e.get("files", {}).items():
        d = Path(path); d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(G / "pool" / "blobs" / sha, d)
    shutil.rmtree("/tests", ignore_errors=True); shutil.copytree("/written", "/tests")
    shutil.rmtree("/logs/verifier", ignore_errors=True); Path("/logs/verifier").mkdir(parents=True)
    try:
        r = subprocess.run(["bash", "/tests/test.sh"], capture_output=True, text=True, timeout=cfg.get("attempt_timeout_sec", 600))
        text = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        text = "TIMEOUT"
    rw = Path("/logs/verifier/reward.txt")
    verdict = "pass" if rw.exists() and rw.read_text().strip() in ("1", "1.0") else "fail"
    if verdict != e["label"]:
        out.append({"id": e["id"], "label": e["label"], "verdict": verdict, "grader_output": text[-1500:]})
print(json.dumps({"attempts": len(entries), "mistakes": out}))
'''

_PROPOSE = """You are improving a verify loop: a judge model that audits and repairs a grader (the tests that decide whether
an attempt at a task is correct), using only the task, the grader and unlabelled attempts by other models.

After the loop finished on one task, its final grader was run on attempts whose correct verdict is known. Below are
the attempts it got wrong, with the grader's own output, followed by the judge's own report from that run.

Write at most {n} rules for the JUDGE that would have prevented this kind of mistake on ANY task. A rule must:
- describe a class of mistake and the behaviour that avoids it, in one or two sentences;
- never name this task, its files, paths, field names or any specific number from it;
- be checkable by the judge from the task, the grader and the evidence it already has.

Answer with JSON only: {{"rules": [{{"rule": "...", "why": "which mistakes above it explains"}}]}}

# Mistakes ({k} of {total} attempts)
{mistakes}

# The judge's report
{report}
"""


def _ledger(**kw) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}) + "\n")


def postmortem(task: str, remote_tests: str) -> dict:
    """The final grader's mistakes on the task's labelled pool, each with the grader's own output."""
    stage = host.data("evolve_stage.py")
    host.run(f"cat > {stage}", stdin=_STAGE)
    r = host.run(f"docker run --rm --entrypoint python3 -v {remote_tests}:/written:ro -v {stage}:/stage.py:ro "
                 f"vbhud-{task} /stage.py", timeout=7200)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    res = json.loads(lines[-1]) if lines else {"error": (r.stdout + r.stderr)[-800:]}
    res["task"] = task
    _ledger(event="postmortem", task=task, tests=remote_tests, attempts=res.get("attempts"),
            mistakes=len(res.get("mistakes", [])), error=res.get("error"))
    return res


def lint(rule: str, task: str) -> str | None:
    """Why a proposed rule is task-specific, or None if it is general."""
    words = {w for w in re.split(r"[-_]", task) if len(w) > 3}
    if any(w.lower() in rule.lower() for w in words):
        return "names the task"
    if re.search(r"(/[\w.-]+){2,}", rule):
        return "names a path"
    if re.search(r"\d{3,}|\d+\.\d+", rule):
        return "names a specific value"
    if len(rule) > 600:
        return "too long to be one rule"
    return None


def _claude(prompt: str, model: str = "opus") -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")}
    r = subprocess.run(["claude", "-p", prompt, "--model", model, "--output-format", "json", "--no-session-persistence",
                        "--setting-sources", "", "--allowedTools", ""], capture_output=True, text=True, env=env, timeout=1800)
    return json.loads(r.stdout).get("result", "")


def propose(task: str, pm: dict, report: str = "") -> list[dict]:
    """Candidate rules from one post-mortem, linted, added to the rulebook as candidates."""
    mistakes = pm.get("mistakes", [])
    if not mistakes:
        _ledger(event="propose", task=task, rules=0, note="no mistakes")
        return []
    text = _claude(_PROPOSE.format(n=MAX_RULES_PER_PROPOSAL, k=len(mistakes), total=pm.get("attempts"),
                                   mistakes=json.dumps(mistakes[:8], indent=1)[:12000], report=report[:8000]))
    m = re.search(r"\{.*\}", text, re.S)
    rules = (json.loads(m.group(0)).get("rules", []) if m else [])[:MAX_RULES_PER_PROPOSAL]
    book = json.loads(RULEBOOK.read_text()) if RULEBOOK.exists() else {"version": 0, "rules": []}
    added = []
    for r in rules:
        why_not = lint(r.get("rule", ""), task)
        rid = f"r{len(book['rules']) + 1}"
        entry = {"id": rid, "rule": r.get("rule", ""), "why": r.get("why", ""), "source_task": task,
                 "status": "rejected" if why_not else "candidate", "lint": why_not}
        book["rules"].append(entry)
        added.append(entry)
        _ledger(event="propose", task=task, rule_id=rid, status=entry["status"], lint=why_not)
    book["version"] += 1
    RULEBOOK.parent.mkdir(parents=True, exist_ok=True)
    RULEBOOK.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")
    return added


def decide(rule_id: str, pairs: list[dict], min_gain: float = 0.05) -> str:
    """Adopt a candidate only if the mean hidden score with it beats the paired score without it by min_gain (one
    attempt moves balanced accuracy by about 0.05 on a 21-attempt pool) and no task drops."""
    gain = sum(p["with"] - p["without"] for p in pairs) / max(1, len(pairs))
    dropped = [p["task"] for p in pairs if p["with"] < p["without"]]
    status = "adopted" if pairs and gain >= min_gain and not dropped else "rejected"
    book = json.loads(RULEBOOK.read_text())
    for r in book["rules"]:
        if r["id"] == rule_id:
            r["status"], r["gate"] = status, {"pairs": pairs, "mean_gain": round(gain, 4), "dropped": dropped}
    book["version"] += 1
    RULEBOOK.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")
    _ledger(event="gate", rule_id=rule_id, status=status, mean_gain=round(gain, 4), dropped=dropped, pairs=pairs)
    return status


def gate(rule_id: str, items: list[str], judge_model: str = "cc:opus", rounds: int = 2) -> str:
    """Paired gate: the same starting grader through the loop without and with the candidate rule, on tasks other
    than the one the rule came from."""
    from .verify_loop import run
    from .writer_api import Spend
    book = json.loads(RULEBOOK.read_text())
    rule = next(r for r in book["rules"] if r["id"] == rule_id)
    one = {"version": 0, "rules": [{**rule, "status": "adopted"}]}
    tmp = LEDGER.parent / f"gate_{rule_id}.json"
    tmp.write_text(json.dumps(one))
    pairs = []
    for item in items:
        task, grader = item.split(":", 1)
        assert task != rule["source_task"], "a rule is never gated on the task it came from"
        scores = {}
        for arm, path in (("without", ""), ("with", str(tmp))):
            os.environ["VB_RULEBOOK"] = path
            r = run(task, Path(grader), rounds, LEDGER.parent / "loops", Spend(1e9), model=judge_model, loop_cap=99.0,
                    tag=f"gate{rule_id}{arm}", judge="v3")
            scores[arm] = r["after"].get("balanced_accuracy", 0.5)
        os.environ.pop("VB_RULEBOOK", None)
        pairs.append({"task": task, **scores})
    return decide(rule_id, pairs)


def main() -> None:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "postmortem":
        pm = postmortem(args[0], args[1])
        out = LEDGER.parent / f"postmortem_{args[0]}.json"
        out.write_text(json.dumps(pm, indent=1))
        print(json.dumps({"task": args[0], "attempts": pm.get("attempts"), "mistakes": len(pm.get("mistakes", [])),
                          "file": str(out), "error": pm.get("error")}))
    elif cmd == "propose":
        pm = json.loads(Path(args[1]).read_text())
        report = Path(args[2]).read_text() if len(args) > 2 and Path(args[2]).exists() else ""
        for r in propose(args[0], pm, report):
            print(json.dumps(r, ensure_ascii=False))
    elif cmd == "gate":
        print(gate(args[0], args[1:]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
