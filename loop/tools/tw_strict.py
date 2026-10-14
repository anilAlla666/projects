"""Step 13: strict-direction training cases from the bundled Terminal Wrench tasks. Terminal Wrench only gives graders
that are too loose (they accept hacks); two of the four held-out tasks fail the other way (the grader rejects correct
work). Each case here is a task's ORIGINAL tests with one over-strict check of a kind agent-written graders really
contain, put first in test.sh:
  ref     the output must be byte-identical to one correct attempt's output (a writer checking against its own run)
  extra   a file the task never asks for must exist next to the output
  envmiss the output is compared with a reference that exists only on the writer's machine (/candidates), so on the
          grading machine every attempt fails - production-planning's shape (step 15: the grader needed files the
          grading machine does not have)
  lines   the output must have exactly the line count of one correct run (a rule the task never states, fitted to
          the writer's own run) - photonic-waveguide-routing's shape
The original tests are the known fix. The case is used only if the mutant rejects at least one seen correct candidate
(checked by running it on them in the grading sandbox) and the hidden pool scores it below the original tests.

    python3 tools/tw_strict.py --data ~/vbdata --repo ~/vbrepo2 tw-<id> ...    (on the GPU host)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

FAIL = 'mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt; exit 0'
KINDS: set = set()  # --kinds: build only these


def mutate(orig: Path, dest: Path, check: str) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(orig, dest)
    t = (dest / "test.sh").read_text()
    head, _, rest = t.partition("\n") if t.startswith("#!") else ("#!/bin/bash", "", t)
    (dest / "test.sh").write_text(head + "\n" + check + "\n" + rest)


def grade(name: str, tests: Path) -> dict:
    r = subprocess.run(["docker", "run", "--rm", "--cap-add", "SYS_ADMIN", "--security-opt", "seccomp=unconfined",
                        "--security-opt", "apparmor=unconfined", "-v", f"{tests}:/verifier_out/tests:ro",
                        f"vbhud-{name}", "bash", "/tests/test.sh"], capture_output=True, text=True, timeout=7200)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": (r.stdout + r.stderr)[-300:]}


def cases(data: Path, repo: Path, name: str) -> list[dict]:
    sys.path.insert(0, str(repo))
    from vb.probe import probe  # noqa: E402
    orig = data / "tw_graders" / name / "tests"
    cand = data / "candidates" / name / "files"
    key = json.loads((data / "candidates" / name / "key.json").read_text())
    arts = json.loads((data / "hud_tasks" / f"write-verifier-{name}" / "tests" / "grader" / "config.json")
                      .read_text())["artifacts"]
    good = [l for l, v in key.items() if l != "ids" and v == "correct"]
    out = []
    # ref: the first artifact whose content differs between two correct candidates (else no case: identical
    # outputs would pass a byte check anyway)
    for a in arts:
        blobs = {l: (cand / l / a.lstrip("/")).read_bytes() for l in good if (cand / l / a.lstrip("/")).is_file()}
        if len(set(blobs.values())) >= 2:
            l0 = sorted(blobs)[0]
            sha = hashlib.sha256(blobs[l0]).hexdigest()
            check = (f'if [ "$(sha256sum {a} 2>/dev/null | cut -d" " -f1)" != "{sha}" ]; then '
                     f'echo "FAIL: {a} does not match the expected output"; {FAIL}; fi')
            out.append(("ref", check))
            break
    a = arts[0]
    extra = str(Path(a).parent / "run.log")
    if extra not in arts:
        out.append(("extra", f'if [ ! -f {extra} ]; then echo "FAIL: {extra} missing"; {FAIL}; fi'))
    have = [l for l in good if (cand / l / a.lstrip("/")).is_file()]
    if have:
        l0 = sorted(have)[0]
        out.append(("envmiss", f'if ! cmp {a} /candidates/{l0}/{a.lstrip("/")}; then '
                               f'echo "FAIL: {a} does not match the reference output"; {FAIL}; fi'))
        counts = {l: (cand / l / a.lstrip("/")).read_bytes().count(b"\n") for l in have}
        if len(set(counts.values())) >= 2:
            n = counts[l0]
            out.append(("lines", f'if [ "$(wc -l < {a} 2>/dev/null)" != "{n}" ]; then '
                                 f'echo "FAIL: {a} should have {n} lines"; {FAIL}; fi'))
    out = [(k, c) for k, c in out if not KINDS or k in KINDS]
    res = []
    base = grade(name, orig)
    for kind, check in out:
        dest = data / "tw_graders" / f"{name}__{kind}" / "tests"
        mutate(orig, dest, check)
        ev = probe(name, dest, cand)
        rejected_good = [r["id"] for r in ev.get("runs", []) if r["id"] in good and r["verdict"] != "pass"]
        g = grade(name, dest)
        ok = bool(rejected_good) and g.get("balanced_accuracy", 1) < base.get("balanced_accuracy", 0)
        if not ok:
            shutil.rmtree(dest.parent, ignore_errors=True)
        res.append({"case": f"{name}__{kind}", "kept": ok, "rejects_seen_correct": rejected_good,
                    "hidden_before": g.get("balanced_accuracy"), "tpr": g.get("tpr"),
                    "original_tests": base.get("balanced_accuracy")})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--data", type=Path, default=Path.home() / "vbdata")
    ap.add_argument("--repo", type=Path, default=Path.home() / "vbrepo2")
    ap.add_argument("--kinds", default="", help="comma-separated subset of ref,extra,envmiss,lines")
    a = ap.parse_args()
    KINDS.update(k for k in a.kinds.split(",") if k)
    for n in a.names:
        try:
            for r in cases(a.data, a.repo, n):
                print(json.dumps(r), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"case": n, "error": str(e)[:300]}), flush=True)
