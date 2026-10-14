"""Data for the verdict model: task text and an attempt's result files in, PASS or FAIL out (the official tests'
verdict). It learns what official tests accept and reject, so the verify loop can tell right evidence from wrong.

One builder (`render`) makes every input, from the attempt's artifact files only: training examples (harvested
leaderboard attempts), the evaluation pools (pools/<task>/kept_job/<entry>/artifacts) and the loop's candidates
(candidates/<task>/files/<letter>). The official grader's output (verifier/, test stdout, ctrf, reward) is never read.
Identifying fields (model, agent, rank, job) are dropped. Binary files are shown as their path and size only.

Split, fixed in docs/research_intent_goal.md (step 8) before training: the harvested training tasks minus the
held-out ones (the step 7 fresh tasks and the later ones that fit the product's machines), with four of the rest as
validation (threshold only).

    python -m vb.verdictdata --out /root/vbench/verdict          # on the server
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .harvest import ATTEMPT_CAP, FILE_CAP, OUT as HARVEST
from .util import DATA

HELD_OUT = ["coq-block-bound", "ctr-optimization", "music-harmony",  # step 7 fresh tasks
            "ontology-kg-querying", "retro-console-soc", "vllm-deepseek-streaming", "wal-recovery-ordering"]
EVAL_POOLS = ["hof-topology-interpenetration", "mp-checkpoint-consolidation", "photonic-waveguide-routing",
              "production-planning", "fin-saccr-rwa", "lake-temp-glm", "roy-polymorph-cn", "sound-change-cascade",
              "coq-block-bound", "ctr-optimization", "music-harmony"]
RESULT_CHARS, FILE_CHARS = 18_000, 8_000
QUESTION = ("\n\nIs this result correct by the task's own official tests? Answer with one word: PASS or FAIL.\nAnswer:")


def task_text(raw: str) -> str:
    return re.sub(r"^<!-- harbor-canary[^\n]*-->\n?", "", raw).strip()


def is_binary(text: str) -> bool:
    return "\x00" in text or text.count("�") > 0.01 * max(len(text), 1)


def render(task_text_: str, files: dict[str, str | None], sizes: dict[str, int]) -> str:
    """files: artifact path -> its text (None if too large to keep); sizes: path -> bytes. The same caps as the
    harvest (FILE_CAP per file, ATTEMPT_CAP per attempt) are applied by the readers below."""
    parts, budget = [], RESULT_CHARS
    for path in sorted(sizes):
        t = files.get(path)
        head = f"=== {path} ({sizes[path]} bytes) ==="
        if t is None:
            body = "[not shown: too large]"
        elif is_binary(t):
            body = "[binary file]"
        else:
            body = t[:min(FILE_CHARS, max(budget, 0))]
            if len(body) < len(t):
                body += f"\n[... {len(t) - len(body)} more characters]"
            budget -= len(body)
        parts.append(head + "\n" + body)
    if not parts:
        parts = ["[no result files]"]
    return f"<task>\n{task_text_}\n</task>\n\n<result>\n" + "\n".join(parts) + "\n</result>" + QUESTION


def read_dir(root: Path) -> tuple[dict, dict]:
    """An attempt's artifact files from a folder that mirrors their paths (a pool entry's artifacts/ or a
    candidate's folder), read with the harvest's caps."""
    files, sizes, used = {}, {}, 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        path = "/" + str(f.relative_to(root))
        n = f.stat().st_size
        sizes[path] = n
        if n > FILE_CAP or used + n > ATTEMPT_CAP:
            files[path] = None
            continue
        used += n
        files[path] = f.read_bytes().decode("utf-8", "replace")
    return files, sizes


def from_harvest(rec: dict) -> tuple[dict, dict]:
    files, sizes = {}, {}
    for name, o in (rec.get("outputs") or {}).items():
        path = "/" + name.removeprefix("artifacts/")
        sizes[path] = o.get("size", 0)
        files[path] = o.get("text") if o.get("kept") else None
    return files, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "verdict")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.name for p in HARVEST.iterdir() if (p / "task.json").is_file())
    train_tasks = [t for t in tasks if t not in HELD_OUT]
    val_tasks = train_tasks[3::8][:4]  # every eighth from the fourth, by name: fixed before training
    train_tasks = [t for t in train_tasks if t not in val_tasks]
    stats = {"train_tasks": train_tasks, "val_tasks": val_tasks, "held_out": HELD_OUT}
    for split, ts in (("train", train_tasks), ("val", val_tasks)):
        rows = []
        for t in ts:
            text = task_text(json.loads((HARVEST / t / "task.json").read_text())["instruction"])
            for f in sorted((HARVEST / t / "attempts").glob("*.json")):
                rec = json.loads(f.read_text())
                rows.append({"id": f"{t}/{rec['trial_id']}", "task": t, "label": rec["label"],
                             "prompt": render(text, *from_harvest(rec)), "completion": " PASS" if rec["label"] == "pass" else " FAIL"})
        random.Random(0).shuffle(rows)
        (a.out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        stats[split] = {"rows": len(rows), "pass": sum(r["label"] == "pass" for r in rows)}
    # evaluation: every kept pool entry of the evaluation and held-out tasks, labelled by the official tests
    ev = []
    for t in EVAL_POOLS:
        man = DATA / "pools" / t / "manifest.json"
        tj = DATA / "hud_tasks" / f"write-verifier-{t}" / "instruction.md"
        if not man.is_file() or not tj.is_file():
            continue
        m = re.search(r"<task-instruction>\n?(.*?)\n?</task-instruction>", tj.read_text(), re.S)
        text = m.group(1).strip()
        for e in json.loads(man.read_text()):
            if not e.get("kept"):
                continue
            d = DATA / "pools" / t / "kept_job" / e["entry_id"] / "artifacts"
            ev.append({"id": f"{t}/{e['entry_id']}", "task": t, "label": e["label"],
                       "prompt": render(text, *read_dir(d)) if d.is_dir() else render(text, {}, {})})
    (a.out / "eval_pools.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev))
    stats["eval_pools"] = {"rows": len(ev), "tasks": sorted({r["task"] for r in ev})}
    # the loop's question: the outside candidates of the nine tasks whose true labels are on file
    cand = []
    for kf in sorted((DATA / "candidate_keys").glob("*.json")):
        t = kf.stem
        tj = DATA / "hud_tasks" / f"write-verifier-{t}" / "instruction.md"
        if not tj.is_file():
            continue
        m = re.search(r"<task-instruction>\n?(.*?)\n?</task-instruction>", tj.read_text(), re.S)
        text = m.group(1).strip()
        for k in json.loads(kf.read_text()):
            d = DATA / "candidates" / t / "files" / k["letter"]
            if not d.is_dir():
                continue
            cand.append({"id": f"{t}/{k['letter']}", "task": t, "label": k["label"],
                         "prompt": render(text, *read_dir(d))})
    (a.out / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cand))
    stats["candidates"] = {"rows": len(cand), "tasks": sorted({r["task"] for r in cand})}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
