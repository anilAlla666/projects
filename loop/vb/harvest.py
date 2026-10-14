"""Step 4, the free part: labelled attempts from Terminal-Bench 4.0 tasks outside the evaluation split
(configs/model_split.json), for training the verifier model.

Per task: up to `per_label` passing and `per_label` failing attempts, spread across jobs (models) in leaderboard
order. Each trial archive is fetched into memory and only the small parts are kept: the official tests' per-test
outcomes (ctrf.json), the tests' output (test-stdout.txt), the label, and the attempt's output files up to a size
bound. Model checkpoints and other large outputs are dropped as they are read. Per task, the task text and the
official tests' source are kept beside them (the requirement side of the pair).

Tasks with community-reported test problems are left out: their labels are not trustworthy.

    python -m vb.harvest [--per-label 30] [--workers 3] [--tasks a,b]      # runs on the server; resumable
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import tarfile
import uuid
from pathlib import Path

from .index import load_attempts
from .pool import round_robin
from .util import DATA, ROOT, TB_ROOT, read_json

OUT = DATA / "harvest"
FILE_CAP, ATTEMPT_CAP, STDOUT_CAP = 64_000, 512_000, 20_000


def training_tasks() -> list[str]:
    split = read_json(ROOT / "configs" / "model_split.json")
    held = set(split["development"] + split["heldout_confirmation"] + split["test"] + list(split["excluded"]))
    m1 = read_json(ROOT / "configs" / "m1.json")["rejected"]
    broken = {t for t, rs in m1.items() if any(r.startswith("community-reported test problem") for r in rs)}
    tasks = sorted({a["task"] for a in load_attempts()})
    return [t for t in tasks if t not in held and t not in broken and (TB_ROOT / t).is_dir()]


def task_record(task: str) -> None:
    d = OUT / task
    if (d / "task.json").is_file():
        return
    src = TB_ROOT / task
    tests = {}
    for f in sorted((src / "tests").rglob("*")):
        if f.is_file() and f.stat().st_size <= FILE_CAP:
            try:
                tests[str(f.relative_to(src / "tests"))] = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"task": task, "instruction": (src / "instruction.md").read_text(encoding="utf-8"),
                                             "official_tests": tests}, indent=1))


def keep(name: str) -> bool:
    return name in ("result.json", "verifier/ctrf.json", "verifier/test-stdout.txt", "verifier/reward.txt",
                    "artifacts/manifest.json") or name.startswith("artifacts/")


def extract(archive: bytes) -> dict:
    out, files, used = {}, {}, 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tf:
        for m in tf:
            name = m.name.lstrip("./")
            if not m.isfile() or not keep(name):
                continue
            data = tf.extractfile(m).read() if m.size <= FILE_CAP or name.startswith("verifier/") else None
            if name.startswith("artifacts/") and name != "artifacts/manifest.json":
                if data is None or used + len(data) > ATTEMPT_CAP:
                    files[name] = {"size": m.size, "kept": False}
                    continue
                used += len(data)
                files[name] = {"size": m.size, "kept": True, "text": data.decode("utf-8", "replace")}
                continue
            out[name] = data.decode("utf-8", "replace") if data is not None else None
    ctrf = json.loads(out["verifier/ctrf.json"]) if out.get("verifier/ctrf.json") else None
    stdout = out.get("verifier/test-stdout.txt") or ""
    return {"tests": [{k: t.get(k) for k in ("name", "status", "message", "trace") if t.get(k) is not None}
                      for t in (ctrf or {}).get("results", {}).get("tests", [])],
            "test_stdout": stdout if len(stdout) <= STDOUT_CAP else stdout[:STDOUT_CAP // 4] + "\n...[cut]...\n" + stdout[-3 * STDOUT_CAP // 4:],
            "reward_txt": (out.get("verifier/reward.txt") or "").strip(), "outputs": files,
            "result": json.loads(out["result.json"]) if out.get("result.json") else None}


async def fetch(att: dict, db, storage, sem: asyncio.Semaphore) -> str:
    path = OUT / att["task"] / "attempts" / f"{att['trial_id']}.json"
    if path.is_file():
        return "cached"
    async with sem:
        try:
            trial = await db.get_trial(uuid.UUID(att["trial_id"]))
            if not trial or not trial.get("archive_path"):
                return "no archive"
            rec = extract(await storage.download_bytes(trial["archive_path"]))
        except Exception as e:  # one bad archive must not stop the harvest
            return f"error: {type(e).__name__}"
    rec = {**{k: att[k] for k in ("task", "trial_id", "job", "rank", "agent", "model", "reward")},
           "label": "pass" if att["reward"] == 1 else "fail", **rec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return "ok"


async def main_async(tasks: list[str], per_label: int, workers: int) -> dict:
    from harbor.upload.db_client import UploadDB
    from harbor.upload.storage import UploadStorage
    db, storage, sem = UploadDB(), UploadStorage(), asyncio.Semaphore(workers)
    atts = [a for a in load_attempts() if a["task"] in set(tasks) and not a["error"] and a["reward"] is not None]
    chosen = []
    for t in tasks:
        task_record(t)
        mine = [a for a in atts if a["task"] == t]
        for label in (1, 0):
            chosen += round_robin([a for a in mine if (a["reward"] == 1) == (label == 1)])[:per_label]
    results = await asyncio.gather(*(fetch(a, db, storage, sem) for a in chosen))
    summary = {}
    for a, r in zip(chosen, results):
        key = r if not r.startswith("error") else "error"
        summary.setdefault(a["task"], {}).setdefault(key, 0)
        summary[a["task"]][key] += 1
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tasks", help="comma-separated; default: every training task")
    a = ap.parse_args()
    tasks = a.tasks.split(",") if a.tasks else training_tasks()
    print(json.dumps({"tasks": len(tasks)}), flush=True)
    print(json.dumps(asyncio.run(main_async(tasks, a.per_label, a.workers)), indent=1))
