"""Pools: leaderboard attempts plus oracle and nop runs; labels checked by regrading with the original tests."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .index import load_attempts
from .regrade import grade, verdict
from .taskdir import official_verifier_image
from .util import DATA, HARBOR, TB_ROOT, read_json, write_json

N_PASS, N_FAIL, MIN_JOBS, FAIL_CANDIDATES = 10, 10, 4, 24


def pool_dir(task: str) -> Path:
    return DATA / "pools" / task


def load_manifest(task: str) -> list[dict]:
    return read_json(pool_dir(task) / "manifest.json")


def round_robin(atts: list[dict]) -> list[dict]:
    jobs: dict[str, list[dict]] = {}
    for a in sorted(atts, key=lambda a: (a["rank"], a["job"], a["trial_id"])):
        jobs.setdefault(a["job"], []).append(a)
    order = sorted(jobs, key=lambda j: (jobs[j][0]["rank"], j))
    out, i = [], 0
    while any(i < len(jobs[j]) for j in order):
        out += [jobs[j][i] for j in order if i < len(jobs[j])]
        i += 1
    return out


def ctrf_counts(trial_dir: Path) -> tuple[int, int]:
    p = Path(trial_dir) / "verifier" / "ctrf.json"
    if not p.is_file():
        return (0, 0)
    s = json.loads(p.read_text(encoding="utf-8"))["results"]["summary"]
    return (int(s.get("passed", 0)), int(s.get("tests", 0)))


def classify_fail(passed: int, total: int, threshold: float = 0.8) -> str:
    return "near_miss" if total and passed / total >= threshold else "real_fail"


def _download(trial_id: str, dest: Path) -> Path:
    tmp = dest.parent / "_dl" / trial_id
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run([HARBOR, "hub", "trial", "download", trial_id, "-o", str(tmp)], check=True, capture_output=True)
    (child,) = [p for p in tmp.iterdir() if p.is_dir()]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / child.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(child), str(target))
    shutil.rmtree(tmp)
    return target


def _run_agent(task: str, agent: str, dest: Path) -> tuple[Path, float | None]:
    runs = dest.parent / "_runs"
    subprocess.run([HARBOR, "run", "-p", str(TB_ROOT), "-i", task, "-a", agent, "-o", str(runs),
                    "--job-name", agent, "-y", "-q"], check=True, capture_output=True)
    (trial,) = [p for p in (runs / agent).iterdir() if (p / "result.json").is_file()]
    r = json.loads((trial / "result.json").read_text())
    reward = (((r.get("verifier_result") or {}).get("rewards")) or {}).get("reward")
    target = dest / f"{agent}__{trial.name}"
    shutil.copytree(trial, target)
    return target, reward


def _job_config(job_dir: Path, task: str) -> None:
    write_json(job_dir / "config.json", {"job_name": job_dir.name, "jobs_dir": str(job_dir.parent),
                                         "n_concurrent_trials": 1, "agents": [{"name": "codex"}],
                                         "datasets": [{"path": str(TB_ROOT), "task_names": [task]}]})


def build_pool(task: str) -> Path:
    root = pool_dir(task)
    job = root / "job"
    if root.exists():
        shutil.rmtree(root)
    job.mkdir(parents=True)
    atts = [a for a in load_attempts() if a["task"] == task and not a["error"] and a["reward"] is not None]
    passes = round_robin([a for a in atts if float(a["reward"]) >= 1.0])[:N_PASS]
    if len({a["job"] for a in passes}) < MIN_JOBS:
        raise RuntimeError(f"{task}: passes come from fewer than {MIN_JOBS} runs")
    entries = []

    def entry(a, d, source, label, reward):
        p, t = ctrf_counts(d)
        return {"entry_id": d.name, "trial_id": a.get("trial_id"), "source": source, "label": label,
                "recorded_reward": reward, "model": a.get("model"), "agent": a.get("agent"), "job": a.get("job"),
                "ctrf": [p, t], "regraded_reward": None, "kept": None, "drop_reason": None}

    for a in passes:
        entries.append(entry(a, _download(a["trial_id"], job), "pass", "pass", float(a["reward"])))
    cand_dir = root / "candidates"
    fails = []
    for a in round_robin([a for a in atts if float(a["reward"]) < 1.0])[:FAIL_CANDIDATES]:
        d = _download(a["trial_id"], cand_dir)
        fails.append((classify_fail(*ctrf_counts(d)) == "near_miss", a, d))
    fails.sort(key=lambda x: not x[0])                          # near misses first, stable otherwise
    for is_near, a, d in fails[:N_FAIL]:
        target = job / d.name
        shutil.move(str(d), str(target))
        entries.append(entry(a, target, "near_miss" if is_near else "real_fail", "fail", float(a["reward"])))
    shutil.rmtree(cand_dir)
    for agent, source, label in (("oracle", "reference", "pass"), ("nop", "nop", "fail")):
        d, reward = _run_agent(task, agent, job)
        entries.append(entry({"agent": agent, "model": None}, d, source, label, reward))
    _job_config(job, task)
    write_json(root / "manifest.json", entries)
    return root / "manifest.json"


def check_labels(task: str) -> dict:
    """Regrade the whole pool with the ORIGINAL tests (the official verifier image, offline); keep an entry only
    when its label reproduces."""
    root = pool_dir(task)
    entries = load_manifest(task)
    rewards = grade(root / "job", TB_ROOT / task, TB_ROOT / task / "tests", root / "label_check", "original",
                    image=official_verifier_image(TB_ROOT / task))
    counts = {"kept": 0, "dropped": 0}
    for e in entries:
        e["regraded_reward"] = rewards.get(e["entry_id"])
        got = verdict(e["regraded_reward"])
        if e["entry_id"] not in rewards:
            e["kept"], e["drop_reason"] = False, "no regrade result"
        elif got == "error":
            e["kept"], e["drop_reason"] = False, "original tests errored on regrade"
        elif e["source"] in ("reference", "nop") and verdict(e["recorded_reward"]) != e["label"]:
            e["kept"], e["drop_reason"] = False, f"{e['source']} did not score as expected on its own run"
        elif got != e["label"]:
            e["kept"], e["drop_reason"] = False, f"label {e['label']} did not reproduce (regraded {got})"
        else:
            e["kept"], e["drop_reason"] = True, None
        counts["kept" if e["kept"] else "dropped"] += 1
    write_json(root / "manifest.json", entries)
    return counts


def kept_job(task: str) -> Path:
    root = pool_dir(task)
    kj = root / "kept_job"
    if kj.exists():
        shutil.rmtree(kj)
    kj.mkdir()
    for e in load_manifest(task):
        if e["kept"]:
            shutil.copytree(root / "job" / e["entry_id"], kj / e["entry_id"], symlinks=True)
    _job_config(kj, task)
    return kj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--check", action="store_true", help="only rerun the label check")
    a = ap.parse_args()
    if not a.check:
        print(build_pool(a.task))
    print(json.dumps(check_labels(a.task)))
    print(kept_job(a.task))
