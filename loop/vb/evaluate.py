"""Grade every verifier on every kept pool, score against the labels, and check the baselines."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from .pool import kept_job, load_manifest
from .leakscan import scan
import subprocess

from .regrade import grade, infra_failures, tests_hash, verdict
from .report import render
from .scorer import Verdict, check_baselines, macro, score_task
from .taskdir import BASELINES, official_verifier_image
from .util import DATA, ROOT, TB_ROOT, new_run_dir, read_json, write_json
from .writer import ALL_WRITERS, _allowed

CACHE = DATA / "cache"


def verifiers_for(task: str) -> dict[str, Path]:
    out = {"original": TB_ROOT / task / "tests", **BASELINES}
    for d in sorted((DATA / "verifiers").glob(f"*/{task}/*/tests")):
        out[f"{d.parents[2].name}/{d.parent.name}"] = d
    return out


def leaks(task: str) -> dict[str, list[str]]:
    """{'<writer>/<trial>': findings} for this task's writer verifiers, scanned now with the current rules."""
    out = {}
    for name, tests in verifiers_for(task).items():
        if "/" in name:
            found = scan(tests, TB_ROOT / task / "tests", _allowed(task), TB_ROOT / task / "environment")
            if found:
                out[name] = found
    return out


def cached_grade(task: str, name: str, tests: Path, job: Path, work: Path, pool_key: str) -> dict:
    """Rewards for one verifier on one kept pool; cached by (pool, verifier content), so reruns only grade new work."""
    key = CACHE / task / f"{name.replace('/', '__')}__{tests_hash(tests)}__{pool_key}.json"
    if key.exists():
        return read_json(key)
    image = official_verifier_image(TB_ROOT / task) if name == "original" else None
    for attempt in range(8):
        job_name = name.replace("/", "__") + (f"__retry{attempt}" if attempt else "")
        rewards = grade(job, TB_ROOT / task, tests, work / task, job_name, image=image, n_concurrent=2)
        bad = infra_failures(work / task / "regrades" / job_name)
        if not bad:
            write_json(key, rewards)
            return rewards
        print(f"{task} {name}: {len(bad)} infrastructure failures, retrying after a network prune", flush=True)
        subprocess.run(["docker", "network", "prune", "-f"], capture_output=True)
        __import__("time").sleep(30)
    raise RuntimeError(f"{task} {name}: infrastructure failures persist after 8 attempts; not scored")


def no_verifiers(task: str) -> list[str]:
    """'<writer>/<trial>' for writer attempts on this task that ended without a usable verifier for a reason of the
    writer's own (e.g. timing out). Scored as rejecting everything (every verdict an error), and listed. Attempts
    killed by our infrastructure (Docker compose failures) were rerun and are not counted."""
    out = []
    for f in (DATA / "verifiers").glob("*/collect_*.json"):
        for r in read_json(f):
            msg = ((r.get("exception") or {}).get("exception_message") or "")
            ours = ("Docker compose command failed" in msg or "does not support route" in msg)   # our infra or config
            if r["task"] == task and not r["ok"] and not ours:
                out.append(f"{r['writer']}/{r['trial']}")
    return sorted(out)


def writer_costs() -> dict[str, float]:
    costs: dict[str, float] = {}
    for f in (DATA / "verifiers").glob("*/collect_*.json"):
        for r in read_json(f):
            c = (r.get("agent_result") or {}).get("cost_usd")
            if c is not None:
                costs[r["writer"]] = costs.get(r["writer"], 0.0) + float(c)
    return costs


BASELINE_SAMPLE = 2        # correct and wrong entries each: enough to prove the swap and the offline rule


def probe_job(task: str, kept: list[dict]) -> tuple[Path, list[dict]]:
    """A job dir with the first BASELINE_SAMPLE correct and wrong kept entries, for the trivial baselines."""
    import shutil
    from .pool import _job_config, pool_dir
    sample = [e for e in kept if e["label"] == "pass"][:BASELINE_SAMPLE] + \
             [e for e in kept if e["label"] == "fail"][:BASELINE_SAMPLE]
    pj = pool_dir(task) / "probe_job"
    if pj.exists():
        shutil.rmtree(pj)
    pj.mkdir()
    for e in sample:
        shutil.copytree(pool_dir(task) / "job" / e["entry_id"], pj / e["entry_id"], symlinks=True)
    _job_config(pj, task)
    return pj, sample


def evaluate_task(task: str, run_dir: Path) -> tuple[dict, list[dict], dict]:
    kept = [e for e in load_manifest(task) if e["kept"]]
    pool_key = __import__("hashlib").sha256(",".join(sorted(e["entry_id"] for e in kept)).encode()).hexdigest()[:12]
    job = kept_job(task)
    pjob, sample = probe_job(task, kept)
    sample_key = __import__("hashlib").sha256(",".join(sorted(e["entry_id"] for e in sample)).encode()).hexdigest()[:12]
    per, rows = {}, []
    for name, tests in verifiers_for(task).items():
        trivial = name in BASELINES
        entries = sample if trivial else kept
        rewards = cached_grade(task, name, tests, pjob if trivial else job, run_dir, sample_key if trivial else pool_key)
        verdicts = [Verdict(e["entry_id"], e["source"], e["label"], verdict(rewards.get(e["entry_id"]))) for e in entries]
        per[name] = score_task(verdicts)
        rows += [{"task": task, "verifier": name, "entry_id": v.entry_id, "source": v.source, "label": v.label,
                  "verdict": v.verdict, "reward": rewards.get(v.entry_id)} for v in verdicts]
        print(task, name, {k: per[name][k] for k in ("tpr", "tnr", "ba", "flakiness")}, flush=True)
    for name in no_verifiers(task):
        verdicts = [Verdict(e["entry_id"], e["source"], e["label"], "error") for e in kept]
        per[name] = {**score_task(verdicts), "no_verifier": True}
        rows += [{"task": task, "verifier": name, "entry_id": v.entry_id, "source": v.source, "label": v.label,
                  "verdict": "error", "reward": None} for v in verdicts]
    return per, rows, leaks(task)


def _task_mean(scores: list[dict]) -> dict:
    """One writer's attempts on one task, averaged, so a task with two attempts does not count twice."""
    def avg(k):
        vals = [s[k] for s in scores if s.get(k) is not None]
        return sum(vals) / len(vals) if vals else None
    return {k: avg(k) for k in ("tpr", "tnr", "ba", "flakiness")}


def evaluate(tasks: list[str], run_dir: Path, workers: int = 5) -> dict:
    per_task, rows, by_writer, flagged = {}, [], {}, {}
    with cf.ThreadPoolExecutor(workers) as ex:
        results = dict(zip(tasks, ex.map(lambda t: evaluate_task(t, run_dir), tasks)))
    for task in tasks:
        per, r, lk = results[task]
        per_task[task], rows, flagged = per, rows + r, {**flagged, **lk}
        for name, s in per.items():
            if "/" in name and name in lk:
                s["leak"] = lk[name]                       # scored and shown, never counted in a writer's macro
            elif "/" in name:
                by_writer.setdefault(name.split("/")[0], {}).setdefault(task, []).append(s)
    problems = []
    for task, per in per_task.items():
        problems += [f"{task}: {p}" for p in check_baselines(per["always_pass"], per["always_fail"], per["original"],
                                                             per["network_probe"])]
    scores = {"name": run_dir.name, "tasks": tasks, "per_task": per_task,
              "macro": {w: macro([_task_mean(ss) for ss in per_t.values()]) for w, per_t in by_writer.items()},
              "baseline_problems": problems,
              "writer_cost_usd": writer_costs(), "leaks": flagged,
              "writers": {n: {"model_id": w["model_id"], "harness": w["harness"]} for n, w in ALL_WRITERS.items()}}
    write_json(run_dir / "scores.json", scores)
    (run_dir / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "summary.md").write_text(render(scores), encoding="utf-8")
    if problems:
        (run_dir / "BASELINE_FAILED.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        raise SystemExit(1)
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t] or read_json(ROOT / "configs" / "m1.json")["tasks"]
    run_dir = new_run_dir("m1_eval")
    evaluate(tasks, run_dir)
    print(run_dir)
