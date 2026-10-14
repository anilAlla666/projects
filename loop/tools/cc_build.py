"""Step 18: build test-writing cases from CodeContests (deepmind/code_contests, CC-BY-4.0).

For a problem with at least 4 correct and 4 incorrect Python 3 solutions and 5 official tests: run up to 5 correct and
8 incorrect solutions on up to 15 official tests in a no-network container (tools/cc_harness.py). Keep a correct
solution if it matches the expected output on every test (whitespace-insensitive), an incorrect one if it fails at
least one. Keep the problem if at least 3 of each survive. One case: the statement, a reference solution, two
unlabelled candidates (one correct, one incorrect, shuffled), and the hidden pool (the other verified solutions).

    /root/cc/bin/python cc_build.py --data /root/ccdata/data --out /root/cccases --want 2000 --workers 6
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

HARNESS = Path(__file__).with_name("cc_harness.py")
IMAGE = "python:3.12-slim"


def norm(s: str | None) -> list[str] | None:
    return None if s is None else s.split()


def run(solutions: dict, inputs: list[str], timeout: float = 4.0) -> dict:
    """Every solution on every input, in one throwaway container with no network."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "job.json").write_text(json.dumps({"solutions": solutions, "inputs": inputs, "timeout": timeout}))
        Path(d, "h.py").write_text(HARNESS.read_text())
        budget = 60 + timeout * len(inputs) * len(solutions)
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "2g", "--cpus", "1",
                            "--pids-limit", "256", "-v", f"{d}:/w:ro", IMAGE, "python3", "/w/h.py"],
                           capture_output=True, text=True, timeout=budget, errors="replace")
    return json.loads(r.stdout.strip().splitlines()[-1])


def build(p: dict, rng: random.Random) -> dict | None:
    good = [s for l, s in zip(p["solutions"]["language"], p["solutions"]["solution"]) if l == 3]
    bad = [s for l, s in zip(p["incorrect_solutions"]["language"], p["incorrect_solutions"]["solution"]) if l == 3]
    tin = p["private_tests"]["input"] + p["generated_tests"]["input"]
    tout = p["private_tests"]["output"] + p["generated_tests"]["output"]
    if len(good) < 4 or len(bad) < 4 or len(tin) < 5:
        return None
    idx = list(range(len(tin)))
    rng.shuffle(idx)
    idx = idx[:15]
    tin, tout = [tin[i] for i in idx], [tout[i] for i in idx]
    rng.shuffle(good)
    rng.shuffle(bad)
    sols = {f"c{i}": s for i, s in enumerate(good[:5])} | {f"w{i}": s for i, s in enumerate(bad[:8])}
    res = run(sols, tin, timeout=2.0)
    exp = [norm(o) for o in tout]
    ok_good = [k for k in sols if k[0] == "c" and all(norm(x) == e for x, e in zip(res[k], exp))]
    ok_bad = [k for k in sols if k[0] == "w" and any(norm(x) != e for x, e in zip(res[k], exp))]
    if len(ok_good) < 3 or len(ok_bad) < 3:
        return None
    ref, vis_c = ok_good[0], ok_good[1]
    vis_w = ok_bad[0]
    visible = [sols[vis_c], sols[vis_w]]
    order = [0, 1]
    rng.shuffle(order)
    return {"name": p["name"], "description": p["description"], "difficulty": p["difficulty"],
            "reference": sols[ref], "visible": [visible[i] for i in order],
            "visible_labels": [["correct", "incorrect"][i] for i in order],
            "hidden_correct": [sols[k] for k in ok_good[2:]], "hidden_incorrect": [sols[k] for k in ok_bad[1:]],
            "official_inputs": tin[:10]}


def key(name: str) -> str:
    """A stable id (Python's hash() is salted per process, so it cannot name files that must survive a restart)."""
    return hashlib.md5(name.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--want", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    files = sorted(glob.glob(str(a.data / "train-*.parquet")))
    rng.shuffle(files)
    kept = len(list(a.out.glob("*.json")))
    with ThreadPoolExecutor(a.workers) as ex:
        for f in files:
            if kept >= a.want:
                break
            rows = pq.read_table(f).to_pylist()
            futs = {ex.submit(build, r, random.Random(int(key(r["name"]), 16))): r["name"] for r in rows
                    if not (a.out / f"{key(r['name'])}.json").exists()}
            for fu in as_completed(futs):
                try:
                    c = fu.result()
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"error": futs[fu], "why": str(e)[:200]}), flush=True)
                    continue
                if c and kept < a.want:
                    (a.out / f"{key(c['name'])}.json").write_text(json.dumps(c))
                    kept += 1
                    print(json.dumps({"kept": kept, "name": c["name"], "hidden": [len(c["hidden_correct"]),
                                                                                  len(c["hidden_incorrect"])]}),
                          flush=True)


if __name__ == "__main__":
    main()
