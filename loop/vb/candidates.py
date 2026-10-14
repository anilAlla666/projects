"""Other models' attempts, for the cross-model experiment (docs/research_intent_goal.md, round 4).

For each task: the five models with the most leaderboard attempts outside the scoring pool, one attempt each (the
first by leaderboard rank and trial id). Correctness is left as it falls and never shown to the writer; the attempts
are named A to E, without their model. Their labels and models go to a key file outside anything a writer sees.
"""
from __future__ import annotations

import argparse
import collections
import shutil
import tomllib
from pathlib import Path

from .index import load_attempts
from .pool import _download, load_manifest
from .util import DATA, TB_ROOT, write_json

CANDIDATES = DATA / "candidates"
N = 5


def choose(task: str) -> list[dict]:
    pool = {e["trial_id"] for e in load_manifest(task) if e.get("trial_id")}
    cand = [a for a in load_attempts() if a["task"] == task and not a["error"] and a["reward"] is not None
            and a["trial_id"] not in pool]
    by_model = collections.defaultdict(list)
    for a in sorted(cand, key=lambda a: (a["rank"], a["trial_id"])):
        by_model[a["model"]].append(a)
    models = sorted(by_model, key=lambda m: (-len(by_model[m]), m))[:N]
    return [by_model[m][0] for m in models]


def build(task: str) -> Path:
    out = CANDIDATES / task
    if out.exists():
        shutil.rmtree(out)
    (out / "_dl").mkdir(parents=True)
    artifacts = tomllib.loads((TB_ROOT / task / "task.toml").read_text(encoding="utf-8")).get("artifacts", [])
    key = []
    for letter, a in zip("ABCDE", choose(task)):
        d = _download(a["trial_id"], out / "_dl")
        for p in artifacts:
            p = p if isinstance(p, str) else p["source"]
            s = d / "artifacts" / p.lstrip("/")
            if s.exists():
                t = out / "files" / letter / p.lstrip("/")
                t.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if s.is_dir() else shutil.copy2)(s, t)
        key.append({"letter": letter, "model": a["model"], "trial_id": a["trial_id"],
                    "label": "pass" if float(a["reward"]) >= 1.0 else "fail"})
    shutil.rmtree(out / "_dl")
    write_json(DATA / "candidate_keys" / f"{task}.json", key)
    return out / "files"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    print(build(ap.parse_args().task))
