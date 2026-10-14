"""Adaptation curve: how much a fine-tuned judge improves on a library it has never seen when given k labelled
examples from that library (k = 0, 10, 50). Per library: the examples of its first tasks (by task id) are the
adaptation pool, the examples of its remaining tasks are the evaluation set, so no task appears on both sides.
Each k is a fresh copy of the judge, trained for a fixed number of steps on k class-balanced examples.

    python scripts/adapt.py CKPT SERIALISED.jsonl OUT.json [--ks 0,10,50] [--libraries N] [--steps 30]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.metrics import balanced_accuracy  # noqa: E402


def task_of(example_id: str) -> str:
    return example_id.split("/")[0]


def split_library(rows: list, seed: int = 0):
    """(adaptation pool, evaluation rows): the first half of the library's tasks, sorted by id, adapt; the rest evaluate."""
    tasks = sorted({task_of(r["example_id"]) for r in rows})
    pool_tasks = set(tasks[: len(tasks) // 2])
    return [r for r in rows if task_of(r["example_id"]) in pool_tasks], [r for r in rows if task_of(r["example_id"]) not in pool_tasks]


def balanced_sample(rows: list, k: int, seed: int) -> list:
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[: k // 2] + neg[: k - k // 2]


def adapt_and_score(judge, heads, train_rows, eval_rows, steps: int, lr: float = 2e-5):
    import torch
    from vmt.judge_data import _collate, judge_losses
    from vmt.score import score
    if train_rows:
        j, h = copy.deepcopy(judge), copy.deepcopy(heads)
        j.train()
        h.train()
        opt = torch.optim.AdamW(list(j.parameters()) + list(h.parameters()), lr=lr, weight_decay=0.0)
        rng = random.Random(0)
        for _ in range(steps):
            b = rng.sample(train_rows, min(8, len(train_rows)))
            loss, _ = judge_losses(j, h, _collate([{**r, "weight": 1.0} for r in b], "cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(j.parameters()) + list(h.parameters()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    else:
        j, h = judge, heads
    s = {x["example_id"]: x["p"] for x in score(j, h, eval_rows)}
    return balanced_accuracy([r["label"] for r in eval_rows], [int(s[r["example_id"]] >= 0.5) for r in eval_rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ks", default="0,10,50")
    ap.add_argument("--libraries", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    judge, heads, _ = load_judge(a.ckpt)
    rows = JudgeDataset(a.data).rows
    by_lib = {}
    for r in rows:
        by_lib.setdefault(r["library"], []).append(r)
    ks = [int(k) for k in a.ks.split(",")]
    results = []
    for lib in sorted(by_lib, key=lambda l: -len(by_lib[l]))[: a.libraries]:
        pool, ev = split_library(by_lib[lib])
        if len({r["label"] for r in ev}) < 2 or len({r["label"] for r in pool}) < 2:
            continue
        line = {"library": lib, "pool": len(pool), "eval": len(ev)}
        for k in ks:
            if k > len(pool):
                continue
            line[f"k{k}"] = round(adapt_and_score(judge, heads, balanced_sample(pool, k, 0) if k else [], ev, a.steps), 4)
        results.append(line)
        print(json.dumps(line), flush=True)
    mean = {f"k{k}": round(sum(r[f"k{k}"] for r in results if f"k{k}" in r) / max(1, sum(f"k{k}" in r for r in results)), 4) for k in ks}
    a.out.write_text(json.dumps({"per_library": results, "mean": mean, "steps": a.steps}, indent=1))
    print(json.dumps({"mean": mean}))


if __name__ == "__main__":
    main()
