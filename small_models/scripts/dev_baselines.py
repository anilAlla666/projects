"""Before any fine-tuning: the free baselines on a split, label balance per attempt kind, and the token-length
distribution of the serialised rows. Tells us the bar the judge must clear and whether the construction leaks the
label through something trivial like the diff's shape.

    python scripts/dev_baselines.py ROOT [--split dev] [--real FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.baselines import diff_shape_predict, fit_diff_shape, rules_plus_fuzz, visible_only  # noqa: E402
from vm.eval.metrics import balanced_accuracy  # noqa: E402


def load(root: Path, split: str) -> list:
    from vm.data.load import load_examples
    return load_examples(root, split)


def report(rows: list, w, name: str) -> dict:
    lab = [e["label"] for e in rows]
    out = {"set": name, "examples": len(rows), "correct": sum(lab), "libraries": len({e["library"] for e in rows}),
           "visible_only": balanced_accuracy(lab, [visible_only(e) for e in rows]),
           "rules_plus_fuzz": balanced_accuracy(lab, [rules_plus_fuzz(e) for e in rows]),
           "diff_shape": balanced_accuracy(lab, diff_shape_predict(w, rows))}
    kinds = Counter((e["attempt_kind"], e["label"]) for e in rows)
    out["kinds"] = {k: {"correct": kinds[(k, 1)], "wrong": kinds[(k, 0)]} for k in sorted({k for k, _ in kinds})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--real", type=Path)
    a = ap.parse_args()
    train = load(a.root, "train")
    w = fit_diff_shape(train[:5000])
    res = [report(load(a.root, a.split), w, a.split)]
    if a.real and a.real.is_file():
        res.append(report([json.loads(l) for l in a.real.read_text().splitlines()], w, "real"))
    ser = a.root / f"serialised_{a.split}.jsonl"
    if ser.is_file():
        n = sorted(len(json.loads(l)["tokens"]) for l in ser.read_text().splitlines())
        q = lambda f: n[min(len(n) - 1, int(f * len(n)))]  # noqa: E731
        res[0]["tokens"] = {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": n[-1],
                            "over_2048": sum(x > 2048 for x in n) / len(n)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
