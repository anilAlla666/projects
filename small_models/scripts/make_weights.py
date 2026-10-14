"""Round-2 example weights: the mined weight (3 confidently wrong, 2 wrong, else 1) times a factor for wrong
examples that pass every visible test, which are 8% of that subset and otherwise teach the model nothing.

    python scripts/make_weights.py ROOT MINED.json OUT.json [--hard-factor 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.data.load import load_examples  # noqa: E402
from vm.eval.baselines import visible_only  # noqa: E402


def combine(mined: dict, hard_wrong: set, factor: float) -> dict:
    out = dict(mined)
    for k in hard_wrong:
        out[k] = out.get(k, 1.0) * factor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("mined", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hard-factor", type=float, default=4.0)
    a = ap.parse_args()
    hard_wrong = {e["example_id"] for e in load_examples(a.root, "train") if e["label"] == 0 and visible_only(e)}
    w = combine(json.loads(a.mined.read_text()), hard_wrong, a.hard_factor)
    a.out.write_text(json.dumps(w))
    print(json.dumps({"weighted": len(w), "hard_wrong": len(hard_wrong), "max": max(w.values()) if w else None}))


if __name__ == "__main__":
    main()
