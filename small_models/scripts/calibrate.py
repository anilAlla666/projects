"""Calibration on dev: the temperature that minimises negative log-likelihood (0.5 to 3.0 in steps of 0.05), and the
smallest symmetric abstain band around 0.5 outside which the judge's verdicts are at least 95% accurate.

    python scripts/calibrate.py CKPT SERIALISED_DEV.jsonl OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, x))))


def fit_temperature(logits: list, labels: list) -> float:
    def nll(t):
        return -sum(math.log(max(1e-12, sigmoid(z / t) if y else 1 - sigmoid(z / t))) for z, y in zip(logits, labels)) / len(labels)
    grid = [round(0.5 + 0.05 * i, 2) for i in range(51)]
    return min(grid, key=nll)


def abstain_band(probs: list, labels: list, target: float = 0.95) -> tuple:
    for k in range(51):
        d = k / 100
        kept = [(p, y) for p, y in zip(probs, labels) if not (0.5 - d < p < 0.5 + d)]
        if kept and sum(int(p >= 0.5) == y for p, y in kept) / len(kept) >= target:
            return 0.5 - d, 0.5 + d, len(kept) / len(probs)
    return 0.0, 1.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("dev", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    from vm.eval.metrics import ece
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    judge, heads, _ = load_judge(a.ckpt)
    rows = JudgeDataset(a.dev).rows
    lab = {r["example_id"]: r["label"] for r in rows}
    s = score(judge, heads, rows)
    logits, labels = [x["logit"] for x in s], [lab[x["example_id"]] for x in s]
    t = fit_temperature(logits, labels)
    probs = [sigmoid(z / t) for z in logits]
    lo, hi, cover = abstain_band(probs, labels)
    res = {"temperature": t, "abstain_low": lo, "abstain_high": hi, "decided_share": cover,
           "dev_ece_before": ece(labels, [sigmoid(z) for z in logits]), "dev_ece_after": ece(labels, probs), "dev": len(rows)}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
