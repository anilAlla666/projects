"""Hard-negative mining: score the training split with a round's judge and weight what it got wrong, 3 when it was
confidently wrong (|p - label| > 0.8) and 2 otherwise. Also reports dev attack success when an attack file is given.

    python scripts/mine.py CKPT SERIALISED_TRAIN.jsonl OUT_WEIGHTS.json [--attack-dev FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def weights_for(scored: list, labels: dict) -> dict:
    out = {}
    for s in scored:
        miss = abs(s["p"] - labels[s["example_id"]])
        if miss > 0.8:
            out[s["example_id"]] = 3.0
        elif miss > 0.5:
            out[s["example_id"]] = 2.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("train", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--attack-dev", type=Path)
    a = ap.parse_args()
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    judge, heads, _ = load_judge(a.ckpt)
    rows = JudgeDataset(a.train).rows
    labels = {r["example_id"]: r["label"] for r in rows}
    w = weights_for(score(judge, heads, rows), labels)
    a.out.write_text(json.dumps(w))
    report = {"train": len(rows), "weight3": sum(v == 3.0 for v in w.values()), "weight2": sum(v == 2.0 for v in w.values())}
    if a.attack_dev:
        att = JudgeDataset(a.attack_dev).rows
        lab = {r["example_id"]: r["label"] for r in att}
        sc = score(judge, heads, att)
        wrong = [s for s in sc if lab[s["example_id"]] == 0]
        right = [s for s in sc if lab[s["example_id"]] == 1]
        report["dev_attack_success"] = sum(s["p"] >= 0.5 for s in wrong) / max(1, len(wrong))
        report["dev_prose_on_correct_accepted"] = sum(s["p"] >= 0.5 for s in right) / max(1, len(right))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
