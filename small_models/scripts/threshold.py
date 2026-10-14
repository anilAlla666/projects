"""Decision-threshold analysis on dev for one checkpoint: balanced accuracy overall and on the hard subset (every
visible test passes) across thresholds, and the AUC of each. Dev only; the test split is never used to choose.

    python scripts/threshold.py CKPT ROOT OUT.json [DATA_DIR]

Raw examples (for the hard-subset map) come from ROOT under $VM_SETTING; the serialised dev rows come from DATA_DIR
(default ROOT).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.data.load import load_examples  # noqa: E402
from vm.eval.baselines import visible_only  # noqa: E402
from vm.eval.metrics import balanced_accuracy  # noqa: E402


def auc(labels, scores):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(pos + neg)
    rank = {}
    for i, s in enumerate(order):
        rank.setdefault(s, []).append(i + 1)
    avg = {s: sum(r) / len(r) for s, r in rank.items()}
    return (sum(avg[s] for s in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ckpt, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    data = Path(sys.argv[4]) if len(sys.argv) > 4 else root
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    rows = JudgeDataset(data / "serialised_dev.jsonl").rows
    vis = {e["example_id"]: bool(visible_only(e)) for e in load_examples(root, "dev")}
    judge, heads, _ = load_judge(ckpt)
    p = {x["example_id"]: x["p"] for x in score(judge, heads, rows)}
    lab = [r["label"] for r in rows]
    sc = [p[r["example_id"]] for r in rows]
    hard = [i for i, r in enumerate(rows) if vis.get(r["example_id"])]
    att_path = data / "serialised_attack_dev.jsonl"
    att = JudgeDataset(att_path).rows if att_path.is_file() else []
    ap = {x["example_id"]: x["p"] for x in score(judge, heads, att)} if att else {}
    res = {"auc": round(auc(lab, sc), 4), "auc_hard": round(auc([lab[i] for i in hard], [sc[i] for i in hard]), 4), "thresholds": []}
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = [int(s >= t) for s in sc]
        aw = [ap[r["example_id"]] >= t for r in att if r["label"] == 0]
        ac = [ap[r["example_id"]] >= t for r in att if r["label"] == 1]
        res["thresholds"].append({"t": t, "ba": round(balanced_accuracy(lab, pred), 4),
                                  "hard_ba": round(balanced_accuracy([lab[i] for i in hard], [pred[i] for i in hard]), 4),
                                  "attack_success": round(sum(aw) / max(1, len(aw)), 4), "prose_correct_accepted": round(sum(ac) / max(1, len(ac)), 4)})
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
