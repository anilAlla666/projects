"""Score the Verifier Bench rows: per task and macro, AUC, balanced accuracy at a threshold, and the spread of the
judge's probabilities (near-constant scores mean no signal, whatever the balanced accuracy says).

    python scripts/vb_score.py CKPT ROWS.jsonl THRESHOLD
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from threshold import auc  # noqa: E402
from vm.eval.metrics import balanced_accuracy  # noqa: E402


def main():
    ckpt, rows_path, t = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    judge, heads, _ = load_judge(ckpt)
    rows = JudgeDataset(rows_path).rows
    p = {x["example_id"]: x["p"] for x in score(judge, heads, rows)}
    out, bas, aucs = {}, [], []
    for task in sorted({r["library"] for r in rows}):
        rs = [r for r in rows if r["library"] == task]
        lab = [r["label"] for r in rs]
        sc = [p[r["example_id"]] for r in rs]
        ba = balanced_accuracy(lab, [int(s >= t) for s in sc])
        a = auc(lab, sc)
        bas.append(ba)
        aucs.append(a)
        out[task] = {"n": len(rs), "correct": sum(lab), "auc": round(a, 3), "ba": round(ba, 3),
                     "p_min": round(min(sc), 4), "p_median": round(statistics.median(sc), 4), "p_max": round(max(sc), 4)}
    print(json.dumps({"checkpoint": ckpt.name, "rows": rows_path.name, "threshold": t, "macro_auc": round(sum(aucs) / len(aucs), 3),
                      "macro_ba": round(sum(bas) / len(bas), 3), "tasks": out}, indent=1))


if __name__ == "__main__":
    main()
