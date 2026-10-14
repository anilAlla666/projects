"""Step 8 evaluation of the verdict model, from its scored files (gpu/score_verdict.py) and the recorded loop runs.

1. Threshold: the P(PASS) cut that maximises balanced accuracy on the validation tasks (training tasks only).
2. The loop's question, on the 45 outside candidates with true labels on file: accuracy of the trained model, the
   untuned model, always pass, always fail, and Opus (every recorded round-0 correct/wrong verdict on them).
3. As a grader by itself: balanced accuracy, TPR and TNR per evaluation pool.

    python tools/verdict_eval.py --scores runs/verdict --keys data/candidate_keys
"""
import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).is_file() else []


def ba(rows, thr):
    pos = [r for r in rows if r["label"] == "pass"]
    neg = [r for r in rows if r["label"] == "fail"]
    tpr = sum(r["p_pass"] >= thr for r in pos) / len(pos) if pos else None
    tnr = sum(r["p_pass"] < thr for r in neg) / len(neg) if neg else None
    b = None if tpr is None or tnr is None else (tpr + tnr) / 2
    return b, tpr, tnr


def threshold(val):
    cuts = sorted({round(r["p_pass"], 6) for r in val}) + [1.01]
    return max(cuts, key=lambda c: (ba(val, c)[0] or 0, -abs(c - 0.5)))


def opus_verdicts(keys):
    """task/letter -> list of Opus's round-0 verdicts (True = judged correct) across every recorded loop run."""
    out = collections.defaultdict(list)
    for f in glob.glob(str(ROOT / "runs" / "**" / "result.json"), recursive=True):
        r = json.loads(Path(f).read_text())
        if "rounds" not in r or not r["rounds"] or r.get("task") not in keys:
            continue
        v = r["rounds"][0].get("verdict") or {}
        ids = r["rounds"][0].get("evidence_ids") or []
        if not v or not any(len(i) == 1 for i in ids):
            continue
        good, bad = set(v.get("correct_candidates") or []), set(v.get("wrong_candidates") or [])
        for k in keys[r["task"]]:
            if k in good or k in bad:
                out[f"{r['task']}/{k}"].append(k in good)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "runs" / "verdict")
    ap.add_argument("--keys", type=Path, default=ROOT / "data" / "candidate_keys")
    a = ap.parse_args()
    keys = {f.stem: {k["letter"]: k["label"] for k in json.loads(f.read_text())} for f in a.keys.glob("*.json")}
    res = {}
    opus_ids = set(opus_verdicts(keys))
    for arm in ("base", "trained"):
        val, cand, pools = (load(a.scores / f"{arm}_{s}.jsonl") for s in ("val", "candidates", "eval_pools"))
        if not val:
            continue
        thr = threshold(val)
        acc = sum((r["p_pass"] >= thr) == (r["label"] == "pass") for r in cand) / len(cand) if cand else None
        covered = [r for r in cand if r["id"] in opus_ids]  # the same candidates Opus judged
        acc_cov = sum((r["p_pass"] >= thr) == (r["label"] == "pass") for r in covered) / len(covered) if covered else None
        per = {}
        for t in sorted({r["task"] for r in pools}):
            b, tpr, tnr = ba([r for r in pools if r["task"] == t], thr)
            per[t] = {"ba": b, "tpr": tpr, "tnr": tnr}
        res[arm] = {"threshold": thr, "val_ba": ba(val, thr)[0], "candidates_acc": acc, "candidates_n": len(cand),
                    "candidates_acc_on_opus_set": acc_cov, "opus_set_n": len(covered),
                    "pools": per}
    cand_ids = [f"{t}/{k}" for t, ks in keys.items() for k in ks]
    truth = {i: keys[i.split("/")[0]][i.split("/")[1]] == "pass" for i in cand_ids}
    ov = opus_verdicts(keys)
    judged = [(i, v) for i, vs in ov.items() for v in vs]
    res["opus"] = {"verdicts": len(judged), "candidates_covered": len(ov),
                   "acc": sum(v == truth[i] for i, v in judged) / len(judged) if judged else None}
    res["always_pass"] = {"acc": sum(truth.values()) / len(truth)}
    res["always_fail"] = {"acc": 1 - sum(truth.values()) / len(truth)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
