"""Score every epoch checkpoint of a fine-tuning run on dev: balanced accuracy, ECE, accept rate per attempt kind,
balanced accuracy by token length, and (when attack files exist) attack success and acceptance of correct fixes
carrying the same prose. Writes RUN_DIR/dev.jsonl, one line per epoch, and prints it.

    python scripts/dev_eval.py RUN_DIR ROOT [--split dev]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.metrics import balanced_accuracy, ece  # noqa: E402

LENGTHS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30))


def summarise(rows: list, scores: dict, attacks: list, att_scores: dict, visible_pass: dict | None = None) -> dict:
    lab = [r["label"] for r in rows]
    p = [scores[r["example_id"]]["p"] for r in rows]
    pred = [int(x >= 0.5) for x in p]
    out = {"examples": len(rows), "ba": round(balanced_accuracy(lab, pred), 4), "ece": round(ece(lab, p), 4)}
    if visible_pass:
        idx = [i for i, r in enumerate(rows) if visible_pass.get(r["example_id"])]
        hl = [lab[i] for i in idx]
        if len(set(hl)) == 2:  # the hard subset: every visible test passes, so running them is at chance here
            out["hard"] = {"n": len(idx), "wrong": hl.count(0), "ba": round(balanced_accuracy(hl, [pred[i] for i in idx]), 4),
                           "wrong_rejected": round(sum(1 - pred[i] for i in idx if lab[i] == 0) / hl.count(0), 4),
                           "correct_accepted": round(sum(pred[i] for i in idx if lab[i] == 1) / hl.count(1), 4)}
    kinds = {}
    for r, y in zip(rows, pred):
        kinds.setdefault(f"{r['attempt_kind']}:{r['label']}", []).append(y)
    out["accept"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(kinds.items())}
    by_len = {}
    for lo, hi in LENGTHS:
        idx = [i for i, r in enumerate(rows) if lo <= len(r["tokens"]) < hi]
        if idx and len({lab[i] for i in idx}) == 2:
            by_len[f"{lo}-{hi if hi < 1 << 30 else 'max'}"] = round(balanced_accuracy([lab[i] for i in idx], [pred[i] for i in idx]), 4)
    out["ba_by_length"] = by_len
    if attacks:
        w = [att_scores[r["example_id"]]["p"] >= 0.5 for r in attacks if r["label"] == 0]
        c = [att_scores[r["example_id"]]["p"] >= 0.5 for r in attacks if r["label"] == 1]
        out["attack_success"] = round(sum(w) / max(1, len(w)), 4)
        out["prose_on_correct_accepted"] = round(sum(c) / max(1, len(c)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    rows = JudgeDataset(a.root / f"serialised_{a.split}.jsonl").rows
    att_path = a.root / f"serialised_attack_{a.split}.jsonl"
    attacks = JudgeDataset(att_path).rows if att_path.is_file() else []
    from vm.data.load import load_examples
    from vm.eval.baselines import visible_only
    vis = {e["example_id"]: bool(visible_only(e)) for e in load_examples(Path.home() / "judge" / "libs", a.split)}
    lines = []
    for ck in sorted(a.run_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        judge, heads, _ = load_judge(ck)
        s = {x["example_id"]: x for x in score(judge, heads, rows)}
        t = {x["example_id"]: x for x in score(judge, heads, attacks)} if attacks else {}
        line = {"checkpoint": ck.name, **summarise(rows, s, attacks, t, vis)}
        lines.append(line)
        print(json.dumps(line), flush=True)
        del judge, heads
    (a.run_dir / f"{a.split}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))


if __name__ == "__main__":
    main()
