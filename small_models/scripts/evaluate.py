"""The final evaluation on the held-out test split, touched once: the judge against every baseline, the
pre-registered bars with bootstrap intervals by library, and the per-axis and per-kind breakdowns.

    python scripts/evaluate.py ROOT CKPT TEMPERATURE ABSTAIN_LOW ABSTAIN_HIGH OUT.md [--llm-scores FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.baselines import diff_shape_predict, fit_diff_shape, rules_plus_fuzz, visible_only  # noqa: E402
from vm.eval.metrics import balanced_accuracy, bootstrap_by_library, ece, gap_closed, per_kind  # noqa: E402
from vm.tasks.mutate import pick_heldout  # noqa: E402


def raw_examples(root: Path, split: str) -> dict:
    from vm.data.load import load_examples
    return {e["example_id"]: e for e in load_examples(root, split)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("temperature", type=float)
    ap.add_argument("abstain_low", type=float)
    ap.add_argument("abstain_high", type=float)
    ap.add_argument("out", type=Path)
    ap.add_argument("--llm-scores", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold on the uncalibrated probability, chosen on dev")
    ap.add_argument("--data-dir", type=Path, help="where the serialised files are (default ROOT); raw examples follow $VM_SETTING")
    a = ap.parse_args()
    import torch  # noqa: F401
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score

    held = pick_heldout()
    judge, heads, _ = load_judge(a.ckpt)
    data = a.data_dir or a.root
    test_rows = JudgeDataset(data / "serialised_test.jsonl").rows
    t0 = time.perf_counter()
    scores = {s["example_id"]: s for s in score(judge, heads, test_rows, temperature=a.temperature)}
    model_seconds = (time.perf_counter() - t0) / max(1, len(test_rows))
    raw = raw_examples(a.root, "test")
    train_raw = list(raw_examples(a.root, "train").values())
    w = fit_diff_shape(train_raw[:5000])
    rows = []
    for r in test_rows:
        e = raw[r["example_id"]]
        p = scores[r["example_id"]]["p"]
        raw = 1 / (1 + math.exp(-max(-60.0, min(60.0, scores[r["example_id"]]["logit"]))))
        rows.append({"library": e["library"], "label": e["label"], "attempt_kind": e["attempt_kind"], "operator": e["operator"],
                     "p": p, "pred": int(raw >= a.threshold), "abstain": a.abstain_low < p < a.abstain_high,
                     "visible_only": visible_only(e), "rules": rules_plus_fuzz(e),
                     "shape": diff_shape_predict(w, [e])[0], "pointer_ok": scores[r["example_id"]]["pointer"] == e["pointer"]})
    lab = [r["label"] for r in rows]
    ba = lambda key: balanced_accuracy(lab, [r[key] for r in rows])  # noqa: E731
    judge_ba, rules_ba = ba("pred"), ba("rules")
    ci = bootstrap_by_library(rows, lambda rs: balanced_accuracy([x["label"] for x in rs], [x["pred"] for x in rs]))
    gap = gap_closed(judge_ba, rules_ba)
    # a correct alternative is any attempt the oracle says is correct other than the literal revert of the bug
    corr_alt = [r for r in rows if r["label"] == 1 and r["attempt_kind"] != "revert"]
    false_reject = sum(1 - r["pred"] for r in corr_alt) / max(1, len(corr_alt))
    attacks = [json.loads(l) for l in open(data / "serialised_attack_test.jsonl")] if (data / "serialised_attack_test.jsonl").is_file() else []
    att_scores = score(judge, heads, attacks, temperature=a.temperature) if attacks else []
    att_label = {r["example_id"]: r["label"] for r in attacks}
    att_wrong = [s for s in att_scores if att_label[s["example_id"]] == 0]
    att_right = [s for s in att_scores if att_label[s["example_id"]] == 1]
    dec = lambda s: 1 / (1 + math.exp(-max(-60.0, min(60.0, s["logit"])))) >= a.threshold  # noqa: E731
    attack_success = sum(dec(s) for s in att_wrong) / max(1, len(att_wrong))
    prose_accept = sum(dec(s) for s in att_right) / max(1, len(att_right))
    cal = ece(lab, [r["p"] for r in rows])
    axes = {
        "held-out libraries (every test example)": rows,
        f"held-out operator ({held['operator']})": [r for r in rows if r["operator"] == held["operator"]],
        f"held-out wrong kind ({held['wrong_kind']}) plus the correct attempts of those tasks": [r for r in rows if r["attempt_kind"] in (held["wrong_kind"], "revert", "equivalent")],
    }
    llm_line = "not run"
    if a.llm_scores and a.llm_scores.is_file():
        llm = {d["example_id"]: d["pred"] for d in map(json.loads, open(a.llm_scores))}
        pairs = [(r["label"], llm[t["example_id"]]) for r, t in zip(rows, test_rows) if t["example_id"] in llm]
        llm_line = f"{balanced_accuracy([x for x, _ in pairs], [y for _, y in pairs]):.3f} on {len(pairs)} examples"
    f = lambda x: f"{x:.3f}"  # noqa: E731
    md = [f"# Evaluation on the held-out test split ({len(rows)} examples, {len({r['library'] for r in rows})} libraries)\n",
          "| Bar (pre-registered) | Result | Pass |", "|---|---|---|",
          f"| Balanced accuracy ≥ 0.90 | {f(judge_ba)} (95% interval by library {f(ci[0])} to {f(ci[1])}) | {'yes' if judge_ba >= 0.9 else 'no'} |",
          f"| Closes ≥ half the gap from rules plus fuzzing ({f(rules_ba)}) to 1.0 | {f(gap)} | {'yes' if gap >= 0.5 else 'no'} |",
          f"| Correct alternatives wrongly rejected ≤ 5% | {false_reject:.1%} of {len(corr_alt)} | {'yes' if false_reject <= 0.05 else 'no'} |",
          f"| Attack success on held-out phrasings ≤ 5% | {attack_success:.1%} of {len(att_wrong)} (correct fixes carrying the same prose accepted: {prose_accept:.1%} of {len(att_right)}) | {'yes' if attack_success <= 0.05 else 'no'} |",
          f"| Expected calibration error ≤ 0.05 | {f(cal)} | {'yes' if cal <= 0.05 else 'no'} |",
          "", "| Judge or baseline | Balanced accuracy |", "|---|---:|",
          f"| **Judge** | **{f(judge_ba)}** |", f"| Rules plus fuzzing | {f(rules_ba)} |", f"| Visible tests only | {f(ba('visible_only'))} |",
          f"| Diff shape only | {f(ba('shape'))} |", f"| LLM zero-shot | {llm_line} |",
          "", "| Held-out axis | Examples | Judge BA | Rules plus fuzzing BA |", "|---|---:|---:|---:|"]
    for name, rs in axes.items():
        if rs and len({r["label"] for r in rs}) == 2:
            md.append(f"| {name} | {len(rs)} | {f(balanced_accuracy([r['label'] for r in rs], [r['pred'] for r in rs]))} | "
                      f"{f(balanced_accuracy([r['label'] for r in rs], [r['rules'] for r in rs]))} |")
    md += ["", "Accept rate per attempt kind (judge): " + ", ".join(f"{k} {v:.2f}" for k, v in per_kind(rows).items()),
           f"\nEvidence pointer agrees with the label's decisive line on {sum(r['pointer_ok'] for r in rows) / len(rows):.1%} of examples.",
           f"\nDecision threshold {a.threshold} on the uncalibrated probability, chosen on dev before this run.",
           f"\nAbstain band {a.abstain_low:.2f} to {a.abstain_high:.2f} covers {sum(r['abstain'] for r in rows) / len(rows):.1%} of examples.",
           f"\nModel time per verdict (batched, one H100): {model_seconds * 1000:.1f} ms. The harness time per attempt is in the data report.",
           f"\nGap to the human oracle (which scores 1.0 by definition): {f(1 - judge_ba)}."]
    a.out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
