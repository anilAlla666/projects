"""The real-attempts evaluation, run once: agent patches with oracle-derived labels, the judge against every
baseline on identical rows, and the hard subset, wrong and right patches that pass every visible test, where
running the visible tests is at chance by construction.

    python scripts/evaluate_real.py REAL_DIR CKPT TEMPERATURE TRAIN_ROOT OUT.md [--llm NAME=FILE ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.baselines import diff_shape_predict, fit_diff_shape, rules_plus_fuzz, visible_only  # noqa: E402
from vm.eval.metrics import balanced_accuracy, bootstrap_by_library, ece  # noqa: E402


def rates(rows: list, key: str) -> str:
    c = [r for r in rows if r["label"] == 1]
    w = [r for r in rows if r["label"] == 0]
    acc_c = sum(r[key] for r in c) / max(1, len(c))
    rej_w = sum(1 - r[key] for r in w) / max(1, len(w))
    ba = balanced_accuracy([r["label"] for r in rows], [r[key] for r in rows]) if c and w else float("nan")
    return f"{ba:.3f} | {acc_c:.1%} of {len(c)} | {rej_w:.1%} of {len(w)}"


def raw_p(x: dict) -> float:
    import math
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, x["logit"]))))


def harness_seconds(real_dir: Path) -> list:
    out = []
    for spec in real_dir.glob("runs/*/a*/spec.json"):
        res = spec.parent / "out" / "result.json"
        if res.is_file():
            out.append(res.stat().st_mtime - spec.stat().st_mtime)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real_dir", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("temperature", type=float)
    ap.add_argument("train_root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--llm", action="append", default=[], help="NAME=FILE of llm_baseline.py output")
    ap.add_argument("--llm-attack", action="append", default=[], help="NAME=FILE, the same judge on the attacked rows")
    ap.add_argument("--setting", default="strict", choices=["strict", "realistic"])
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold on the uncalibrated probability, chosen on dev")
    a = ap.parse_args()
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    sfx = "" if a.setting == "strict" else f"_{a.setting}"
    raw = {e["example_id"]: e for e in map(json.loads, (a.real_dir / f"examples{sfx}.jsonl").read_text().splitlines())}
    ser = JudgeDataset(a.real_dir / f"serialised_real{sfx}.jsonl").rows
    judge, heads, _ = load_judge(a.ckpt)
    score(judge, heads, ser[:8], temperature=a.temperature)  # warm-up, so the timing below is steady state
    t0 = time.perf_counter()
    s = {x["example_id"]: x for x in score(judge, heads, ser, temperature=a.temperature)}
    model_ms = (time.perf_counter() - t0) / max(1, len(ser)) * 1000
    from vm.data.load import load_examples
    train = load_examples(a.train_root, "train")
    w = fit_diff_shape(train[:5000])
    llms = {}
    for spec in a.llm:
        name, path = spec.split("=", 1)
        llms[name] = {d["example_id"]: d["pred"] for d in map(json.loads, Path(path).read_text().splitlines())}
    rows = []
    for r in ser:
        e = raw[r["example_id"]]
        row = {"library": e["library"], "label": e["label"], "p": s[r["example_id"]]["p"], "judge": int(raw_p(s[r["example_id"]]) >= a.threshold),
               "visible_only": visible_only(e), "rules": rules_plus_fuzz(e), "shape": diff_shape_predict(w, [e])[0]}
        for name, m in llms.items():
            row[name] = m.get(r["example_id"])
        rows.append(row)
    hard = [r for r in rows if r["visible_only"] == 1]
    lab = [r["label"] for r in rows]
    ci = bootstrap_by_library(rows, lambda rs: balanced_accuracy([x["label"] for x in rs], [x["judge"] for x in rs]))
    md = [f"# Real attempts, {a.setting} setting ({len(rows)} agent patches, {len({r['library'] for r in rows})} repositories, "
          f"{sum(lab)} correct and {len(lab) - sum(lab)} wrong by our own oracle runs)\n",
          "Every label was re-derived by running the instance's failing-to-passing and passing-to-passing tests ourselves.",
          "The judge, every baseline and every LLM saw identical rows. The hard subset is the patches that pass every visible",
          "test: running the visible tests accepts all of them, so it is at chance there by construction.\n",
          "| Judge or baseline | All: BA, correct accepted, wrong rejected | Hard subset: BA, correct accepted, wrong rejected |",
          "|---|---|---|"]
    for key, name in [("judge", "**Judge (116M)**"), ("visible_only", "Visible tests only"), ("rules", "Rules plus fuzzing"), ("shape", "Diff shape only")] + \
            [(n, f"LLM zero-shot: {n}") for n in llms]:
        rs, hs = [r for r in rows if r.get(key) is not None], [r for r in hard if r.get(key) is not None]
        md.append(f"| {name} | {rates(rs, key)} | {rates(hs, key)} |")
    hl = [r["label"] for r in hard]
    md += ["", f"Judge balanced accuracy on all rows: 95% interval by repository {ci[0]:.3f} to {ci[1]:.3f}. ECE {ece(lab, [r['p'] for r in rows]):.3f}.",
           f"Hard subset: {len(hard)} patches, {sum(hl)} correct and {len(hl) - sum(hl)} wrong.",
           f"Model time per verdict (batched, one H100): {model_ms:.1f} ms.",
           f"Decision threshold {a.threshold} on the uncalibrated probability, chosen on dev before this run."]
    att_path = a.real_dir / f"serialised_attack_real{sfx}.jsonl"
    if att_path.is_file():
        att = JudgeDataset(att_path).rows
        att_lab = {r["example_id"]: r["label"] for r in att}
        js = {x["example_id"]: int(raw_p(x) >= a.threshold) for x in score(judge, heads, att, temperature=a.temperature)}
        judges = {"**Judge (116M)**": js}
        for spec in a.llm_attack:
            name, path = spec.split("=", 1)
            judges[f"LLM zero-shot: {name}"] = {d["example_id"]: d["pred"] for d in map(json.loads, Path(path).read_text().splitlines())}
        md += ["", f"## With a persuasive note added ({len(att)} patches; the note was written by a 32B model from phrasings no judge trained on)", "",
               "| Judge | Wrong patches accepted (attack success) | Correct patches still accepted | Wrong accepted without the note |", "|---|---|---|---|"]
        plain = {"**Judge (116M)**": "judge", **{f"LLM zero-shot: {n}": n for n in llms}}
        by_id = {r_id: row for r_id, row in zip((r["example_id"] for r in ser), rows)}
        for name, m in judges.items():
            w = [m[k] for k in m if att_lab.get(k) == 0]
            c = [m[k] for k in m if att_lab.get(k) == 1]
            base_key = plain.get(name)
            # the same patches without the note: an attack row's id is its base example's id plus "/attack"
            base_w = [by_id[k.rsplit("/", 1)[0]][base_key] for k in m if att_lab.get(k) == 0
                      and base_key and by_id.get(k.rsplit("/", 1)[0], {}).get(base_key) is not None]
            md.append(f"| {name} | {sum(w) / max(1, len(w)):.1%} of {len(w)} | {sum(c) / max(1, len(c)):.1%} of {len(c)} | "
                      f"{(sum(base_w) / max(1, len(base_w))):.1%} of {len(base_w)} |")
    hs_ = harness_seconds(a.real_dir)
    if hs_:
        md.append(f"Harness time per patch (container start, apply, run every oracle test): median {statistics.median(hs_):.1f} s over {len(hs_)} runs; "
                  "the judge needs only the visible half of that run.")
    a.out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
