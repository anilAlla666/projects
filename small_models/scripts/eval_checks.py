"""Grader writing, step 3: score the written graders on the real agent fixes, with labels from the oracle.

Self-validation needs no reference: a check is kept only if it fails (or errors) on the unfixed code, since a
check the unfixed code passes does not test the bug. Graders compared, all without the reference fix's tests:
- existing tests: the repository's own passing-to-passing tests (accept when all pass)
- checks: the kept checks (accept when all pass)
- existing tests + checks (accept when all pass)
- judge: our model reading the same evidence (existing tests + kept checks), at THRESHOLD

    python scripts/eval_checks.py REAL_DIR CHECK_RUNS CKPT THRESHOLD TOKENIZER OUT.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.eval.metrics import balanced_accuracy  # noqa: E402


def main():
    real, runs, ckpt, th, tokp, out = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]),
                                       Path(sys.argv[5]), Path(sys.argv[6]))
    from vm.corpus.tok import load
    from vm.data.example import serialise
    insts = {i["instance_id"]: i for i in json.loads((real / "instances.json").read_text())}
    ex = [json.loads(l) for l in (real / "examples_realistic.jsonl").read_text().splitlines()]
    prompts = {p["instance_id"]: p for p in json.loads((real / "check_prompts.json").read_text())}
    tok = load(tokp)
    rows, judge_rows, per_task = [], [], {}
    for e in ex:
        iid, key = e["example_id"].split("/")[0].removeprefix("real:"), e["example_id"].split("/")[1]
        base_p, att_p = runs / iid / "base" / "out" / "result.json", runs / iid / key / "out" / "result.json"
        if not base_p.is_file() or not att_p.is_file():
            continue
        base, att = json.loads(base_p.read_text()), json.loads(att_p.read_text())
        kept = sorted(c for c, o in base.get("outcomes", {}).items() if o in ("failed", "error"))
        if "apply_failed" in att:
            continue
        per_task.setdefault(iid, {"checks": len(base.get("outcomes", {})), "kept": len(kept)})
        r = json.loads((real.parent / "main" / "runs" / iid / key / "out" / "result.json").read_text()) if False else None
        inst = insts[iid]
        # existing tests: the passing-to-passing tests, outcomes from the oracle run of this attempt
        orun = json.loads((real / "runs" / iid / key / "out" / "result.json").read_text())
        p2p = inst["p2p"]
        p2p_fail = [t for t in p2p if orun["outcomes"].get(t) != "passed"]
        chk = att.get("outcomes", {})
        chk_fail = [c for c in kept if chk.get(c) != "passed"]
        row = {"iid": iid, "label": e["label"], "existing": int(not p2p_fail), "checks": int(not chk_fail) if kept else None,
               "both": int(not p2p_fail and not chk_fail)}
        rows.append(row)
        n_vis = len(p2p) + len(kept)
        failing = [f"{t}: failed" for t in p2p_fail][:10] + [f"grader check {c}: failed" for c in chk_fail][:10]
        lines = [f"visible tests: {n_vis - len(p2p_fail) - len(chk_fail)} of {n_vis} pass"] + failing
        lines += [x["text"] for x in e["evidence"] if x["text"].startswith(("test files edited:", "fuzz", "diff:"))]
        pr = prompts[iid]
        text = (f"In `{pr['path']}`, the issue to fix:\n{pr['problem'][:2500]}\n\nThe function `{pr['code_name']}` it concerns:\n")
        jex = {**e, "task_text": text, "evidence": [{"id": i, "text": t} for i, t in enumerate(lines)]}
        try:
            judge_rows.append({**serialise(jex, tok), "example_id": e["example_id"], "attempt_kind": "real", "operator": "real", "library": e["library"]})
        except ValueError:
            rows.pop()
    from vmt.finetune import load_judge
    from vmt.score import score
    judge, heads, _ = load_judge(ckpt)
    raw = {}
    for x in score(judge, heads, judge_rows):
        z = max(-60.0, min(60.0, x["logit"]))
        raw[x["example_id"]] = 1 / (1 + 2.718281828459045 ** (-z))
    for r, jr in zip(rows, judge_rows):
        r["judge"] = int(raw[jr["example_id"]] >= th)
    lab = [r["label"] for r in rows]
    res = {"attempts": len(rows), "correct": sum(lab), "tasks": len(per_task),
           "checks_written": sum(v["checks"] for v in per_task.values()), "checks_kept": sum(v["kept"] for v in per_task.values()),
           "tasks_with_kept_checks": sum(v["kept"] > 0 for v in per_task.values())}
    hard = [r for r in rows if r["existing"] == 1]
    for key in ("existing", "checks", "both", "judge"):
        rs = [r for r in rows if r[key] is not None]
        hs = [r for r in hard if r[key] is not None]
        res[key] = {"ba": round(balanced_accuracy([r["label"] for r in rs], [r[key] for r in rs]), 4),
                    "correct_accepted": round(sum(r[key] for r in rs if r["label"] == 1) / max(1, sum(r["label"] for r in rs)), 4),
                    "wrong_rejected": round(sum(1 - r[key] for r in rs if r["label"] == 0) / max(1, sum(1 - r["label"] for r in rs)), 4),
                    "hard_ba": round(balanced_accuracy([r["label"] for r in hs], [r[key] for r in hs]), 4) if len({r["label"] for r in hs}) == 2 else None,
                    "hard_wrong_rejected": round(sum(1 - r[key] for r in hs if r["label"] == 0) / max(1, sum(1 - r["label"] for r in hs)), 4),
                    "n": len(rs)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
