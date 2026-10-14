"""Score written graders on the labelled real attempts. A grader's kept tests are the ones that fail (or error) on the
unfixed code: a test the unfixed code passes does not test the issue (self-validation, no reference needed). The
grader accepts an attempt when every kept test passes. Labels come from the maintainers' tests (the oracle).

    python gw/eval_graders.py REAL_DIR RUNS_DIR [RUNS_DIR ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vm.eval.metrics import balanced_accuracy  # noqa: E402


def score(real: Path, runs: Path) -> dict:
    insts = {i["instance_id"]: i for i in json.loads((real / "instances.json").read_text())}
    ex = [json.loads(l) for l in (real / "examples_realistic.jsonl").read_text().splitlines()]
    rows, tasks = [], {}
    for e in ex:
        iid, key = e["example_id"].split("/")[0].removeprefix("real:"), e["example_id"].split("/")[1]
        bp, ap_ = runs / iid / "base" / "out" / "result.json", runs / iid / key / "out" / "result.json"
        t = tasks.setdefault(iid, {"grader": False, "kept": 0})
        if not bp.is_file():
            rows.append({"iid": iid, "label": e["label"], "pred": 1, "existing": None, "has_grader": False})
            continue
        b = json.loads(bp.read_text())
        kept = [k for k, v in b.get("outcomes", {}).items() if v != "passed"] if "outcomes" in b else []
        t["kept"] = len(kept)
        t["grader"] = bool(kept)
        if not ap_.is_file():
            continue
        r = json.loads(ap_.read_text())
        if "apply_failed" in r:
            continue
        if "outcomes" in r and kept:
            pred = int(all(r["outcomes"].get(k) == "passed" for k in kept) and not r.get("timed_out"))
        elif kept:
            pred = 0  # the grader's tests could not run on this attempt
        else:
            pred = 1  # no useful test: the grader accepts everything
        orun = json.loads((real / "runs" / iid / key / "out" / "result.json").read_text())
        existing = int(all(orun["outcomes"].get(t_) == "passed" for t_ in insts[iid]["p2p"]))
        rows.append({"iid": iid, "label": e["label"], "pred": pred, "existing": existing, "has_grader": bool(kept)})
    lab = [r["label"] for r in rows]
    hard = [r for r in rows if r["existing"] == 1]
    per = {}
    for r in rows:
        per.setdefault(r["iid"], []).append(r)
    macro = [balanced_accuracy([x["label"] for x in v], [x["pred"] for x in v]) for v in per.values() if len({x["label"] for x in v}) == 2]
    return {"attempts": len(rows), "tasks": len(per), "tasks_with_useful_grader": sum(t["grader"] for t in tasks.values()),
            "ba": round(balanced_accuracy(lab, [r["pred"] for r in rows]), 4),
            "macro_ba_per_task": round(sum(macro) / max(1, len(macro)), 4),
            "correct_accepted": round(sum(r["pred"] for r in rows if r["label"]) / max(1, sum(lab)), 4),
            "wrong_rejected": round(sum(1 - r["pred"] for r in rows if not r["label"]) / max(1, len(lab) - sum(lab)), 4),
            "hard_wrong_rejected": round(sum(1 - r["pred"] for r in hard if not r["label"]) / max(1, sum(1 - r["label"] for r in hard)), 4),
            "hard_correct_accepted": round(sum(r["pred"] for r in hard if r["label"]) / max(1, sum(r["label"] for r in hard)), 4)}


if __name__ == "__main__":
    real = Path(sys.argv[1])
    print(json.dumps({Path(p).name: score(real, Path(p)) for p in sys.argv[2:]}, indent=1))
