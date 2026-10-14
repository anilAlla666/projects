"""The verifier's results in one table, read from the run folders (never typed): for each agent-written grader, its
balanced accuracy before the loop and after each arm, with the arm's cost.

    python tools/verifier_results.py            # markdown to stdout
    python tools/verifier_results.py --json     # the same rows as JSON

Arms:
- development and held-out graders (steps 3, 5, 6): loop alone (results/step3/expected.json), loop with the trained
  model's list (runs/verify_loop_req), loop with the untuned model's list (runs/verify_loop_base);
- test graders (step 6): loop alone and loop with the trained model's list (runs/step6_loops), from the same grader.
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def results(folder: pathlib.Path) -> dict:
    out = {}
    for f in sorted(folder.glob("*/result.json")):
        r = json.loads(f.read_text())
        out[r["loop"]] = r
    return out


def ba(r: dict | None, side: str = "after"):
    return None if r is None else r[side].get("balanced_accuracy")


def dev_rows() -> list[dict]:
    step3 = json.loads((ROOT / "results" / "step3" / "expected.json").read_text())
    req, base = results(ROOT / "runs" / "verify_loop_req"), results(ROOT / "runs" / "verify_loop_base")
    rows = []
    for key, s3 in step3.items():
        task = s3["task"]
        name = key.split("__", 1)[1]
        rid = f"{task}__{key}"  # graders staged as data/eval_graders/<step-3 loop id>/tests
        r, b = req.get(f"{rid}__req"), base.get(f"{rid}__base")
        rows.append({"set": "held-out" if task in ("mp-checkpoint-consolidation", "photonic-waveguide-routing")
                     else "development", "grader": name, "task": task, "before": s3["before"],
                     "loop_alone": s3["after"], "loop_alone_usd": s3["usd"],
                     "trained": ba(r), "trained_usd": r and r["usd"],
                     "untuned": ba(b), "untuned_usd": b and b["usd"]})
    return rows


def test_rows() -> list[dict]:
    loops = results(ROOT / "runs" / "step6_loops")
    rows = []
    for lid, r in loops.items():
        if not lid.endswith("__alone"):
            continue
        rid = lid[: -len("__alone")]
        t = loops.get(f"{rid}__req")
        rows.append({"set": "test", "grader": rid.split("__", 1)[1], "task": r["task"], "before": ba(r, "before"),
                     "loop_alone": ba(r), "loop_alone_usd": r["usd"], "trained": ba(t), "trained_usd": t and t["usd"]})
    return rows


def product_rows() -> list[dict]:
    """Step 7: vb make (solver evidence, judge v1) on fresh tasks, and judge v1 on roy-polymorph-cn's step 6 grader."""
    rows = []
    for f in sorted((ROOT / "runs" / "step7").glob("*/loop/*/result.json")):
        r = json.loads(f.read_text())
        rows.append({"set": "fresh (step 7)", "task": r["task"], "grader": "writer base__0", "before": ba(r, "before"),
                     "product": ba(r), "product_usd": r["usd"]})
    for f in sorted((ROOT / "runs" / "step7").glob("*__v1check/result.json")):
        r = json.loads(f.read_text())
        rows.append({"set": "regression (step 7)", "task": r["task"], "grader": "step 6 writer base__0",
                     "before": ba(r, "before"), "product": ba(r), "product_usd": r["usd"]})
    return rows


def fmt(x):
    return "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = dev_rows() + test_rows()
    if a.json:
        print(json.dumps(rows + product_rows(), indent=1))
        return
    print("| set | task | grader | before | loop alone | + trained list | + untuned list |")
    print("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['set']} | {r['task']} | {r['grader']} | {fmt(r['before'])} | {fmt(r['loop_alone'])} | "
              f"{fmt(r['trained'])} | {fmt(r.get('untuned'))} |")
    prod = product_rows()
    if prod:
        print()
        print("| set | task | grader | before | after, judge v1 (fresh: its own solver evidence; regression: the leaderboard attempts) | cost |")
        print("|---|---|---|---:|---:|---:|")
        for r in prod:
            print(f"| {r['set']} | {r['task']} | {r['grader']} | {fmt(r['before'])} | {fmt(r['product'])} | "
                  f"${r['product_usd']:.2f} |")
    for s in ("development", "held-out", "test"):
        sub = [r for r in rows if r["set"] == s]
        if not sub:
            continue
        for k in ("before", "loop_alone", "trained", "untuned"):
            v = [r.get(k) for r in sub]
            if v and all(x is not None for x in v):
                print(f"{s} mean {k}: {sum(v) / len(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
