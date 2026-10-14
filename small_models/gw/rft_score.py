"""Reward-filtered training, step 3: score every draft on its task's fixes and keep, per task, the best draft that
fails on the unfixed code, passes the maintainers' fix and every correct agent fix, and fails at least one wrong one.

    python gw/rft_score.py RFT_DIR RUNS_DIR GEN_DIR OUT.jsonl
"""
import json
import sys
from pathlib import Path


def main():
    rft, runs, gen, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    insts = {i["instance_id"]: i for i in json.loads((rft / "instances.json").read_text())}
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in (rft / "prompts.jsonl").read_text().splitlines()}
    best, stats = {}, {"drafts": 0, "ran": 0, "fail_unfixed": 0, "pass_all_correct": 0, "kept": 0}
    for d in sorted(runs.iterdir()):
        iid = d.name.rsplit(".", 1)[0]
        inst = insts.get(iid)
        if not inst or iid not in prompts:
            continue
        stats["drafts"] += 1
        b = d / "base" / "out" / "result.json"
        if not b.is_file():
            continue
        br = json.loads(b.read_text())
        kept = [k for k, v in br.get("outcomes", {}).items() if v != "passed"]
        stats["ran"] += "outcomes" in br
        if not kept:
            continue
        stats["fail_unfixed"] += 1
        ok_correct, caught = True, 0
        for i, at in enumerate(inst["attempts"]):
            r = d / f"a{i}" / "out" / "result.json"
            if not r.is_file():
                ok_correct = ok_correct and not at["target"]
                continue
            res = json.loads(r.read_text())
            if "apply_failed" in res:
                continue
            passed = "outcomes" in res and all(res["outcomes"].get(k) == "passed" for k in kept)
            if at["target"] and not passed:
                ok_correct = False
            if not at["target"] and not passed:
                caught += 1
        if not ok_correct:
            continue
        stats["pass_all_correct"] += 1
        if caught == 0:
            continue
        patch = (gen / f"{d.name}.patch").read_text()
        score = (caught, -len(patch))
        if iid not in best or score > best[iid][0]:
            best[iid] = (score, patch)
    stats["kept"] = len(best)
    out.write_text("".join(json.dumps({"id": i, "prompt": prompts[i], "completion": p}) + "\n" for i, (_, p) in sorted(best.items())))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
