"""Reward-filtered training, step 1: training tasks that come with labelled agent fixes (nebius/SWE-agent-trajectories
on SWE-rebench images), from repositories outside the exam set. Writes an instances file in the real-set format (the
maintainers' fix is attempt "gold", always correct) and the writer prompts with repository context.

    python gw/rft_select.py RAW_DIR REAL_DIR OUT_DIR [--tasks 300]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("real", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tasks", type=int, default=300)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    cols = ["instance_id", "repo", "docker_image", "license_name", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config", "test_patch",
            "patch", "base_commit", "problem_statement"]
    reb = {r["instance_id"]: r for f in sorted(glob.glob(str(a.raw / "*SWE-rebench*.parquet"))) for r in pq.read_table(f, columns=cols).to_pylist()
           if r["docker_image"]}
    held = {i["repo"] for i in json.loads((a.real / "instances.json").read_text())}
    trajs = {}
    for f in sorted(glob.glob(str(a.raw / "traj_*.parquet"))):
        for t in pq.read_table(f, columns=["instance_id", "target", "generated_patch"]).to_pylist():
            if t["instance_id"] in reb and (t["generated_patch"] or "").strip():
                trajs.setdefault(t["instance_id"], []).append(t)
    picked, repos = [], set()
    for iid in sorted(trajs, key=lambda i: hashlib.sha256(f"gw-rft:{i}".encode()).hexdigest()):
        r = reb[iid]
        cfg = r["install_config"] or {}
        if r["repo"] in held or r["repo"] in repos or cfg.get("log_parser") != "parse_log_pytest":
            continue
        if not re.search(r"\b(mit|apache|bsd|isc)\b", r["license_name"] or "", re.I):
            continue
        seen, good, bad = set(), [], []
        for t in trajs[iid]:
            if t["generated_patch"] in seen:
                continue
            seen.add(t["generated_patch"])
            (good if t["target"] else bad).append(t)
        if not bad:
            continue
        repos.add(r["repo"])
        atts = [{"model": "maintainers", "target": True, "patch": r["patch"]}] + \
               [{"model": "agent", "target": True, "patch": t["generated_patch"]} for t in good[:2]] + \
               [{"model": "agent", "target": False, "patch": t["generated_patch"]} for t in bad[:4]]
        picked.append({"instance_id": iid, "repo": r["repo"], "docker_image": r["docker_image"], "base_commit": r["base_commit"],
                       "test_patch": r["test_patch"], "patch": r["patch"], "problem_statement": r["problem_statement"],
                       "attempts": atts})
        if len(picked) >= a.tasks:
            break
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "instances.json").write_text(json.dumps(picked))
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(32) as ex:
        built = [b for b in ex.map(lambda r: context.build(r, eval_mode=True), picked) if b]
    (a.out / "prompts.jsonl").write_text("".join(json.dumps({"id": b["id"], "prompt": b["prompt"], "completion": b["completion"]}) + "\n" for b in built))
    print(json.dumps({"tasks": len(picked), "prompts": len(built), "attempts": sum(len(p["attempts"]) for p in picked)}))


if __name__ == "__main__":
    main()
