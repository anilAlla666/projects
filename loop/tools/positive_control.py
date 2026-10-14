"""Positive control for the writer-grader route: the OFFICIAL tests, built through our own build_image (the route
every writer grader takes: our docker build, the ENTRYPOINT layer, pinned, offline), must still score ~1.0 on
every pool. If a task falls short, its writer numbers are not trustworthy."""
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from vb.pool import kept_job, load_manifest
from vb.regrade import grade, verdict
from vb.scorer import Verdict, score_task
from vb.util import TB_ROOT

work = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/vbench/positive_control")


def one(task):
    kept = [e for e in load_manifest(task) if e["kept"]]
    rewards = grade(kept_job(task), TB_ROOT / task, TB_ROOT / task / "tests", work / task, "original_built", n_concurrent=2)
    s = score_task([Verdict(e["entry_id"], e["source"], e["label"], verdict(rewards.get(e["entry_id"]))) for e in kept])
    return task, s


tasks = json.load(open("configs/m1.json"))["tasks"]
with cf.ThreadPoolExecutor(5) as ex:
    for task, s in ex.map(one, tasks):
        print(task, {k: s[k] for k in ("tpr", "tnr", "ba", "flakiness")}, flush=True)
print("CONTROL_DONE")
