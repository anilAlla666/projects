"""Step 11, amendment 1: the cumulative-layout-shift loop, on the five leaderboard attempts outside the pool (the
solver step crashed), with what is left of the $14.50: $6.14, less $0.30 for one call's overshoot.

    VB_HOST=root@... VB_DATA=/root/vbench python tools/run_step11b.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

task, cap = "cumulative-layout-shift", 5.84
grader = pathlib.Path("runs/step11") / task / "writer" / f"{task}__base__0" / "verifier_out" / "tests"
spend = Spend(cap)
r = run(task, grader, 3, pathlib.Path("runs/step11") / task / "loop", spend, loop_cap=cap, tag="v3", judge="v3")
print(json.dumps({"STEP11_DONE": {task: {"before": r["before"].get("balanced_accuracy"), "after": r["after"].get(
    "balanced_accuracy"), "tpr": r["after"].get("tpr"), "tnr": r["after"].get("tnr")}}, "usd": round(spend.usd, 3)}),
      flush=True)
