"""One verify loop on a saved grader, for a regression check: the same grader and the task's current candidates, a
named judge version, one cap. Prints the result line.

    VB_HOST=root@... VB_DATA=/root/vbench python tools/run_check.py TASK GRADER_DIR --judge v3 --cap 2.5 --tag v3check
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("grader", type=pathlib.Path)
ap.add_argument("--judge", default="v3")
ap.add_argument("--cap", type=float, default=2.5)
ap.add_argument("--tag", default="check")
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/step7c"))
a = ap.parse_args()
r = run(a.task, a.grader, 3, a.out, Spend(a.cap), loop_cap=a.cap, tag=a.tag, judge=a.judge)
print(json.dumps({"loop": r["loop"], "usd": r["usd"], "before_ba": r["before"].get("balanced_accuracy"),
                  "after_ba": r["after"].get("balanced_accuracy"), "guard": [h.get("guard") for h in r["rounds"]]}),
      flush=True)
