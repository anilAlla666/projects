"""Step 6, test tasks: for each task, once its writer grader exists, the loop alone and the loop with the trained
model's list, from that same grader (docs/research_intent_goal.md, step 6). Two tasks at a time, so at most four
loops run at once; one spend cap over all of them.

    VB_HOST=root@178.156.248.194 VB_DATA=/root/vbench python tools/run_step6.py --cap-usd 40
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

TASKS = ["fin-saccr-rwa", "lake-temp-glm", "roy-polymorph-cn", "sound-change-cascade"]
# layout-config-recreation2 is excluded by the pre-registered rule: its official tests accept 0 of 11 correct
# attempts through our route (logs/step6.log), the fault that excluded layout-config-recreation
ap = argparse.ArgumentParser()
ap.add_argument("--cap-usd", type=float, default=40.0)
ap.add_argument("--loop-cap", type=float, default=5.0)
ap.add_argument("--writers", type=pathlib.Path, default=ROOT / "runs" / "step6_writers")
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "step6_loops")
a = ap.parse_args()
spend = Spend(a.cap_usd)


def task(t: str) -> list:
    res, t0 = a.writers / f"{t}__base__0" / "result.json", time.time()
    while not res.is_file():  # the writer finishes on its own schedule (its own wall limit is one hour)
        if time.time() - t0 > 4 * 3600:
            print(json.dumps({"task": t, "excluded": "no writer result after 4 hours"}), flush=True)
            return []
        time.sleep(30)
    grader = res.parent / "verifier_out" / "tests"
    if not (grader / "test.sh").is_file():
        print(json.dumps({"task": t, "excluded": "the writer left no tests/test.sh"}), flush=True)
        return []
    reqs = json.loads((ROOT / "data" / "reqmodel" / "lists" / f"{t}.json").read_text())

    def arm(tag, rq):  # an arm that already finished is read back, never rerun (no rerun after a score)
        done = a.out / f"{t}__{res.parent.name}__{tag}" / "result.json"
        if done.is_file():
            return json.loads(done.read_text())
        return run(t, grader, 3, a.out, spend, loop_cap=a.loop_cap, requirements=rq, tag=tag, judge="v0")

    with cf.ThreadPoolExecutor(2) as ex:
        arms = [ex.submit(arm, "alone", None), ex.submit(arm, "req", reqs)]
        return [f.result() for f in arms]


with cf.ThreadPoolExecutor(2) as ex:
    out = [r for rs in ex.map(task, TASKS) for r in rs]
print(json.dumps({"spent_usd": round(spend.usd, 2), "loops": [{"loop": r["loop"], "usd": r["usd"],
      "before": r["before"].get("balanced_accuracy"), "after": r["after"].get("balanced_accuracy")} for r in out]},
      indent=1), flush=True)
