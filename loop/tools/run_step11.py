"""Step 11 (docs/research_intent_goal.md): the loop on its own (judge v3, no author), on two coding tasks, one shared
cap of $14.50 of the $15 approved. layout-config-recreation2: the agent writes the grader, the loop uses the five
leaderboard attempts already on file. cumulative-layout-shift: the agent writes the grader, our three solver agents
make the attempts (as the product would), then the loop.

    VB_HOST=root@... VB_DATA=/root/vbench python tools/run_step11.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.solve import solve                                # noqa: E402
from vb.verify_loop import run                            # noqa: E402
from vb.writer_api import Spend, prepare, rollout         # noqa: E402

TOTAL = 14.50
OUT = pathlib.Path("runs/step11")
spend = Spend(TOTAL)
done = {}


def write(task: str) -> pathlib.Path:
    image, text = prepare(task, "base")
    w = rollout(task, "base", 0, image, text, OUT / task / "writer", spend, cap_usd=3.0)
    print(json.dumps({"writer": task, "usd": w.get("usd"), "stop": w.get("stop"), "grade": w.get("grade")}), flush=True)
    return OUT / task / "writer" / f"{task}__base__0" / "verifier_out" / "tests"


for task, own_solvers in (("layout-config-recreation2", False), ("cumulative-layout-shift", True)):
    grader = write(task)
    if not (grader / "test.sh").is_file():
        done[task] = {"error": "the writer left no tests/test.sh"}
        continue
    if own_solvers:
        print(json.dumps({"solvers": task, "results": solve(task, OUT / task / "solvers", spend)}, default=str), flush=True)
    cap = round(min(5.0, TOTAL - spend.usd) if task == "layout-config-recreation2" else TOTAL - spend.usd, 2)
    if cap < 1.0:
        done[task] = {"skipped": f"only ${cap} left for the loop"}
        continue
    r = run(task, grader, 3, OUT / task / "loop", spend, loop_cap=cap, tag="v3", judge="v3")
    done[task] = {k: r[k].get(m) for k in ("before", "after") for m in ("balanced_accuracy",)} | {
        "after_tpr": r["after"].get("tpr"), "after_tnr": r["after"].get("tnr")}
print(json.dumps({"STEP11_DONE": done, "usd": round(spend.usd, 3)}), flush=True)
