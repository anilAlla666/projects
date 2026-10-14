"""Step 13 (docs/research_intent_goal.md): our own model, served on our own A100 behind vLLM, as the loop's judge, on
the four held-out tasks. No API spend: the judge is local and judge v3 calls no other model. Same graders as the
recorded Opus runs (copied to runs/step13/graders/<task>/tests).

    VB_HOST=root@... VB_DATA=/root/vbench VB_LOCAL_URL=http://127.0.0.1:8000/v1 \
        python tools/run_step13.py <served model name> <tag> [task ...]
"""
import concurrent.futures as cf
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

HELD_OUT = ["layout-config-recreation2", "photonic-waveguide-routing", "mp-checkpoint-consolidation",
            "production-planning"]
served, tag = sys.argv[1], sys.argv[2]
tasks = sys.argv[3:] or HELD_OUT
spend = Spend(1.0)  # a local judge costs $0; the cap only guards against a mistake that reaches a paid model


def one(task: str) -> dict:
    grader = pathlib.Path("runs/step13/graders") / task / "tests"
    try:
        r = run(task, grader, 3, pathlib.Path("runs/step13") / tag, spend, model="local:" + served, loop_cap=1.0,
                tag=tag, judge="v3")
    except Exception as e:  # noqa: BLE001
        return {"task": task, "error": str(e)[:300]}
    return {"task": task, "before": r["before"].get("balanced_accuracy"), "after": r["after"].get("balanced_accuracy"),
            "tpr": r["after"].get("tpr"), "tnr": r["after"].get("tnr"),
            "rounds": [{"turns": h.get("turns"), "stop": h.get("stop"), "seconds": h.get("seconds"),
                        "guard": h.get("guard")} for h in r["rounds"]]}


with cf.ThreadPoolExecutor(3) as ex:  # the Docker server has room for three loop containers at once
    for res in ex.map(one, tasks):
        print(json.dumps({"STEP13": tag, **res}), flush=True)
print(json.dumps({"STEP13_DONE": tag, "api_usd": round(spend.usd, 4)}), flush=True)
