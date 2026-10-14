"""Step 13: the loop on Terminal Wrench tasks (tools/tw_bundle.py) with our local model, on the A100 as the Docker
machine. Each finished round is a training candidate; it is kept only if the hidden pool (attempts the loop never saw)
shows the repaired grader rejecting more hacks while still accepting the correct attempts.

    VB_HOST=ubuntu@<a100> VB_DATA=/home/ubuntu/vbdata VB_REMOTE_REPO=/home/ubuntu/vbrepo2 VB_REMOTE_PYTHON=python3 \
    VB_LOCAL_URL=http://127.0.0.1:8000/v1 python tools/run_step13_tw.py <served model> <sample tag> <workers> TASK ...
"""
import concurrent.futures as cf
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

served, tag, workers, tasks = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
spend = Spend(float(os.environ.get("VB_SPEND_CAP", "1.0")))  # step 19: the teacher run sets its budget
LOOP_CAP = float(os.environ.get("VB_LOOP_CAP", "1.0"))
ROUNDS = int(os.environ.get("VB_ROUNDS", "3"))  # step 13: training rollouts need one round each (amendment 4)


def one(item: str) -> dict:
    task, _, case = item.partition(":")  # TASK, or TASK:CASE for a strict-direction mutant (tools/tw_strict.py)
    if pathlib.Path("runs/step13/STOP_TW").exists():  # a graceful stop: finish what is running, start nothing new
        return {"task": item, "skipped": "STOP_TW"}
    try:
        r = run(task, pathlib.Path("runs/step13/tw_graders") / (case or task) / "tests", ROUNDS, pathlib.Path("runs/step13/tw"),
                spend, model=served if served.startswith(("openrouter:", "claude-")) else "local:" + served, loop_cap=LOOP_CAP, tag=tag, judge="v3")
    except Exception as e:  # noqa: BLE001
        return {"task": task, "error": str(e)[:300]}
    return {"task": task, "before": r["before"].get("balanced_accuracy"), "after": r["after"].get("balanced_accuracy"),
            "tpr": r["after"].get("tpr"), "tnr": r["after"].get("tnr"),
            "rounds": [{"turns": h.get("turns"), "stop": h.get("stop"), "guard": h.get("guard")} for h in r["rounds"]]}


with cf.ThreadPoolExecutor(workers) as ex:
    for res in ex.map(one, tasks):
        print(json.dumps({"TW": tag, **res}), flush=True)
print(json.dumps({"TW_DONE": tag, "api_usd": round(spend.usd, 4)}), flush=True)
