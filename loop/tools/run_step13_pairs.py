"""Step 13: run (model, arm, task) loops in ONE process, so two arms preparing the same task's machine take turns
(writer_api.prepare's lock is per process; two processes collided on 12 Oct and five loops failed to start).

    VB_DATA=... VB_LOCAL_URL=... python tools/run_step13_pairs.py SERVED:TAG:TASK ...
"""
import concurrent.futures as cf
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

spend = Spend(1.0)


def one(item: str) -> dict:
    served, tag, task = item.split(":")
    try:
        r = run(task, pathlib.Path("runs/step13/graders") / task / "tests", 3, pathlib.Path("runs/step13") / tag, spend,
                model="local:" + served, loop_cap=1.0, tag=tag, judge="v3")
    except Exception as e:  # noqa: BLE001
        return {"STEP13": tag, "task": task, "error": str(e)[:300]}
    return {"STEP13": tag, "task": task, "before": r["before"].get("balanced_accuracy"),
            "after": r["after"].get("balanced_accuracy"), "tpr": r["after"].get("tpr"), "tnr": r["after"].get("tnr"),
            "rounds": [{"turns": h.get("turns"), "stop": h.get("stop"), "seconds": h.get("seconds"),
                        "guard": h.get("guard")} for h in r["rounds"]]}


with cf.ThreadPoolExecutor(len(sys.argv) - 1) as ex:
    for res in cf.as_completed([ex.submit(one, i) for i in sys.argv[1:]]):
        print(json.dumps(res.result()), flush=True)
print(json.dumps({"PAIRS_DONE": True, "api_usd": round(spend.usd, 4)}), flush=True)
