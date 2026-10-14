"""Step 12, stage 1 (docs/research_intent_goal.md): a small open model, untrained, as the loop's judge on the run
where Opus took layout-config-recreation2 from 0.50 to 1.00 for $1.70. Same grader, same five attempts, judge v3,
3 rounds. Shared cap $4.70 of the $5 approved.

    VB_HOST=root@... VB_DATA=/root/vbench python tools/run_step12.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

task = "layout-config-recreation2"
grader = pathlib.Path("runs/step11") / task / "writer" / f"{task}__base__0" / "verifier_out" / "tests"
spend = Spend(4.70)
done = {}
for tag, model in (("qwen38_27b", "qwen/qwen3.8-27b"), ("gemma4_31b", "google/gemma-4-31b-it")):
    cap = round(min(2.30, spend.cap - spend.usd), 2)
    if cap < 0.5:
        done[tag] = "skipped: cap"; continue
    r = run(task, grader, 3, pathlib.Path("runs/step12"), spend, model="openrouter:" + model, loop_cap=cap,
            tag=tag, judge="v3")
    done[tag] = {"before": r["before"].get("balanced_accuracy"), "after": r["after"].get("balanced_accuracy"),
                 "tpr": r["after"].get("tpr"), "tnr": r["after"].get("tnr"), "usd": r["usd"],
                 "rounds": [{"turns": h.get("turns"), "stop": h.get("stop"), "guard": h.get("guard")} for h in r["rounds"]]}
    print(json.dumps({"MODEL_DONE": tag, **done[tag]}), flush=True)
print(json.dumps({"STEP12_DONE": done, "usd": round(spend.usd, 3)}), flush=True)
