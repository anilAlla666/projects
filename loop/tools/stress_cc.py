"""Stress test (14 Oct): the whole autonomous path on tasks the loop has never touched, at $0 API spend.

Per task, one after another: an agent writes the grader (base cell: the task and its machine only), then the verify
loop repairs it (judge v3 with the do-no-harm guard, no author questions, evidence = the five leaderboard attempts
outside the hidden pool), and the hidden pool scores the agent's grader and the repaired one. Both models run as
Claude Code headless on the operator's subscription (`cc:` models in vb/writer_api.py); no API key reaches them.

    VB_HOST=root@<docker server> VB_DATA=/root/vbench VB_REMOTE_REPO=/root/vbench/repo_tw \\
    VB_REMOTE_PYTHON=/root/venv/bin/python VB_WRITER_MODEL=cc:sonnet \\
        .venv/bin/python -u tools/stress_cc.py cc:opus 2 TASK ...
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run                       # noqa: E402
from vb.writer_api import Spend, prepare, rollout    # noqa: E402

judge, rounds, tasks = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
OUT = pathlib.Path(os.environ.get("VB_STRESS_OUT", "runs/stress"))
K, TAG = int(os.environ.get("VB_STRESS_K", "0")), os.environ.get("VB_STRESS_TAG", "cc")
for task in tasks:
    try:
        image, text = prepare(task, "base")
        w = rollout(task, "base", K, image, text, OUT / "writers", Spend(1e9))
        grader = OUT / "writers" / f"{task}__base__{K}" / "verifier_out" / "tests"
        if not (grader / "test.sh").is_file():
            print(json.dumps({"STRESS": task, "writer": w.get("stop"), "error": "the writer left no grader"}), flush=True)
            continue
        r = run(task, grader, rounds, OUT / "loops", Spend(1e9), model=judge, loop_cap=99.0, tag=TAG, judge="v3")
        b, a = r["before"], r["after"]
        print(json.dumps({"STRESS": task, "writer_stop": w.get("stop"), "writer_ba": w.get("grade", {}).get("balanced_accuracy"),
                          "before": b.get("balanced_accuracy"), "after": a.get("balanced_accuracy"),
                          "after_tpr": a.get("tpr"), "after_tnr": a.get("tnr"),
                          "rounds": [{k: h.get(k) for k in ("turns", "stop", "seconds", "guard", "notional_usd")}
                                     for h in r["rounds"]]}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"STRESS": task, "error": str(e)[:300]}), flush=True)
print(json.dumps({"STRESS_DONE": True}), flush=True)
