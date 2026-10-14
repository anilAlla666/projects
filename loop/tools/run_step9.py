"""Step 9 (docs/research_intent_goal.md): judge v4 on roy then music-harmony, one shared cap, one run each.

    VB_HOST=root@... VB_DATA=/root/vbench python tools/run_step9.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

TOTAL = 4.18  # $4.70 less the $0.52 the aborted first roy run and the hand check spent (amendment, step 9)
OUT = pathlib.Path("runs/step9")
if sys.argv[1:2] == ["music"]:  # the music-harmony rerun (amendment 2): its own cap, given on the command line
    spend = Spend(float(sys.argv[2]))
    music = run("music-harmony", pathlib.Path("runs/step7b/music-harmony/writer/music-harmony__base__0/verifier_out/tests"),
                2, OUT, spend, loop_cap=float(sys.argv[2]), tag="v4b", judge="v4")
    print(json.dumps({"STEP9_DONE": {"music": {"before": music["before"].get("balanced_accuracy"),
                                               "after": music["after"].get("balanced_accuracy")}},
                      "usd": round(spend.usd, 3)}), flush=True)
    sys.exit(0)
spend = Spend(TOTAL)
done = {}
roy = run("roy-polymorph-cn", pathlib.Path("runs/step6_writers/roy-polymorph-cn__base__0/verifier_out/tests"), 2, OUT,
          spend, loop_cap=1.30, tag="v4b", judge="v4")
done["roy"] = {"before": roy["before"].get("balanced_accuracy"), "after": roy["after"].get("balanced_accuracy")}
music_cap = round(TOTAL - spend.usd, 2)
if music_cap >= 1.0:
    music = run("music-harmony", pathlib.Path("runs/step7b/music-harmony/writer/music-harmony__base__0/verifier_out/tests"),
                2, OUT, spend, loop_cap=music_cap, tag="v4", judge="v4")
    done["music"] = {"before": music["before"].get("balanced_accuracy"), "after": music["after"].get("balanced_accuracy")}
print(json.dumps({"STEP9_DONE": done, "usd": round(spend.usd, 3)}), flush=True)
