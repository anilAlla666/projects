"""Step 3: the verify loop on a fixed list of graders, sharing one spend cap.

    python tools/run_loops.py --cap-usd 60 --concurrency 2 TASK:GRADER_DIR ...
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import run           # noqa: E402
from vb.writer_api import Spend          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("items", nargs="+")
ap.add_argument("--cap-usd", type=float, default=60.0)
ap.add_argument("--loop-cap", type=float, default=9.0)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--concurrency", type=int, default=2)
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/verify_loop"))
ap.add_argument("--tag", default="", help="suffix for the loop ids of this arm")
a = ap.parse_args()
spend = Spend(a.cap_usd)
def item(i):  # TASK:GRADER or TASK:GRADER:REQUIREMENTS_JSON
    p = i.split(":")
    return p[0], pathlib.Path(p[1]), (json.loads(pathlib.Path(p[2]).read_text()) if len(p) > 2 else None)


items = [item(i) for i in a.items]
with cf.ThreadPoolExecutor(a.concurrency) as ex:
    res = list(ex.map(lambda it: run(it[0], it[1], a.rounds, a.out, spend, loop_cap=a.loop_cap, requirements=it[2], tag=a.tag, judge="v0"), items))
print(json.dumps({"spent_usd": round(spend.usd, 2), "loops": [{"loop": r["loop"], "usd": r["usd"],
      "before": r["before"].get("balanced_accuracy"), "after": r["after"].get("balanced_accuracy"),
      "after_tpr": r["after"].get("tpr"), "after_tnr": r["after"].get("tnr")} for r in res]}, indent=1))
