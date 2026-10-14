"""Step 13: score every finished round of the Terminal Wrench loops on the hidden pool (attempts the loop never saw)
and keep the rounds that made the grader better: the guard kept the round's repair, the hidden balanced accuracy
rose, and the hidden TPR did not fall. Rounds are read from the run logs (a loop stopped early has no result.json).
Writes runs/step13/tw/<loop>/keep.json for tools/sft_build.py.

    VB_HOST=ubuntu@<a100> VB_DATA=/home/ubuntu/vbdata VB_REMOTE_REPO=/home/ubuntu/vbrepo2 VB_REMOTE_PYTHON=python3 \
        python tools/tw_score.py runs/step13_tw*.log
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb import host                      # noqa: E402
from vb.verify_loop import grade         # noqa: E402

ROOT = pathlib.Path("runs/step13/tw")
rounds = {}
for log in sys.argv[1:]:
    for line in open(log):
        if line.startswith('{"loop'):
            d = json.loads(line)
            if "round" in d and d["loop"].startswith("tw-") and (ROOT / d["loop"]).is_dir():  # skip loop summaries
                rounds.setdefault(d["loop"], {})[d["round"]] = d
scores = {}


def hidden(task: str, base: str, j: int) -> dict | None:
    if (base, j) not in scores:
        ok = host.run(f"test -f {base}/r{j}/tests/test.sh && echo y").stdout.strip() == "y"
        scores[(base, j)] = grade(task, f"{base}/r{j}/tests") if ok else None
    return scores[(base, j)]


for lid, rs in sorted(rounds.items()):
    task = lid.split("__")[0]
    base = f"{host.DATA_DIR}/loop/{lid}"
    keep, every = [], []
    for k, h in sorted(rs.items()):
        msgs = ROOT / lid / f"r{k}_transcript_msgs.json"
        if not msgs.exists():
            continue
        b, a = hidden(task, base, k), hidden(task, base, k + 1)
        good = bool((h.get("guard") or {}).get("kept")) and bool(b) and bool(a) and \
            a.get("balanced_accuracy", 0) > b.get("balanced_accuracy", 0) and a.get("tpr", 0) >= b.get("tpr", 0)
        every.append({"round": k, "guard": h.get("guard"), "hidden_before": b, "hidden_after": a, "good": good,
                      "wrong_candidates": (h.get("verdict") or {}).get("wrong_candidates")})
        if good:
            keep.append({"round": k, "msgs": str(msgs), "before": b["balanced_accuracy"], "after": a["balanced_accuracy"]})
    (ROOT / lid / "keep.json").write_text(json.dumps({"task": task, "loop": lid, "rounds": keep, "all": every}, indent=1))
    print(json.dumps({"loop": lid, "kept": [x["round"] for x in keep],
                      "hidden": [((e["hidden_before"] or {}).get("balanced_accuracy"),
                                  (e["hidden_after"] or {}).get("balanced_accuracy")) for e in every]}), flush=True)
