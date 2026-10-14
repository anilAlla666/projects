"""Step 17: the first reinforcement-learning batch, built offline from rounds the base model already ran.

Every round of the given sampling passes was scored on the hidden pool by tools/tw_score.py (keep.json, "all").
Reward = hidden balanced accuracy after the round minus before. Rounds are grouped by case (the loop id without its
pass tag); a case whose rounds all scored the same carries no signal and is skipped. Each round's advantage is
clip((reward - group mean) / SCALE, -CLIP, CLIP), with negatives multiplied by NEG. Output: one conversation per line,
in sft_build.py's shape, plus "weight" (the advantage) and "reward".

    python tools/rl_build.py o8,o9,o10,o11,o12,o13 > runs/step13/rl1.jsonl
"""
import collections
import glob
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.sft_build import HELD_OUT, TOOLS_OAI  # noqa: E402

SCALE, CLIP = 0.2, 2.0
NEG = float(__import__("os").environ.get("RL_NEG", "0.5"))  # step 17b: 1.0, failures weighted fully


def main() -> None:
    tags = set(sys.argv[1].split(","))
    groups = collections.defaultdict(list)
    for k in sorted(glob.glob("runs/step13/tw/*/keep.json")):
        lid = pathlib.Path(k).parent.name
        case, tag = lid.rsplit("__", 1)
        if tag not in tags:
            continue
        keep = json.loads(pathlib.Path(k).read_text())
        assert keep["task"] not in HELD_OUT
        for e in keep["all"]:
            b = (e.get("hidden_before") or {}).get("balanced_accuracy")
            a = (e.get("hidden_after") or {}).get("balanced_accuracy")
            msgs = pathlib.Path(k).parent / f"r{e['round']}_transcript_msgs.json"
            if b is None or a is None or not msgs.exists():
                continue
            groups[case].append({"task": keep["task"], "file": str(msgs), "reward": round(a - b, 6)})
    n = used = 0
    for case, rs in sorted(groups.items()):
        rewards = [r["reward"] for r in rs]
        if len(set(rewards)) < 2:
            continue
        mean = statistics.mean(rewards)
        for r in rs:
            adv = max(-CLIP, min(CLIP, (r["reward"] - mean) / SCALE))
            adv = adv * NEG if adv < 0 else adv
            if adv == 0:
                continue
            conv = json.loads(pathlib.Path(r["file"]).read_text())
            print(json.dumps({"source": "own", "task": r["task"], "file": r["file"], "system": conv["system"],
                              "tools": TOOLS_OAI, "messages": conv["messages"], "weight": round(adv, 4),
                              "reward": r["reward"]}))
            used += 1
        n += 1
    print(f"{used} rounds from {n} cases with mixed outcomes", file=sys.stderr)


if __name__ == "__main__":
    main()
