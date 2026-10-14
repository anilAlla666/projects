"""How much a persuasive note changes the judge's verdict on dev: the attacked rows against the same examples
without the note, at a given decision threshold.

    python scripts/note_effect.py CKPT ROOT THRESHOLD
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ckpt, root, t = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])  # ROOT holds the serialised dev and attack rows
    from vmt.finetune import load_judge
    from vmt.judge_data import JudgeDataset
    from vmt.score import score
    judge, heads, _ = load_judge(ckpt)
    att = JudgeDataset(root / "serialised_attack_dev.jsonl").rows
    base = {r["example_id"]: r for r in JudgeDataset(root / "serialised_dev.jsonl").rows}
    pairs = [(a, base[a["example_id"].rsplit("/", 1)[0]]) for a in att if a["example_id"].rsplit("/", 1)[0] in base]
    pa = {x["example_id"]: x["p"] for x in score(judge, heads, [a for a, _ in pairs])}
    pb = {x["example_id"]: x["p"] for x in score(judge, heads, [b for _, b in pairs])}
    out = {}
    for lab, name in ((0, "wrong"), (1, "correct")):
        ps = [(pb[b["example_id"]] >= t, pa[a["example_id"]] >= t) for a, b in pairs if a["label"] == lab]
        out[name] = {"n": len(ps), "accepted_without_note": round(sum(x for x, _ in ps) / len(ps), 4),
                     "accepted_with_note": round(sum(y for _, y in ps) / len(ps), 4),
                     "flipped_to_accept": sum((not x) and y for x, y in ps), "flipped_to_reject": sum(x and not y for x, y in ps)}
    print(json.dumps({"checkpoint": str(ckpt), "threshold": t, **out}))


if __name__ == "__main__":
    main()
