"""Step 19: training conversations from the teacher's (Opus's) loop rounds, with its summarized reasoning.

A round is used if its repair raised agreement with the evidence by at least MIN_GAIN items (the guard's own counts:
the candidates the judge classified, the empty attempt and its near misses), whether or not the guard kept it: the
guard's "nothing lost" rule rejected two rounds that rose 5 to 11 and 5 to 10, and the hidden pools of these cases are
too small (as few as 2 correct and 4 wrong attempts) to score one round reliably (amendment 2). The transcript gives
the conversation: the brief, each turn's summarized thinking (as reasoning), its text and bash calls, and each
command's output (the log keeps the first 4,000 characters).

    python tools/teacher_build.py runs/step19_oa2.log runs/step19_ob.log runs/step19_smoke.log > runs/step13/teacher.jsonl
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.sft_build import HELD_OUT, TOOLS_OAI  # noqa: E402
from vb.writer_api import SYSTEM  # noqa: E402

MIN_GAIN = 3
ROOT = pathlib.Path("runs/step13/tw")


def brief_tw(task: str) -> str:
    """The exact brief the judge saw on this task: the first user message of our own model's round on it (the same
    judge v3 text and task instruction; the Opus log does not record its first message)."""
    for p in sorted(ROOT.glob(f"{task}__{task}__o*/r0_transcript_msgs.json")):
        return json.loads(p.read_text())["messages"][0]["content"]
    raise FileNotFoundError(f"no own round on {task} to take the brief from")


def conversation(path: pathlib.Path, first_user: str) -> list:
    msgs, pending = [{"role": "user", "content": first_user}], []
    for line in open(path):
        x = json.loads(line)
        if "content" in x:
            think = "\n\n".join(b.get("thinking", "") for b in x["content"] if b.get("type") == "thinking").strip()
            text = "".join(b.get("text", "") for b in x["content"] if b.get("type") == "text").strip()
            calls = [b for b in x["content"] if b.get("type") == "tool_use"]
            m = {"role": "assistant", "content": text}
            if think:
                m["reasoning"] = think
            if calls:
                m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": (c.get("input") or {}).get("command", "")})}}
                    for c in calls]
            msgs.append(m)
            pending = [c["id"] for c in calls]
        elif "tool" in x and pending:
            msgs.append({"role": "tool", "tool_call_id": pending.pop(0), "content": x["output"]})
    return msgs


def main() -> None:
    n = 0
    for log in sys.argv[1:]:
        for line in open(log):
            if not line.startswith('{"loop'):
                continue
            d = json.loads(line)
            g = d.get("guard") or {}
            if "round" not in d or g.get("old_agree") is None or g.get("new_agree") is None:
                continue
            if g["new_agree"] - g["old_agree"] < MIN_GAIN:
                continue
            task = d["loop"].split("__")[0]
            assert task not in HELD_OUT
            t = ROOT / d["loop"] / f"r{d['round']}_transcript.jsonl"
            if not t.exists():
                continue
            msgs = conversation(t, brief_tw(task))
            print(json.dumps({"source": "teacher", "task": task, "file": str(t), "system": SYSTEM, "tools": TOOLS_OAI,
                              "messages": msgs, "gain": g["new_agree"] - g["old_agree"]}))
            n += 1
    print(f"{n} teacher rounds", file=sys.stderr)


if __name__ == "__main__":
    main()
