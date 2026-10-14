"""Step 13: the training set for our judge model, as chat conversations in the exact shape the local judge sees
(system prompt, the judge-v3 brief, the bash tool, older outputs trimmed by writer_api.trimmed).

Sources, none from a held-out task:
  opus  Opus loop rounds on training tasks from loops whose hidden score rose (teacher text; Opus's thinking is not
        returned by the API, so these turns carry no reasoning)
  own   our model's own rounds on Terminal Wrench tasks, kept only when the hidden pool (attempts the loop never saw)
        scored the round's repaired grader above the grader it started from, with no correct attempt lost

    python tools/sft_build.py > runs/step13/sft.jsonl          (opus + own: adapter1, step 13)
    python tools/sft_build.py own > runs/step13/sft_own.jsonl  (own only: step 14)
    python tools/sft_build.py own o8,o9,nu > ...               (own only, these sampling tags: step 16)
"""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vb.verify_loop import JUDGES                   # noqa: E402
from vb.writer_api import SYSTEM, TOOLS, instruction  # noqa: E402

HELD_OUT = {"layout-config-recreation2", "photonic-waveguide-routing", "mp-checkpoint-consolidation",
            "production-planning"}
OPUS_LOOPS = [  # Opus loops on training tasks whose hidden score rose (research doc steps 3 to 7c); v4 excluded
    "runs/step6_loops/fin-saccr-rwa__fin-saccr-rwa__base__0__alone",
    "runs/step6_loops/fin-saccr-rwa__fin-saccr-rwa__base__0__req",
    "runs/step6_loops/sound-change-cascade__sound-change-cascade__base__0__alone",
    "runs/step6_loops/sound-change-cascade__sound-change-cascade__base__0__req",
    "runs/step7/roy-polymorph-cn__roy-polymorph-cn__base__0__v2check",
    "runs/step7c/roy-polymorph-cn__roy-polymorph-cn__base__0__v3check",
    "runs/step7b/music-harmony/loop/music-harmony__music-harmony__base__0__make",
    "runs/verify_loop/hof-topology-interpenetration__hof-topology-interpenetration",
    "runs/verify_loop_base/hof-topology-interpenetration__hof-topology-interpenetration__hof-topology-interpenetration__base",
    "runs/verify_loop_req/hof-topology-interpenetration__hof-topology-interpenetration__hof-topology-interpenetration__req",
]
TOOLS_OAI = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]


def brief(task: str) -> str:
    base = pathlib.Path(f"runs/step13/instructions/{task}.md").read_text()
    return JUDGES["v3"] + instruction(base, "cross")


def from_opus(path: str, task: str) -> list:
    msgs, pending = [{"role": "user", "content": brief(task)}], []
    for line in open(path):
        x = json.loads(line)
        if "content" in x:
            text = "".join(c.get("text", "") for c in x["content"] if c.get("type") == "text").strip()
            calls = [c for c in x["content"] if c.get("type") == "tool_use"]
            m = {"role": "assistant", "content": text}
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
    own_only = sys.argv[1:2] == ["own"]  # step 14: the model's own verified repairs and nothing else
    tags = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None  # step 16: only loops with these tags
    for d in ([] if own_only else OPUS_LOOPS):
        r = json.loads(pathlib.Path(d, "result.json").read_text())
        assert r["task"] not in HELD_OUT
        for t in sorted(glob.glob(f"{d}/r*_transcript.jsonl")):
            msgs = from_opus(t, r["task"])
            if len(msgs) > 3:
                print(json.dumps({"source": "opus", "task": r["task"], "file": t, "system": SYSTEM, "tools": TOOLS_OAI,
                                  "messages": msgs}))
                n += 1
    for k in sorted(glob.glob("runs/step13/tw/*/keep.json")):  # written by tools/tw_score.py
        keep = json.loads(pathlib.Path(k).read_text())
        if tags and keep["loop"].rsplit("__", 1)[-1] not in tags:
            continue
        for rnd in keep["rounds"]:
            conv = json.loads(pathlib.Path(rnd["msgs"]).read_text())
            assert keep["task"] not in HELD_OUT
            print(json.dumps({"source": "own", "task": keep["task"], "file": rnd["msgs"], "system": conv["system"],
                              "tools": TOOLS_OAI, "messages": conv["messages"]}))
            n += 1
    print(f"{n} conversations", file=sys.stderr)


if __name__ == "__main__":
    main()
