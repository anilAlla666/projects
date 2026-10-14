"""A verifier writer without HUD: the model runs through the Anthropic API from this machine (the key stays here) and
every command it issues runs inside the task's writer container on the server, with no network. The tests it leaves
in /verifier_out/tests are scored by the task's hidden pool grader, as on HUD.

Used for the held-out confirmation (docs/research_intent_goal.md) after HUD access ended: base and cross run in this
same harness, so the comparison between them is like for like.

    python -m vb.writer_api --tasks mp-checkpoint-consolidation,photonic-waveguide-routing --cells base,cross \
        --runs 3 --concurrency 3 --cap-usd 30 --out runs/heldout_api
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import host
from .experiment import CELLS, CROSS, DERIVE, GOAL, INTENT, MUTANTS, PROPERTIES, PURPOSE

MODEL = "claude-sonnet-5"
PRICES = {"claude-haiku-5-5": (1.0, 5.0), "claude-sonnet-5": (3.0, 15.0), "claude-sonnet-5-5": (3.0, 15.0),
          "claude-opus-5-5": (5.0, 25.0)}  # $/M in, out (list prices, assumed)
MAX_TURNS, WALL_S, ROLLOUT_CAP_USD, CMD_TIMEOUT_S, OUT_CAP = 150, 3600, 6.0, 600, 20000
WRITER_MODEL = os.environ.get("VB_WRITER_MODEL", "claude-sonnet-5")  # the grader writer; cc:sonnet = subscription
BLOCKS = {"intent": INTENT, "purpose": PURPOSE, "goal": GOAL, "derive": DERIVE, "mutants": MUTANTS, "cross": CROSS,
          "properties": PROPERTIES}
SYSTEM = ("You work inside a Linux machine through the bash tool. Each call runs in a fresh shell; files you write "
          "persist. The machine has no network. When you have finished the work you were asked to do, reply without "
          "calling a tool.")
TOOLS = [{"name": "bash", "description": "Run a bash command on the machine. Returns its output (stdout and stderr; "
          "long output is cut in the middle) and exit code. Commands time out after 600 seconds.",
          "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]


def ssh(cmd: str, stdin: str | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    """A command on the Docker machine (this one, or VB_HOST's: vb/host.py)."""
    return host.run(cmd, stdin, timeout)


def instruction(base: str, cell: str) -> str:
    extra = "".join(BLOCKS[p] for p in CELLS[cell])
    if not extra:
        return base
    anchor = "## What to write"
    assert anchor in base
    return base.replace(anchor, extra.strip() + "\n\n" + anchor, 1)


_PREPARE_LOCKS: dict = {}
_PREPARE_GUARD = threading.Lock()


def prepare(task: str, cell: str) -> tuple[str, str]:
    """Build the writer's machine for (task, cell) on the server; return its image tag and the instruction text.
    Two callers for the same (task, cell) share one build folder, so they take turns (two verify-loop arms on one
    task started together crashed with 'cannot create directory ... File exists')."""
    with _PREPARE_GUARD:
        lock = _PREPARE_LOCKS.setdefault((task, cell), threading.Lock())
    with lock:
        return _prepare(task, cell)


def _prepare(task: str, cell: str) -> tuple[str, str]:
    name = f"write-verifier-{task}"
    ctx, image = f"{host.DATA_DIR}/writer_envs/{cell}/{task}", f"vbw-{cell}-{task}"
    script = f"""set -e
rm -rf {ctx}; mkdir -p {ctx}; cp -a {host.DATA_DIR}/hud_tasks/{name}/environment/. {ctx}/
if [ "{'cross' in CELLS[cell]}" = "True" ]; then
  cp -a {host.DATA_DIR}/candidates/{task}/files {ctx}/_candidates
  printf '\\nCOPY _candidates /candidates\\n' >> {ctx}/Dockerfile
fi
docker build -q -t {image} {ctx} > /dev/null
cat {host.DATA_DIR}/hud_tasks/{name}/instruction.md"""
    r = ssh("bash -s", script, timeout=3600)
    if r.returncode:
        raise RuntimeError(f"prepare {task}/{cell}: {r.stderr[-800:]}")
    return image, instruction(r.stdout, cell)


def run_cmd(container: str, command: str) -> str:
    try:
        r = ssh(f"docker exec -i {container} timeout {CMD_TIMEOUT_S} bash -l", command, timeout=CMD_TIMEOUT_S + 120)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        out, code = "", 124
    if len(out) > OUT_CAP:
        out = out[:OUT_CAP // 3] + f"\n... [{len(out) - OUT_CAP} characters cut] ...\n" + out[-2 * OUT_CAP // 3:]
    return f"{out}\n[exit code {code}]"


class Spend:
    def __init__(self, cap: float):
        self.cap, self.usd, self.lock = cap, 0.0, threading.Lock()

    def add(self, usd: float) -> float:
        with self.lock:
            self.usd += usd
            return self.usd


def cost(u, model: str = MODEL) -> float:
    pin, pout = PRICES[model]
    return (u.input_tokens * pin + u.output_tokens * pout + (u.cache_creation_input_tokens or 0) * pin * 1.25
            + (u.cache_read_input_tokens or 0) * pin * 0.1) / 1e6


OPENROUTER = "openrouter:"  # a model named "openrouter:<id>" runs through OpenRouter's OpenAI-compatible API
LOCAL = "local:"  # a model named "local:<id>" runs on our own GPU behind vLLM (step 13); cost $0, URL in VB_LOCAL_URL
LOCAL_EFFORT = "medium"
CC = "cc:"  # a model named "cc:<model>" runs as Claude Code headless on the operator's subscription (stress test, 14 Oct)
CC_NOTE = """

## How to run commands

You work inside the task's machine through one helper in your current folder: run `./box '<command>'`, with the whole
shell command quoted once (heredocs and pipes go inside the quotes). Every command must go through ./box; nothing
else you could run reaches the machine. Its output is capped at 20,000 characters."""


KEEP_FULL, TRIM_TO = 4, 400  # step 13: a local model sees the last 8 command outputs in full, older ones cut


def trimmed(msgs: list, keep: int = KEEP_FULL, cut: int = TRIM_TO) -> list:
    """The conversation as a local model sees it (step 13, amendment 5): the most recent turns are whole, and every
    turn before a block boundary is shortened - its command output and its command cut to `cut` characters, its
    reasoning dropped. Boundaries move in blocks of `keep` tool calls, so the start of the conversation (the server's
    cached prefix) changes once every `keep` calls and `keep` to 2*`keep`-1 recent outputs stay whole. Measured on a
    50-turn round: the longest context fell from 78,663 to 28,795 tokens, inside the 32k the GPU can train on.
    Training renders through this same function."""
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    cut_n = ((len(tool_idx) - keep) // keep) * keep if keep and len(tool_idx) > keep else 0
    old = set(tool_idx[:cut_n])
    bound = tool_idx[cut_n - 1] if cut_n else -1
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content") or ""
        if i in old and len(c) > cut:
            m = m | {"content": c[:cut] + f"\n... [older output cut to {cut} of {len(c)} characters; run the command "
                                          f"again to see it]"}
        if i < bound and m.get("role") == "assistant":
            m = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
            calls = []
            for x in m.get("tool_calls") or []:
                try:
                    cmd = json.loads(x["function"].get("arguments") or "{}").get("command", "")
                except (json.JSONDecodeError, AttributeError):
                    cmd = str(x["function"].get("arguments"))
                if len(cmd) > cut:
                    cmd = cmd[:cut] + "\n# ... [older command cut]"
                calls.append({**x, "function": {**x["function"], "arguments": json.dumps({"command": cmd})}})
            if calls:
                m["tool_calls"] = calls
        out.append(m)
    return out


def _openrouter_step(model: str, msgs: list) -> tuple[dict, float, str]:
    """One chat completion with the bash tool through OpenRouter, or through our own vLLM server when `model` starts
    with LOCAL: (assistant message, cost in $, finish reason)."""
    import os
    import httpx2 as httpx
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOLS]
    body = {"messages": [{"role": "system", "content": SYSTEM}] + msgs, "tools": tools, "max_tokens": 32000}
    if model.startswith(LOCAL):
        # step 15 test: VB_TRIM=0 sends the whole conversation; VB_TRIM=outputs keeps every command and output whole
        # and drops only old reasoning (step 15b: the whole conversation overflowed 131k tokens in 26 of 36 rounds,
        # half of it the model's own old reasoning); default 1 is step 13's trimming
        trim = os.environ.get("VB_TRIM", "1")
        view = msgs if trim == "0" else trimmed(msgs, cut=10**9) if trim == "outputs" else trimmed(msgs)
        body["messages"] = [{"role": "system", "content": SYSTEM}] + view
        body["max_tokens"] = 16000
        # step 13 amendment: Qwen's template defaults to its highest effort, whose reasoning ran past 16k tokens
        # without acting (a round ended after 8 turns); every local run uses medium
        body["chat_template_kwargs"] = {"reasoning_effort": LOCAL_EFFORT}
        url, headers = os.environ.get("VB_LOCAL_URL", "http://127.0.0.1:8000/v1") + "/chat/completions", {}
        body["model"] = model[len(LOCAL):]
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
        body |= {"model": model, "usage": {"include": True},
                 "provider": {"sort": "throughput"}}  # step 12: the slowest provider cost 3 min a turn
    tries = 241 if model.startswith(LOCAL) else 21  # our own server: ride out a lost tunnel for 2 hours (step 13)
    for wait in range(tries):
        try:
            r = httpx.post(url, timeout=1800, headers=headers, json=body)
            if r.status_code == 400 and model.startswith(LOCAL):  # the conversation outgrew the server's context:
                return {"role": "assistant", "content": ""}, 0.0, "context_full"  # end the round, never retry
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"openrouter {r.status_code}")
            r.raise_for_status()
            j = r.json()
            if not j.get("choices"):  # OpenRouter returns an upstream failure (e.g. a free provider overloaded) as 200
                raise RuntimeError(f"openrouter error body: {str(j.get('error'))[:200]}")
            break
        except Exception:
            if wait == tries - 1:
                raise
            time.sleep(30)
    c = j["choices"][0]
    return c["message"], float((j.get("usage") or {}).get("cost") or 0.0), c.get("finish_reason") or ""


def _openrouter_loop(container: str, text: str, log_path: Path, spend: Spend, model: str, max_turns: int,
                     wall_s: int, cap_usd: float, intercept=None) -> dict:
    """The same agent as agent_loop, for a model behind OpenRouter (step 12: a small open model as the judge)."""
    msgs, usd, turns, t0, stop = [{"role": "user", "content": text}], 0.0, 0, time.time(), "end_turn"
    with Path(log_path).open("w") as log:
        while True:
            if turns >= max_turns:
                stop = "max_turns"; break
            if time.time() - t0 > wall_s:
                stop = "wall_clock"; break
            if usd > cap_usd:
                stop = "cost_cap"; break
            m, c, fin = _openrouter_step(model, msgs)
            if fin == "context_full":
                stop = fin; break
            usd += c
            spend.add(c)
            turns += 1
            calls = m.get("tool_calls") or []
            log.write(json.dumps({"turn": turns, "usd": round(usd, 4), "stop": fin,
                                  "content": [{"type": "text", "text": m.get("content") or ""}] + [
                                      {"type": "tool_use", "input": x["function"].get("arguments")} for x in calls]}) + "\n")
            log.flush()
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content",
                                                             "reasoning")} | {"role": "assistant"})
            if not calls and fin == "length":  # cut off at the output limit: continue, as the Anthropic path does
                msgs.append({"role": "user", "content": "Your reply was cut off at the output limit. Continue; write "
                             "large files in several smaller commands."})
                continue
            if not calls:
                stop = fin or "end_turn"; break
            for x in calls:
                try:
                    cmd = json.loads(x["function"].get("arguments") or "{}").get("command", "")
                except json.JSONDecodeError:
                    cmd = ""
                o = (intercept(cmd) if intercept else None) if cmd else "[no command given]"
                if o is None:
                    o = run_cmd(container, cmd)
                msgs.append({"role": "tool", "tool_call_id": x["id"], "content": o})
                log.write(json.dumps({"turn": turns, "tool": {"command": cmd}, "output": o[:4000]}) + "\n")
    # step 13: the exact conversation the model saw (full tool outputs, its reasoning), for training on its own rounds
    Path(str(log_path).replace(".jsonl", "") + "_msgs.json").write_text(
        json.dumps({"system": SYSTEM, "messages": msgs}, ensure_ascii=False))
    return {"stop": stop, "turns": turns, "usd": round(usd, 4), "seconds": round(time.time() - t0)}


def _claude_code_loop(container: str, text: str, log_path: Path, model: str, wall_s: int) -> dict:
    """One agent as Claude Code headless (`claude -p`), billed to the operator's subscription: the API key and base URL
    are removed from its environment so it cannot fall back to API billing. Its only tool is ./box, which runs one
    command in `container` exactly as run_cmd does. The stream (every message and tool call) goes to log_path."""
    import shutil
    import tempfile
    wd = Path(tempfile.mkdtemp(prefix="vbcc-"))
    inner = f"docker exec -i {container} timeout {CMD_TIMEOUT_S} bash -l"
    pipe = f"ssh -o BatchMode=yes -o ServerAliveInterval=30 {host.HOST} '{inner}'" if host.REMOTE else inner
    (wd / "box").write_text(f"#!/bin/bash\nprintf '%s' \"$*\" | {pipe} 2>&1 | head -c {OUT_CAP}\n")
    (wd / "box").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")}
    cmd = ["claude", "-p", text + CC_NOTE, "--model", model[len(CC):], "--allowedTools", "Bash(./box:*)",
           "--append-system-prompt", SYSTEM, "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--setting-sources", ""]
    t0, stop, turns, usd = time.time(), "error", 0, 0.0
    try:
        p = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True, timeout=wall_s, errors="replace")
        Path(log_path).write_text(p.stdout)
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                turns, usd = ev.get("num_turns") or 0, float(ev.get("total_cost_usd") or 0)
                stop = "error" if ev.get("is_error") else (ev.get("stop_reason") or "end_turn")
    except subprocess.TimeoutExpired as e:
        Path(log_path).write_text((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        stop = "wall_clock"
    shutil.rmtree(wd, ignore_errors=True)
    # usd is the notional list price Claude Code reports; the subscription pays it, so it is not added to Spend
    return {"stop": stop, "turns": turns, "usd": 0.0, "notional_usd": round(usd, 4), "seconds": round(time.time() - t0)}


def agent_loop(container: str, text: str, log_path: Path, spend: Spend, model: str = MODEL,
               max_turns: int = MAX_TURNS, wall_s: int = WALL_S, cap_usd: float = ROLLOUT_CAP_USD,
               intercept=None) -> dict:
    """One agent: the model through the API, its bash calls run in `container`; stops when it replies without a
    tool call or at a limit. Every reply and tool output goes to log_path. `intercept(command)` may answer a command
    itself (returning its output) instead of running it in the container (returning None)."""
    if model.startswith(CC):
        return _claude_code_loop(container, text, log_path, model, wall_s)
    if model.startswith(LOCAL):  # our own GPU: same agent, same 3-hour round as OpenRouter
        return _openrouter_loop(container, text, log_path, spend, model, max_turns,
                                max(wall_s, 3 * 3600), cap_usd, intercept)
    if model.startswith(OPENROUTER):  # a capability test, not a latency test: 3 hours a round (step 12 amendment)
        return _openrouter_loop(container, text, log_path, spend, model[len(OPENROUTER):], max_turns,
                                max(wall_s, 3 * 3600), cap_usd, intercept)
    import anthropic
    client, msgs, usd, turns, t0, stop = anthropic.Anthropic(max_retries=8), [{"role": "user", "content": text}], 0.0, 0, time.time(), "end_turn"
    with Path(log_path).open("w") as log:
        while True:
            if turns >= max_turns:
                stop = "max_turns"; break
            if time.time() - t0 > wall_s:
                stop = "wall_clock"; break
            if usd > cap_usd:
                stop = "cost_cap"; break
            last = msgs[-1]
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            for m in msgs:  # one rolling cache point, on the newest message
                for b in m["content"] if isinstance(m["content"], list) else []:
                    if isinstance(b, dict):
                        b.pop("cache_control", None)
            last["content"][-1]["cache_control"] = {"type": "ephemeral"}
            for wait in range(21):  # a network drop on this machine (step 9: music-harmony lost $1.82 to one) waits
                try:              # up to 10 minutes; a failed connection is not billed, so retrying costs nothing
                    # step 19: VB_THINK=<effort> asks for the teacher's reasoning, returned as a summary, for training
                    extra = {"thinking": {"type": "adaptive", "display": "summarized"},
                             "extra_body": {"output_config": {"effort": os.environ["VB_THINK"]}}} \
                        if os.environ.get("VB_THINK") else {}
                    with client.messages.stream(model=model, max_tokens=32000, system=SYSTEM, tools=TOOLS,
                                                messages=msgs, **extra) as st:
                        resp = st.get_final_message()  # streaming: the SDK refuses a long non-streamed reply
                    break
                except Exception as e:  # a dropped connection, or a stream that stalls mid-reply (step 11's solvers)
                    stalled = type(e).__module__.split(".")[0] in ("httpx", "httpx2", "httpcore", "httpcore2")
                    if not (isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)) or stalled) \
                            or wait == 20:
                        raise
                    time.sleep(30)
            c = cost(resp.usage, model)
            usd += c
            spend.add(c)
            turns += 1
            blocks = [b.model_dump(exclude_none=True) for b in resp.content]
            log.write(json.dumps({"turn": turns, "usd": round(usd, 4), "stop": resp.stop_reason, "content": blocks}) + "\n")
            log.flush()
            msgs.append({"role": "assistant", "content": blocks})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls and resp.stop_reason == "max_tokens":  # cut off mid-reply, not finished: ask it to go on
                msgs.append({"role": "user", "content": "Your reply was cut off at the output limit. Continue; write "
                             "large files in several smaller commands."})
                continue
            if not calls:
                stop = resp.stop_reason; break
            results = []
            for b in calls:
                cmd = (b.input or {}).get("command", "")
                o = intercept(cmd) if intercept else None
                if o is None:
                    o = run_cmd(container, cmd)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": o})
                log.write(json.dumps({"turn": turns, "tool": b.input, "output": o[:4000]}) + "\n")
            msgs.append({"role": "user", "content": results})
    return {"stop": stop, "turns": turns, "usd": round(usd, 4), "seconds": round(time.time() - t0)}


def rollout(task: str, cell: str, k: int, image: str, text: str, out: Path, spend: Spend,
            cap_usd: float = ROLLOUT_CAP_USD) -> dict:
    rid = f"{task}__{cell}__{k}"
    d = out / rid
    if (d / "result.json").is_file():
        return json.loads((d / "result.json").read_text())
    if spend.usd >= spend.cap:  # never start a rollout past the cap; a started one finishes under its own cap
        return {"task": task, "cell": cell, "k": k, "skipped": "spend cap reached", "grade": {}}
    d.mkdir(parents=True, exist_ok=True)
    container = f"vbw-run-{rid}"[:120]
    ssh(f"docker rm -f {container} >/dev/null 2>&1; docker run -d --name {container} --network none --cpus 2.5 "
        f"--memory 5g --entrypoint sleep {image} infinity && docker exec {container} mkdir -p /verifier_out")
    t0 = time.time()
    try:
        r = agent_loop(container, text, d / "transcript.jsonl", spend, model=WRITER_MODEL, cap_usd=cap_usd)
        stop, turns, usd = r["stop"], r["turns"], r["usd"]
    finally:
        tests = f"{host.DATA_DIR}/writer_runs/{rid}"
        ssh(f"rm -rf {tests}; mkdir -p {tests}; docker cp {container}:/verifier_out/. {tests}/ 2>/dev/null; "
            f"docker rm -f {container} >/dev/null 2>&1")
    g = ssh(f"T={host.DATA_DIR}/writer_runs/{rid}/tests; if [ -f $T/test.sh ]; then docker run --rm --cap-add SYS_ADMIN "
            f"--security-opt seccomp=unconfined --security-opt apparmor=unconfined -v $T:/verifier_out/tests:ro "
            f"vbhud-{task} bash /tests/test.sh 2>/dev/null | tail -1; else echo '{{\"reward\": 0, \"balanced_accuracy\": 0.5, "
            f"\"note\": \"no tests written\"}}'; fi", timeout=7200)
    try:
        grade = json.loads(g.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        grade = {"error": (g.stdout + g.stderr)[-500:]}
    host.get(f"{host.DATA_DIR}/writer_runs/{rid}", d / "verifier_out")
    res = {"task": task, "cell": cell, "k": k, "model": WRITER_MODEL, "stop": stop, "turns": turns, "usd": round(usd, 4),
           "minutes": round((time.time() - t0) / 60, 1), "grade": grade}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k2: res[k2] for k2 in ("task", "cell", "k", "stop", "turns", "usd", "minutes")} |
                     {"ba": grade.get("balanced_accuracy"), "tpr": grade.get("tpr"), "tnr": grade.get("tnr")}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--cells", default="base,cross")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--cap-usd", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("runs/heldout_api"))
    a = ap.parse_args()
    tasks, cells = a.tasks.split(","), a.cells.split(",")
    prep = {(t, c): prepare(t, c) for t in tasks for c in cells}
    spend = Spend(a.cap_usd)
    jobs = [(t, c, k) for k in range(a.runs) for t in tasks for c in cells]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda j: rollout(*j, *prep[(j[0], j[1])], a.out, spend), jobs))
    summary = {}
    for r in res:
        summary.setdefault(f"{r['task']}/{r['cell']}", []).append(r["grade"].get("balanced_accuracy"))
    print(json.dumps({"spent_usd": round(spend.usd, 2), "ba": summary}, indent=1))


if __name__ == "__main__":
    main()
