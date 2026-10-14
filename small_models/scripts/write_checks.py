"""Grader writing, step 1: an LLM reads only the task (the issue text) and the code it concerns, and writes a small
pytest file of checks a correct fix must pass. No reference fix and no hidden test is shown. One file per task.

    python scripts/write_checks.py PROMPTS.json OUT_DIR [--model claude-sonnet-5-5] [--cap-usd 12]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import threading
from pathlib import Path

PRICE = {"input": 3.0, "output": 15.0}

PROMPT = """You are writing the grader for a bug fix in the Python repository `{repo}`. Someone else will submit a
fix; your checks decide whether it is correct. You do not see the fix or the project's hidden tests.

The issue the fix must resolve:
<issue>
{problem}
</issue>

The code the issue concerns, before any fix (`{path}`, `{code_name}`):
```python
{code}
```

Write one pytest file of 4 to 8 test functions that a correct fix must pass and the current code fails.
Rules:
- Test only behaviour the issue asks for. Prefer the issue's own examples and expected outputs.
- Do not test details the issue leaves open (exact wording of messages, internal helper names, performance).
- Import from the package as it is laid out in the repository; the repository root is the working directory.
- No network, no files outside a temporary directory, each test under 2 seconds.
- Each test checks one thing and has a descriptive name.

Reply with the test file only, in one ```python code block."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--cap-usd", type=float, default=12.0)
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    a.out.mkdir(parents=True, exist_ok=True)
    lock, spend = threading.Lock(), {"in": 0, "out": 0, "stopped": False}
    log = open(a.out / "write_log.jsonl", "a")

    def cost():
        return (spend["in"] * PRICE["input"] + spend["out"] * PRICE["output"]) / 1e6

    def one(p):
        dst = a.out / f"{p['instance_id']}.py"
        if dst.is_file() or spend["stopped"]:
            return
        try:
            r = client.messages.create(model=a.model, max_tokens=8000, messages=[{"role": "user", "content": PROMPT.format(**p)}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            u = r.usage
            n_in, n_out = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0), u.output_tokens
        except Exception as x:  # noqa: BLE001
            text, n_in, n_out = "", 0, 0
            print("failed", p["instance_id"], type(x).__name__, flush=True)
        m = re.search(r"```python\n(.*?)```", text, re.S)
        if m:
            dst.write_text(m.group(1))
        with lock:
            spend["in"] += n_in
            spend["out"] += n_out
            if cost() >= a.cap_usd:
                spend["stopped"] = True
            log.write(json.dumps({"instance_id": p["instance_id"], "ok": bool(m), "in": n_in, "out": n_out}) + "\n")
            log.flush()

    prompts = json.loads(a.prompts.read_text())
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(one, prompts))
    print(json.dumps({"written": len(list(a.out.glob("*.py"))), "usd": round(cost(), 2), "stopped_at_cap": spend["stopped"]}))


if __name__ == "__main__":
    main()
