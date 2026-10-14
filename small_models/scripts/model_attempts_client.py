"""Ask a local vLLM server (OpenAI-compatible) for fixes to every task that has none yet, as tasks appear.

    python scripts/model_attempts_client.py ROOT MODEL [--n 4] [--follow]

Appends {task_id, kind, files, untrusted} lines to ROOT/model_attempts.jsonl; with --follow it keeps polling for
new tasks until ROOT/../tasks.log says the final task pass is done."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vm.tasks.model_attempts import parse, prompt  # noqa: E402

URL = "http://127.0.0.1:8000/v1/chat/completions"
SEED = 0


def ask(task: dict, model: str, n: int) -> list:
    """Failures (a prompt over the context window, a server error) give no attempts rather than stopping the run."""
    try:
        return _ask(task, model, n)
    except Exception as e:  # noqa: BLE001
        print("ask failed", task["task_id"], repr(e)[:120], flush=True)
        return []


def _ask(task: dict, model: str, n: int) -> list:
    if len(task["function_lines"]) > 200:
        return []
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt(task)}], "n": n,
                       "temperature": 0.8, "max_tokens": 1024, "seed": SEED}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        choices = json.load(r)["choices"]
    return [a for a in (parse(task, c["message"]["content"]) for c in choices) if a is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--out", default="model_attempts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,dev,test")
    a = ap.parse_args()
    global SEED
    SEED = a.seed
    splits = set(a.splits.split(","))
    out_path = a.root / a.out
    done = {json.loads(l)["task_id"] for l in out_path.read_text().splitlines()} if out_path.is_file() else set()
    asked = set(done)
    out = open(out_path, "a")
    with cf.ThreadPoolExecutor(48) as ex:
        while True:
            tasks = [t for p in sorted(a.root.glob("*/tasks.json")) for t in json.loads(p.read_text()) if t["split"] in splits]
            new = [t for t in tasks if t["task_id"] not in asked]
            asked |= {t["task_id"] for t in new}
            for t, res in zip(new, ex.map(lambda t: (t, ask(t, a.model, a.n)), new)):
                task, attempts = res
                for att in attempts:
                    out.write(json.dumps({"task_id": task["task_id"], "kind": att.kind, "files": att.files, "untrusted": att.untrusted}) + "\n")
                if not attempts:
                    out.write(json.dumps({"task_id": task["task_id"], "kind": "none"}) + "\n")
                out.flush()
            final = (a.root.parent / "tasks.log").is_file() and "TASKS_FINAL_DONE" in (a.root.parent / "tasks.log").read_text()
            if not a.follow or (final and not new):
                break
            time.sleep(30)
    print("DONE", len(asked))


if __name__ == "__main__":
    main()
