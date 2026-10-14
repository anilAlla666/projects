"""A frontier model as the judge, through the Anthropic API, on exactly the prompt the open-model baseline gets
(scripts/llm_baseline.py). Runs from a machine that already holds ANTHROPIC_API_KEY; stops before the spend cap.

    python scripts/api_judge.py EXAMPLES.jsonl OUT.jsonl [--model claude-sonnet-5-5] [--cap-usd 30]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_baseline import parse_verdict, prompt  # noqa: E402

PRICE = {"input": 3.0, "output": 15.0}  # USD per million tokens, Sonnet list price; the report states the assumption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--cap-usd", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    exs = [json.loads(l) for l in a.examples.read_text().splitlines()]
    done = {}
    if a.out.is_file():
        done = {d["example_id"]: d for d in map(json.loads, a.out.read_text().splitlines())}
    lock, spend = threading.Lock(), {"in": 0, "out": 0, "stopped": False}
    f = open(a.out, "a")

    def cost():
        return (spend["in"] * PRICE["input"] + spend["out"] * PRICE["output"]) / 1e6

    def one(e):
        if e["example_id"] in done or spend["stopped"]:
            return
        t0 = time.perf_counter()
        try:
            r = client.messages.create(model=a.model, max_tokens=4096,
                                       messages=[{"role": "user", "content": prompt(e)}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            u = r.usage
            n_in, n_out = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0), u.output_tokens
        except Exception as x:  # noqa: BLE001
            text, n_in, n_out = f"error: {type(x).__name__}", 0, 0
        pred = parse_verdict(text)
        with lock:
            spend["in"] += n_in
            spend["out"] += n_out
            if cost() >= a.cap_usd:
                spend["stopped"] = True
            f.write(json.dumps({"example_id": e["example_id"], "label": e["label"], "pred": pred, "reply": text.strip()[:300],
                                "seconds": round(time.perf_counter() - t0, 2), "in": n_in, "out": n_out}) + "\n")
            f.flush()

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, exs))
    rows = [json.loads(l) for l in a.out.read_text().splitlines()]
    print(json.dumps({"model": a.model, "scored": len(rows), "errors": sum(r["reply"].startswith("error") for r in rows),
                      "accuracy": round(sum(r["pred"] == r["label"] for r in rows) / max(1, len(rows)), 4),
                      "usd_this_run": round(cost(), 2), "stopped_at_cap": spend["stopped"],
                      "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None}))


if __name__ == "__main__":
    main()
