"""A frontier model writes the graders through the Anthropic API, on exactly the prompts the open models get.

    python gw/api_generate.py PROMPTS.jsonl OUT_DIR [--model claude-sonnet-5-5] [--cap-usd 10]
"""
import argparse
import concurrent.futures as cf
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import extract  # noqa: E402

PRICE = {"input": 3.0, "output": 15.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--cap-usd", type=float, default=10.0)
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    a.out.mkdir(parents=True, exist_ok=True)
    lock, spend = threading.Lock(), {"in": 0, "out": 0, "stopped": False}

    def cost():
        return (spend["in"] * PRICE["input"] + spend["out"] * PRICE["output"]) / 1e6

    def one(r):
        dst = a.out / f"{r['id']}.patch"
        if dst.is_file() or spend["stopped"]:
            return
        try:
            m = client.messages.create(model=a.model, max_tokens=20000, messages=[{"role": "user", "content": r["prompt"]}])
            text = "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            u = m.usage
            n_in, n_out = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0), u.output_tokens
        except Exception as x:  # noqa: BLE001
            text, n_in, n_out = "", 0, 0
            print("failed", r["id"], type(x).__name__, flush=True)
        dst.write_text(extract(text))
        (a.out / f"{r['id']}.raw.txt").write_text(text)
        with lock:
            spend["in"] += n_in
            spend["out"] += n_out
            if cost() >= a.cap_usd:
                spend["stopped"] = True

    rows = [json.loads(l) for l in a.prompts.read_text().splitlines()]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(one, rows))
    print(json.dumps({"written": sum(1 for p in a.out.glob("*.patch") if p.read_text().strip()), "of": len(rows),
                      "usd": round(cost(), 2), "stopped_at_cap": spend["stopped"]}))


if __name__ == "__main__":
    main()
