"""Ask a served model to write the grader (a test patch) for each held-out task.

    python gw/generate.py PROMPTS.jsonl MODEL_NAME OUT_DIR [--url http://127.0.0.1:8000]
"""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request
from pathlib import Path


def extract(text: str) -> str:
    """The reply's diff; or, when the reply is a plain Python test file, that file as a new tests/ file, so no writer
    loses on output format alone."""
    m = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    starts = [k for k in (body.find("diff --git"), body.find("--- a/"), body.find("--- /dev/null")) if k >= 0]
    if starts:
        return body[min(starts):]
    py = re.search(r"```python\n(.*?)```", text, re.S)
    if py and "def test" in py.group(1):
        lines = py.group(1).rstrip("\n").split("\n")
        return ("--- /dev/null\n+++ b/tests/test_grader_generated.py\n"
                f"@@ -0,0 +1,{len(lines)} @@\n" + "".join("+" + l + "\n" for l in lines))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("model")
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=1, help="drafts per task, sampled at temperature 0.8 when above 1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in a.prompts.read_text().splitlines()]

    def one(r):
        body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": r["prompt"]}], "n": a.n,
                           "temperature": 0.0 if a.n == 1 else 0.8, "max_tokens": 4096}).encode()
        try:
            req = urllib.request.Request(a.url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                texts = [c["message"]["content"] for c in json.load(resp)["choices"]]
        except Exception as x:  # noqa: BLE001
            texts = [""] * a.n
            print("failed", r["id"], type(x).__name__, flush=True)
        for k, text in enumerate(texts):
            stem = r["id"] if a.n == 1 else f"{r['id']}.{k}"
            (a.out / f"{stem}.patch").write_text(extract(text))
            (a.out / f"{stem}.raw.txt").write_text(text)

    with cf.ThreadPoolExecutor(32) as ex:
        list(ex.map(one, rows))
    print(json.dumps({"written": sum(1 for p in a.out.glob("*.patch") if p.read_text().strip()), "of": len(rows)}))


if __name__ == "__main__":
    main()
