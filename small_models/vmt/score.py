"""Score a serialised file with a fine-tuned judge: probability correct, evidence pointer, hidden-failure bucket
and bug line per example. Optional temperature T divides the verdict logit.

    python vmt/score.py CKPT SERIALISED.jsonl OUT.jsonl [--temperature T]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmt.finetune import load_judge  # noqa: E402
from vmt.judge_data import JudgeDataset, _collate  # noqa: E402


@torch.no_grad()
def score(judge, heads, rows, tokens_per_batch: int = 65536, temperature: float = 1.0):
    judge.eval()
    heads.eval()
    out = []
    rows = sorted(rows, key=lambda r: len(r["tokens"]))
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and (j - i + 1) * len(rows[j]["tokens"]) <= tokens_per_batch:
            j += 1
        chunk = rows[i:j]
        b = _collate([{**r, "weight": 1.0} for r in chunk], "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = judge(b["tokens"], b["segments"])
            v = judge.verdict_logit(h, b["verdict_pos"]).float() / temperature
            p = judge.pointer_logits(h, b["verdict_pos"], b["line_mask"]).float()
            k = heads.bucket_logits(h, b["verdict_pos"]).float()
            g = heads.bug_logits(h, b["verdict_pos"], b["code_mask"]).float()
        for n, r in enumerate(chunk):
            ptr_pos = int(p[n].argmax())
            out.append({"example_id": r["example_id"], "logit": float(v[n] * temperature), "p": float(torch.sigmoid(v[n])),
                        "pointer": r["line_starts"].index(ptr_pos) if ptr_pos in r["line_starts"] else -1,
                        "bucket": int(k[n].argmax()),
                        "bug_line": r["code_line_starts"].index(int(g[n].argmax())) if r["code_line_starts"] and int(g[n].argmax()) in r["code_line_starts"] else -1})
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    judge, heads, _ = load_judge(a.ckpt)
    rows = JudgeDataset(a.data).rows
    res = score(judge, heads, rows, temperature=a.temperature)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in res))
    print(json.dumps({"scored": len(res)}))


if __name__ == "__main__":
    main()
