"""P(PASS) for every row of a verdict-model file, read from the next-token logits after the chat prompt (the same
template the trainer uses), never from generated text. PASS and FAIL are each the log-sum-exp over their first
tokens with and without a leading space, so the untuned model is not penalised for spacing.

    python gpu/score_verdict.py ROWS.jsonl OUT.jsonl [--adapter DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter))
    model.eval()

    def first(words):
        return sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in words})
    yes, no = first([" PASS", "PASS"]), first([" FAIL", "FAIL"])
    assert not set(yes) & set(no), (yes, no)
    rows = [json.loads(l) for l in a.rows.read_text().splitlines()]
    with a.out.open("w") as f:
        for r in rows:
            ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                          return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            sy, sn = torch.logsumexp(lp[yes], 0), torch.logsumexp(lp[no], 0)
            p = torch.sigmoid(sy - sn).item()
            f.write(json.dumps({"id": r["id"], "task": r["task"], "label": r["label"], "p_pass": p,
                                "mass": (sy.exp() + sn.exp()).item(), "tokens": ids.shape[1]}) + "\n")
            f.flush()
    print("scored", len(rows), a.out, flush=True)


if __name__ == "__main__":
    main()
