"""Generate requirement lists with the trained model (or the untuned base): one greedy reply per prompt.

    python gpu/gen_req.py PROMPTS.jsonl OUT.jsonl [--adapter DIR] [--model Qwen/Qwen2.5-7B-Instruct]
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new", type=int, default=4000)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter))
    model.eval()
    rows = [json.loads(l) for l in a.prompts.read_text().splitlines()]
    with a.out.open("w") as f:
        for r in rows:
            ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                          return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "text": text}) + "\n")
            f.flush()
            print(r["id"], len(text), flush=True)


if __name__ == "__main__":
    main()
