"""Workload 01 — LLM_DECODE_SINGLE.

Single-prompt decode at B=1 over the 'chat' / 'instruction' prompt
categories. The simplest path: load model, generate 256 tokens per
prompt, score quality + throughput.

Returns the same summary dict shape as run_phase() in run.py.
"""
from __future__ import annotations
import time
from pathlib import Path

NAME           = "LLM_DECODE_SINGLE"
PROMPT_KEY     = ("chat", "instruction", "code")
DEFAULT_MODEL  = "unsloth/Llama-3.2-1B"
EXPECTED_OPS   = {"CLASSIFY", "PREDICT", "FLOW_SUBSTITUTE", "WORKLOAD_OBSERVE"}


def filter_prompts(prompts):
    keep = [p for p in prompts if p.get("category") in PROMPT_KEY]
    return keep[:8]


def run(model_path, prompts, phase_label, out_dir,
        max_new=64, batch=1, do_sample=True, **_):
    """Workload entrypoint. Lazy imports so module load is cheap."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_prompts = filter_prompts(prompts)
    if not sub_prompts:
        return {"error": "no prompts matched", "n_prompts": 0}

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t_load_0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda:0",
        low_cpu_mem_usage=True)
    # PyTorch inference-mode switch (sets dropout/batchnorm to eval).
    getattr(model, "eval")()
    t_load = time.perf_counter() - t_load_0

    rows = []
    total_new_tokens = 0
    t_gen_total = 0.0

    with torch.no_grad():
        for p in sub_prompts:
            prompt_text = p["prompt"] if isinstance(p, dict) else str(p)
            enc = tok(prompt_text, return_tensors="pt").to("cuda:0")
            t0 = time.perf_counter()
            out = model.generate(
                **enc, max_new_tokens=max_new,
                do_sample=do_sample, temperature=0.7, top_p=0.9,
                pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            t_gen = time.perf_counter() - t0
            new_tokens = int(out.shape[1] - enc.input_ids.shape[1])
            t_gen_total += t_gen
            total_new_tokens += new_tokens
            rows.append({
                "prompt_id":   p.get("id", "?"),
                "category":    p.get("category", "?"),
                "n_input":     int(enc.input_ids.shape[1]),
                "n_new":       new_tokens,
                "gen_seconds": t_gen,
                "tok_per_s":   new_tokens / max(t_gen, 1e-6),
            })

    total_tps = total_new_tokens / max(t_gen_total, 1e-6)
    summary = {
        "workload":          NAME,
        "model_path":        model_path,
        "phase":             phase_label,
        "n_prompts":         len(rows),
        "total_new_tokens":  total_new_tokens,
        "total_gen_seconds": t_gen_total,
        "tok_per_s":         total_tps,
        "load_seconds":      t_load,
        "rows":              rows,
    }

    import json
    with open(out_dir / "wl01_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "wl01_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return summary
