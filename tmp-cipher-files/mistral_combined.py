"""V2 + V3 + V4 combined: load Mistral once, run baseline + Marlin in same process.

Steps:
  A. Load Mistral-7B FP16 (~30s)
  B. Disable Marlin actuator via env (CIPHER_MARLIN=off already set externally)
  C. FP16 baseline:
     - B=1 forward × 5; capture logits at fwd 5; measure tok/s
     - B=8 forward × 5; capture logits at fwd 5
  D. Enable Marlin runtime: actuator already loaded at libcipher_rt init,
     toggle via runtime hint. (Actually env is read at init. So we run this
     script TWICE: once with CIPHER_MARLIN=off, once with =on, comparing
     logits across runs.)
"""
import os, time, json, sys
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of GPU computing is to make every joule count. Energy efficiency means doing more useful work per watt."
MODE = sys.argv[1] if len(sys.argv) > 1 else "fp16"  # "fp16" or "marlin"
OUT  = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/mistral_{MODE}.pt"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[{MODE}] loading Mistral-7B...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
mdl.train(False)
print(f"[{MODE}] loaded in {time.time()-t0:.1f}s", flush=True)

p = tok(PROMPT, return_tensors="pt").to("cuda")
input_ids_b8 = p.input_ids.repeat(8, 1)
attn_mask_b8 = p.attention_mask.repeat(8, 1)

# B=1 forwards: 8 to ensure Marlin warmup completes
print(f"[{MODE}] === B=1 × 8 (warmup) ===", flush=True)
y_b1 = None
last_ms_b1 = []
for i in range(8):
    t0 = time.time()
    with torch.no_grad():
        out = mdl(**p)
    torch.cuda.synchronize()
    ms = (time.time()-t0)*1000
    if i >= 5: last_ms_b1.append(ms)
    if i == 7: y_b1 = out.logits.clone()
    print(f"  B=1 fwd {i+1}: {ms:.0f}ms", flush=True)

# B=8 forwards: 8 to ensure Marlin warmup (in case different shapes trigger new quantize)
print(f"[{MODE}] === B=8 × 8 (prefill warmup) ===", flush=True)
y_b8 = None
last_ms_b8 = []
for i in range(8):
    t0 = time.time()
    with torch.no_grad():
        out = mdl(input_ids=input_ids_b8, attention_mask=attn_mask_b8)
    torch.cuda.synchronize()
    ms = (time.time()-t0)*1000
    if i >= 5: last_ms_b8.append(ms)
    if i == 7: y_b8 = out.logits.clone()
    print(f"  B=8 fwd {i+1}: {ms:.0f}ms", flush=True)

n_tok_b1 = p.input_ids.shape[-1]      # prompt tokens
n_tok_b8 = input_ids_b8.numel()       # batch × seq

import statistics as _s
avg_b1 = _s.mean(last_ms_b1) if last_ms_b1 else 0
avg_b8 = _s.mean(last_ms_b8) if last_ms_b8 else 0

result = {
    "mode": MODE,
    "n_prompt_tokens": int(n_tok_b1),
    "n_b8_tokens": int(n_tok_b8),
    "b1_mean_ms_last3": round(avg_b1, 2),
    "b8_mean_ms_last3": round(avg_b8, 2),
    "b1_tok_per_s": round(n_tok_b1 * 1000.0 / avg_b1, 2) if avg_b1 else None,
    "b8_tok_per_s": round(n_tok_b8 * 1000.0 / avg_b8, 2) if avg_b8 else None,
    "b1_argmax_last_token": int(y_b1[0,-1].argmax()),
    "b8_argmax_last_token": int(y_b8[0,-1].argmax()),
}
torch.save({"result": result,
            "y_b1": y_b1[0,-1].cpu().float(),  # last token logits, B=1
            "y_b8": y_b8[0,-1].cpu().float()}, # last token logits, B=8 (batch 0)
           OUT)
print(json.dumps(result, indent=2), flush=True)
print(f"[{MODE}] saved {OUT}", flush=True)
