"""V2 — Mistral-7B forward × 6 with Marlin enabled. Verify kernels fire."""
import os, time, json
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of GPU computing is to make every joule count. Energy efficiency means doing more useful work per watt."

from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading Mistral-7B...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
mdl.train(False)

p = tok(PROMPT, return_tensors="pt").to("cuda")

# B=1 forwards to cross stability threshold (=4) and let Marlin fire
print("=== B=1 forward × 8 (Marlin should activate at fwd 5+) ===", flush=True)
y_first = None
y_last = None
for i in range(8):
    t0 = time.time()
    with torch.no_grad():
        out = mdl(**p)
    torch.cuda.synchronize()
    if i == 0: y_first = out.logits.clone()
    if i == 7: y_last  = out.logits.clone()
    print(f"  fwd {i+1}: {(time.time()-t0)*1000:.0f}ms argmax={int(out.logits[0,-1].argmax())}", flush=True)

# Save B=1 first (FP16) and last (Marlin) for accuracy compare
torch.save({
    "logits_fp16_b1": y_first[0,-1].cpu().float(),
    "logits_marlin_b1": y_last[0,-1].cpu().float(),
    "logits_full_fp16_b1": y_first.cpu().float(),
    "logits_full_marlin_b1": y_last.cpu().float(),
}, "/tmp/mistral_b1_logits.pt")
print("B=1 logits saved", flush=True)
