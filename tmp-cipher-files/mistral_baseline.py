"""V1.2 — Mistral-7B baseline forward, capture for accuracy reference + measure substrate routing telemetry."""
import os, sys, time, hashlib, json
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "The future of GPU computing is to make every joule count. Energy efficiency means doing more useful work per watt."

from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading Mistral-7B (fp16)...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
mdl.train(False)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

p = tok(PROMPT, return_tensors="pt").to("cuda")
print(f"prompt tokens: {p.input_ids.shape[-1]}", flush=True)

# B=1 single decode-shape forward
print("=== B=1 forward ===", flush=True)
t0 = time.time()
with torch.no_grad():
    out_b1 = mdl(**p)
torch.cuda.synchronize()
ms_b1 = (time.time()-t0)*1000
logits_b1 = out_b1.logits[0,-1].float().cpu()
top5_b1 = torch.topk(logits_b1, 5)
print(f"B=1: {ms_b1:.0f}ms argmax={int(logits_b1.argmax())} top5={top5_b1.indices.tolist()}", flush=True)

# B=8 prefill — replicate the prompt across batch
print("=== B=8 forward (replicated prompt) ===", flush=True)
input_ids_b8 = p.input_ids.repeat(8, 1)
attn_mask_b8 = p.attention_mask.repeat(8, 1)
t0 = time.time()
with torch.no_grad():
    out_b8 = mdl(input_ids=input_ids_b8, attention_mask=attn_mask_b8)
torch.cuda.synchronize()
ms_b8 = (time.time()-t0)*1000
logits_b8 = out_b8.logits[0,-1].float().cpu()  # take batch idx 0
top5_b8 = torch.topk(logits_b8, 5)
print(f"B=8: {ms_b8:.0f}ms argmax={int(logits_b8.argmax())} top5={top5_b8.indices.tolist()}", flush=True)

# Save logits for accuracy comparison later
torch.save({
    "logits_b1": out_b1.logits[0,-1].cpu(),
    "logits_b8": out_b8.logits[0,-1].cpu(),  # batch idx 0
}, "/tmp/mistral_fp16_baseline.pt")
print("baseline logits saved to /tmp/mistral_fp16_baseline.pt", flush=True)

# Memory
print(f"cuda mem: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated", flush=True)
