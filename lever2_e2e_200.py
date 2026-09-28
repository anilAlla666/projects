#!/usr/bin/env python3
"""V3 KV redirect end-to-end on 200 decode tokens, with tok/s timing.

Compares baseline (no CIPHER) vs V3+RoPE on the same prompt. Reports:
  - generated text (coherence check)
  - tok/s
  - kv_redirect counters
"""
import os, sys, ctypes, time
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "Explain how a CPU executes instructions step by step"
N_TOKENS = 200

print("[e2e] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
model.train(False)
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")

mode = (
    f"V3={os.environ.get('CIPHER_KV_RDR_V3','')!r} "
    f"RoPE={os.environ.get('CIPHER_KV_RDR_V3_ROPE','')!r}"
)
print(f"[e2e] {mode}")

rt.cipher_kv_redirect_reset.argtypes = []
try: rt.cipher_kv_redirect_reset()
except Exception: pass

class Stats(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("fa_launches_seen", ctypes.c_uint64),
        ("fa_launches_rewritten", ctypes.c_uint64),
        ("bytes_copied", ctypes.c_uint64),
    ]
rt.cipher_kv_redirect_stats.argtypes = [ctypes.POINTER(Stats)]
rt.cipher_kv_redirect_stats.restype  = ctypes.c_int

# Warm up
with torch.no_grad():
    _ = model.generate(prompt_ids, max_new_tokens=4, do_sample=False, use_cache=True)
torch.cuda.synchronize()

# Timed run
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

gen_ids = out[0, prompt_ids.shape[1]:].tolist()
gen_text = tok.decode(gen_ids)
print(f"[e2e] {N_TOKENS} tokens in {elapsed:.2f}s = {N_TOKENS/elapsed:.2f} tok/s")
print(f"[e2e] first 60 tokens: {gen_ids[:60]}")
print(f"[e2e] generated text:")
print(f"    {gen_text!r}")

s = Stats()
rt.cipher_kv_redirect_stats(ctypes.byref(s))
print(f"[e2e] kv_redirect: enabled={s.enabled} fa_seen={s.fa_launches_seen} "
      f"fa_rewritten={s.fa_launches_rewritten} bytes={s.bytes_copied}")
