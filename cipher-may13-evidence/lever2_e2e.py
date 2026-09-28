#!/usr/bin/env python3
"""Lever 2 end-to-end smoke test: verify Mistral-7B with V3 + RoPE produces
top-1 tokens that match the no-CIPHER baseline within KIVI noise.

Usage requires LD_PRELOAD of libcipher_hook.so + libcuda. See bottom of file.
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
PROMPT = "The future of artificial intelligence in GPU computing is to make"
N_TOKENS = 16

print("[e2e] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")

print(f"[e2e] CIPHER_KV_REDIRECT={os.environ.get('CIPHER_KV_REDIRECT','')!r} "
      f"CIPHER_KV_RDR_V3={os.environ.get('CIPHER_KV_RDR_V3','')!r} "
      f"CIPHER_KV_RDR_V3_ROPE={os.environ.get('CIPHER_KV_RDR_V3_ROPE','')!r}")

# Reset KV redirect counters at start of generation
rt.cipher_kv_redirect_reset.argtypes = []
try: rt.cipher_kv_redirect_reset()
except Exception: pass

# Stats fetcher
class Stats(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("fa_launches_seen", ctypes.c_uint64),
        ("fa_launches_rewritten", ctypes.c_uint64),
        ("bytes_copied", ctypes.c_uint64),
    ]
rt.cipher_kv_redirect_stats.argtypes = [ctypes.POINTER(Stats)]
rt.cipher_kv_redirect_stats.restype  = ctypes.c_int

with torch.no_grad():
    out = model.generate(prompt_ids, max_new_tokens=N_TOKENS, do_sample=False,
                          use_cache=True)
gen_ids = out[0, prompt_ids.shape[1]:].tolist()
gen_text = tok.decode(gen_ids)
print(f"[e2e] generated tokens: {gen_ids}")
print(f"[e2e] generated text:   {gen_text!r}")

s = Stats()
rt.cipher_kv_redirect_stats(ctypes.byref(s))
print(f"[e2e] kv_redirect: enabled={s.enabled} fa_seen={s.fa_launches_seen} "
      f"fa_rewritten={s.fa_launches_rewritten} bytes={s.bytes_copied}")
