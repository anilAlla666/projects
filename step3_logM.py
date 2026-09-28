#!/usr/bin/env python3
"""STEP 3 — Log M dimension in cublasGemmEx at batch=8.

Goal: determine whether decode at batch=8 issues one batched GEMM with M=8
(so batching is already free / captured) or eight M=1 GEMMs called serially
(so we should buffer inputs and batch).

Approach: run a 30-step decode with the patched stack, set CIPHER_KERNEL_LOG=on
so libcipher_hook prints `[CIPHER GEMM-EX] #N m=X n=Y k=Z` for the first
200 cublasGemmEx calls. We dump the unique (m,n,k) tuples to a JSON.
"""
import os, sys, ctypes, time, subprocess, json, re
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralMLP

MODEL = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")

print(f"[step3] B=8 decode shape probe (no INT4 substitution)")

prompt = ("Energy efficiency means doing more useful work per watt. ") * 16
prompt_ids = tok(prompt, return_tensors="pt").input_ids[:, :128].to("cuda").expand(8, -1).contiguous()
B = 8
prompt_len = prompt_ids.shape[1]

cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=200,
                    device="cuda", dtype=torch.float16)
with torch.no_grad():
    cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
    out = model(input_ids=prompt_ids, cache_position=cp,
                past_key_values=cache, use_cache=True, return_dict=True)
torch.cuda.synchronize()
input_ids = out.logits[:, -1:].argmax(-1).contiguous()
cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)

# 30 decode steps. The hook will log first 200 cublasGemmEx with M,N,K.
print("[step3] running 30 decode steps to elicit cublasGemmEx logs ...")
for _ in range(30):
    with torch.no_grad():
        out = model(input_ids=input_ids, cache_position=cache_pos,
                    past_key_values=cache, use_cache=True, return_dict=True)
    input_ids = out.logits.argmax(-1)
    cache_pos += 1

torch.cuda.synchronize()
print("[step3] decode loop done")
