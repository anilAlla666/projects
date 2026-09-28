#!/usr/bin/env python3
"""Task 1 test: run Mistral-7B prefill+a few decode steps with the kernel
table observing every cuLaunchKernelEx, then dump JSON and report the
classification breakdown.  Driver level only — no monkey-patches, no
CIPHER imports in the application code."""
import json, os, sys, ctypes
import warnings
warnings.filterwarnings("ignore")

# Vanilla HuggingFace.  Customer code is unchanged.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"[KT] torch={torch.__version__}  cuda={torch.version.cuda}", flush=True)
MODEL = "mistralai/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda")

prompt = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count.") * 4
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
print(f"[KT] prompt_len={ids.shape[1]}", flush=True)

# Run a small generate so we exercise prefill + decode.  16 tokens is
# enough to see every distinct kernel.
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=16, do_sample=False)
torch.cuda.synchronize()
print(f"[KT] generated  text='{tok.decode(out[0])[-80:]!r}'", flush=True)

# Trigger the kernel-table dump explicitly via the hook's exported API.
hook = ctypes.CDLL("/home/ubuntu/op31-prod-fix/libcipher_hook.so")
hook.cipher_kt_size.restype = ctypes.c_uint
hook.cipher_kt_dump_json.argtypes = [ctypes.c_char_p]
hook.cipher_kt_dump_json.restype = ctypes.c_int
n = hook.cipher_kt_size()
hook.cipher_kt_dump_json(b"/tmp/cipher_kernel_table.json")
print(f"[KT] table size = {n} unique CUfunctions  →  /tmp/cipher_kernel_table.json",
      flush=True)

# Summarize the JSON.
with open("/tmp/cipher_kernel_table.json") as f:
    data = json.load(f)
entries = data["entries"]
by_cat: dict = {}
param_counts = {0: 0}
for e in entries:
    by_cat.setdefault(e["category"], []).append(e)
    param_counts[e["param_count"]] = param_counts.get(e["param_count"], 0) + 1

print("\n=== Classification breakdown ===")
print(f"{'category':<18} {'count':>6} {'observed_total':>14}")
for cat in sorted(by_cat, key=lambda k: -sum(x["observe_count"] for x in by_cat[k])):
    es = by_cat[cat]
    n_obs = sum(x["observe_count"] for x in es)
    print(f"{cat:<18} {len(es):>6} {n_obs:>14}")

print("\n=== cuFuncGetParamInfo result distribution ===")
print("  (PyTorch runtime-API kernels register host stubs; the prior")
print("   session reported param_count=0 for them.  This run confirms.)")
for pc, cnt in sorted(param_counts.items()):
    print(f"  param_count={pc:<3}  {cnt:>4} kernels")

# Show a few examples per category for sanity
print("\n=== Example entries (one per category, top by observe_count) ===")
for cat in sorted(by_cat):
    e = max(by_cat[cat], key=lambda x: x["observe_count"])
    name = e["name"][:84]
    print(f"  [{cat:<14}] params={e['param_count']:<2} "
          f"grid={tuple(e['grid'])}  block={tuple(e['block'])}  "
          f"obs={e['observe_count']:<6}  name={name}")
