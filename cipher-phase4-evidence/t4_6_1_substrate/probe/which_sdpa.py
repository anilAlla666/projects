"""Diagnose which attention variant Mistral 7B actually uses on this stack.
Hook torch.nn.functional.scaled_dot_product_attention to see what gets called
+ enable backend logging to learn which ATen op dispatches."""
import os, sys, time, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hook torch.nn.functional.scaled_dot_product_attention to log calls
_orig_sdpa = F.scaled_dot_product_attention
_call_count = {"sdpa": 0}
def _hook(*args, **kwargs):
    _call_count["sdpa"] += 1
    if _call_count["sdpa"] <= 2:
        q = args[0] if args else kwargs.get("query")
        print(f"[hook] F.scaled_dot_product_attention call#{_call_count['sdpa']} "
              f"Q={tuple(q.shape)} dtype={q.dtype} dev={q.device}", flush=True)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _hook

# Override at::_ops dispatch as well via torch._C ops listing
# Look at what's available
ops = [n for n in dir(torch.ops.aten) if "scaled_dot_product" in n]
print("[probe] aten ops containing scaled_dot_product:", ops, flush=True)

# torch._C._scaled_dot_product_flash_attention introspection
mp = "/home/ubuntu/models/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(mp)
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16,
                                         attn_implementation="sdpa").cuda()
m.train(False)

with torch.inference_mode():
    inp = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    # Enable the SDP backend report
    print("[probe] running with explicit SDP flash-only ctx...", flush=True)
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        out = m.generate(inp, max_new_tokens=4, do_sample=False)
print(f"[probe] sdpa hook calls = {_call_count['sdpa']}", flush=True)
print(f"[probe] output = {tok.decode(out[0])}", flush=True)
