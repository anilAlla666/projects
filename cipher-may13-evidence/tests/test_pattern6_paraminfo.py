#!/usr/bin/env python3
"""Pattern 6 — cuFuncGetParamInfo on every observed CUfunction.

Runs Mistral once to populate the kernel table, then queries each
unique kernel via cuFuncGetParamInfo to learn (offset, size) per arg."""
import os, ctypes, json
import torch
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
hook = ctypes.CDLL(os.path.join(ROOT, "libcipher_hook.so"))
hook.cipher_kernel_table_report.restype = None

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16, device_map="cuda")
enc = tok("Hello.", return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = m.generate(**enc, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()
hook.cipher_kernel_table_report()

with open("/tmp/cipher_kernel_table.json") as f:
    data = json.load(f)
kernels = data["kernels"]
kernels.sort(key=lambda k: -k["count"])

# Probe cuFuncGetParamInfo for each unique fn pointer.
libcuda = ctypes.CDLL("libcuda.so.1")
cuFuncGetParamInfo = libcuda.cuFuncGetParamInfo
cuFuncGetParamInfo.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_size_t),
                                ctypes.POINTER(ctypes.c_size_t)]
cuFuncGetParamInfo.restype = ctypes.c_int

print(f"{'count':>10s}  {'class':>14s}  {'#params':>8s}  param_sizes  name")
for k in kernels[:30]:
    fn = ctypes.c_void_p(int(k["fn"], 16))
    sizes = []
    for i in range(32):
        off = ctypes.c_size_t(0); sz = ctypes.c_size_t(0)
        rc = cuFuncGetParamInfo(fn, i, ctypes.byref(off), ctypes.byref(sz))
        if rc != 0:
            break
        sizes.append(sz.value)
    nm = k["name"]
    if len(nm) > 70: nm = nm[:67] + "..."
    print(f"  {k['count']:>10d}  {k['class']:>14s}  {len(sizes):>8d}  "
          f"{str(sizes):28s}  {nm}")
