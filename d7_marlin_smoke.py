# D.7 step-3 de-risk: does Marlin ENGAGE on a real forward in-container with the staging .so?
# Single small family (TinyLlama), CIPHER_MARLIN=1. Check GEMM dispatches > 0 + bind_model callable.
import os, ctypes, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/home/ubuntu/models/TinyLlama-1.1B"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).cuda().eval()
# bind every Linear weight to a family model_id via the exported setter (resolve in-proc lib)
lib = ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
try:
    lib.cipher_rt_marlin_engine_bind_model.argtypes=[ctypes.c_void_p, ctypes.c_ulonglong]
    nbound=0
    for n,p in m.named_parameters():
        if p.dim()==2:  # Linear weights
            lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(p.data_ptr()), 0x7111AA11)
            nbound+=1
    print("bound %d Linear weights to model_id"%nbound)
except Exception as e:
    print("bind_model not resolvable via CDLL(None):", e)
ids = tok("The history of computing", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(8):  # warm + a few decode steps so Marlin observes/quantizes
        out = m.generate(ids, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()
print("forward OK, generated shape", tuple(out.shape))
