#!/usr/bin/env python3
# Fallback capture: torch HF (AutoModelForCausalLM, SDPA) prefill+decode on TinyLlama.
# A DIFFERENT attention lib than vLLM's _vllm_fa3_C (torch ATen SDPA flash/cuDNN) +
# cublasGemmEx decode GEMMs — so it validates the AGNOSTIC claim (does GEOMETRY
# recognize attention + GEMM regardless of the framework/lib?). Run with
# CUDA_INJECTION64_PATH + CIPHER_GEOM_CAPTURE=1 + CIPHER_MARLIN=on.
import os, ctypes, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/home/ubuntu/models/TinyLlama-1.1B"
t = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = t("The history of computing spans several distinct eras, each defined by", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(3):
        m.generate(ids, max_new_tokens=32, do_sample=False)   # prefill + decode
torch.cuda.synchronize()
print("GEN done", flush=True)
so = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
lib = ctypes.CDLL(so)
if hasattr(lib, "cipher_kernel_table_report"):
    lib.cipher_kernel_table_report(); print("kernel_table dumped", flush=True)
print("CAPTURE_DONE", flush=True)
