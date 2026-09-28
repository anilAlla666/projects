#!/usr/bin/env python3
# GATE-1 g1.3 PROBE A (engagement reality): in the REAL torch path, does an fp16 GEMM reach the cublasGemmEx shim
# (where Koopman+EDMD-collect fire), or route through cublasLt (which the shim only counts, NOT routes -- cipher_rt_
# cublas_shim.c:259-266)? LD_PRELOAD the deployed .so + CIPHER_KOOPMAN=1 VERBOSE=1; read koopman calls_total. >0 =
# cublasGemmEx engaged; 0 = torch uses cublasLt, Koopman/EDMD never sees the GEMM (inert in engine path). No capture.
import os, sys, ctypes
import torch
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_koopman_calls_total","cipher_rt_koopman_calls_handled","cipher_rt_koopman_is_active",
          "cipher_rt_koopman_bf16_observed"):
    getattr(lib,s).restype=ctypes.c_ulong
print(f"[koopman] is_active={lib.cipher_rt_koopman_is_active()} (CIPHER_KOOPMAN={os.environ.get('CIPHER_KOOPMAN')})",flush=True)
# real fp16 GEMMs across a few shapes (decode + prefill-ish)
for (m,k,n) in [(1,2048,2048),(1,4096,4096),(8,4096,11008),(32,4096,4096)]:
    A=torch.randn(m,k,dtype=torch.float16,device="cuda"); B=torch.randn(k,n,dtype=torch.float16,device="cuda")
    for _ in range(20): C=torch.matmul(A,B)
torch.cuda.synchronize()
# a tiny real HF-style decode to exercise the actual model GEMM entry points
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B")
    M=AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B",torch_dtype=torch.float16,device_map="cuda").eval()
    ids=tok("The history of AI",return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        for _ in range(8): ids=torch.cat([ids,M(ids).logits[:,-1:].argmax(-1)],dim=1)
    torch.cuda.synchronize()
    hf_ok=True
except Exception as e:
    hf_ok=f"{type(e).__name__}: {str(e)[:120]}"
ct=lib.cipher_rt_koopman_calls_total(); hd=lib.cipher_rt_koopman_calls_handled(); bf=lib.cipher_rt_koopman_bf16_observed()
print(f"[ENGAGEMENT] koopman calls_total={ct} handled={hd} bf16_observed={bf} (hf_decode={hf_ok})",flush=True)
print(f"[VERDICT] {'cublasGemmEx PATH LIVE -- Koopman sees torch fp16 GEMMs (calls_total>0)' if ct>0 else 'cublasGemmEx path NOT hit by torch fp16 GEMMs (calls_total=0) -> torch routes via cublasLt; shipped Koopman/EDMD is INERT in the engine GEMM path (shim only counts cublasLt)'}",flush=True)
sys.stdout.flush(); os._exit(0)
