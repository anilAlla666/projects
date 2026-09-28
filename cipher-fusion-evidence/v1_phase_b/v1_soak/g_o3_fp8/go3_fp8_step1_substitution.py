#!/usr/bin/env python3
# G-O3 STEP 1 -- SUBSTITUTION REALITY FIRST: on a COMPUTE-BOUND workload (prefill of a long prompt / large-batch
# forward, cublas n>64), does CIPHER's FP8 actuator actually SUBSTITUTE (handled>0, FP8 GEMM replacing the bf16 GEMM
# at the cublasGemmEx boundary) AND keep output numerically sane (not just a counter)? Gate per advisor:
# weights_quantized>0 AND handled>0 AND logits finite + close to bf16 baseline. Real model (Mistral-7B / Llama-3.1-8B
# bf16), NOT decode, NOT synthetic. Substrate-line: cublasGemmEx library-symbol-intercept (LD_PRELOAD or
# CUDA_INJECTION64_PATH) -- NOT vLLM, NOT nvjet, NOT monkeypatch.
import os, sys, ctypes
import torch
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL=os.environ.get("GO3_MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
SEQ=int(os.environ.get("GO3_SEQ","2048")); BATCH=int(os.environ.get("GO3_BATCH","1"))
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_fp8_is_active","cipher_rt_fp8_calls_total","cipher_rt_fp8_calls_handled",
          "cipher_rt_fp8_calls_skipped","cipher_rt_fp8_weights_quantized","cipher_rt_fp8_max_n"):
    getattr(lib,s).restype=ctypes.c_ulong
def stat(): return dict(active=lib.cipher_rt_fp8_is_active(),total=lib.cipher_rt_fp8_calls_total(),
    handled=lib.cipher_rt_fp8_calls_handled(),skipped=lib.cipher_rt_fp8_calls_skipped(),
    wq=lib.cipher_rt_fp8_weights_quantized(),max_n=lib.cipher_rt_fp8_max_n())
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
print(f"[go3-s1] CIPHER_FP8={os.environ.get('CIPHER_FP8')} fp8_is_active={lib.cipher_rt_fp8_is_active()} model={os.path.basename(MODEL)} seq={SEQ} batch={BATCH}",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
cfg=m.config
# compute-bound prefill: a long prompt repeated to SEQ tokens, batch BATCH -> cublas n = BATCH*SEQ >> 64
base="The history of artificial intelligence and high performance computing spans many decades of research. "
ids=tok(base,return_tensors="pt").input_ids
ids=ids.repeat(1, (SEQ//ids.shape[1])+1)[:, :SEQ].repeat(BATCH,1).cuda()
print(f"[go3-s1] prefill input {tuple(ids.shape)} (tokens/fwd={ids.numel()})",flush=True)
print(f"[go3-s1] BEFORE fwd: {stat()}",flush=True)
# Run several forwards: FP8_STABILITY=2 observations + one-time weight prequant must warm up before substitution.
with torch.no_grad():
    for i in range(5):
        out=m(ids).logits; torch.cuda.synchronize()
        print(f"[go3-s1] after fwd#{i+1}: {stat()}",flush=True)
st=stat()
fin=torch.isfinite(out).all().item(); lo=out.float().abs().max().item()
print(f"[go3-s1] AFTER fwd: {st}",flush=True)
print(f"[go3-s1] logits finite={fin} max|logit|={lo:.2f} argmax_last={int(out[0,-1].argmax())}",flush=True)
subst_real = (st['handled']>0 and st['wq']>0 and fin)
print(f"[go3-s1 VERDICT] SUBSTITUTION-REAL={subst_real} (handled={st['handled']}>0, weights_quantized={st['wq']}>0, logits_finite={fin}, max_n={st['max_n']})",flush=True)
if not subst_real:
    print(f"[go3-s1] substitution NOT confirmed under this injection -- if handled=0/wq=0, retry CUDA_INJECTION64_PATH; if NVRTC/cublasLt resolution failed, diagnose (do NOT conclude intercept-only yet)",flush=True)
sys.stdout.flush(); os._exit(0)
