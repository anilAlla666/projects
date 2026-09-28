#!/usr/bin/env python3
# D.9 TRAINING-PATH INTERCEPT PROBE (measure-only). CIPHER ACTIVE (default injection; NO unset,
# NO monkeypatch). Does CIPHER's driver-boundary GOT-patch FIRE on plain-torch training / large-batch
# GEMMs (torch nn.Linear -> public cuBLAS), or bypass (cublasLt/nvjet) like vLLM? The decisive signal
# is CIPHER's own teardown telemetry: MATMUL calls + CLASSIFY gemm_total/max_n (max_n large => big
# GEMMs reach cublasGemmEx = actuatable; max_n small => big GEMMs bypass).
import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("PROBE_START inj=[%s] marlin=[%s]" % (os.environ.get("CUDA_INJECTION64_PATH",""), os.environ.get("CIPHER_MARLIN","")), flush=True)

# ---- Part A: REAL torch training step (TinyLlama fwd+bwd+optimizer; fits one H100) ----
tl="/home/ubuntu/models/TinyLlama-1.1B"
m=AutoModelForCausalLM.from_pretrained(tl, dtype=torch.bfloat16).cuda().train()
opt=torch.optim.AdamW(m.parameters(), lr=1e-5)
B,S=8,512
for step in range(3):
    ids=torch.randint(0,30000,(B,S),device="cuda")
    out=m(ids, labels=ids)          # real CE loss
    out.loss.backward()             # backward GEMMs (the training caller path)
    opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
print("TRAIN_3_STEPS_DONE TinyLlama B=%d S=%d (fwd+bwd+AdamW)"%(B,S), flush=True)
del m, opt; torch.cuda.empty_cache()

# ---- Part B: Mistral-7B B=64 large-batch FORWARD (big compute-bound GEMMs, plain torch) ----
ms="/home/ubuntu/models/Mistral-7B-v0.1"
m2=AutoModelForCausalLM.from_pretrained(ms, dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    ids=torch.randint(0,30000,(64,512),device="cuda")   # B=64 x S=512 = 32768 tokens, big GEMMs
    for _ in range(3):
        m2(ids); torch.cuda.synchronize()
print("MISTRAL_B64_FORWARD_DONE (big GEMMs N=4096/14336 via torch nn.Linear)", flush=True)
print("PROBE_END (CIPHER teardown counters follow at exit)", flush=True)
