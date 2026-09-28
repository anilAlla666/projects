#!/usr/bin/env python3
# Isolate the BIG-GEMM question: Mistral-7B B=64 forward (N=4096/14336), CIPHER active, many iters
# so the classifier emits its CLASSIFY line with max_n. Decisive: max_n large => big GEMMs reach
# cublasGemmEx (FIRES+ACTUATABLE); max_n small => big GEMMs bypass to cublasLt.
import os, torch
from transformers import AutoModelForCausalLM
print("BIGGEMM_PROBE inj=[%s]"%os.environ.get("CUDA_INJECTION64_PATH",""), flush=True)
m=AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/Mistral-7B-v0.1", dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    for _ in range(15):
        m(torch.randint(0,30000,(64,512),device="cuda")); torch.cuda.synchronize()
print("DONE 15x Mistral B=64xS=512 forward (big GEMMs via torch nn.Linear)", flush=True)
