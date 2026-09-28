# Bounded: does Marlin OBSERVE the big Linear GEMMs + SUBSTITUTE at B=8 (M<=64 regime)?
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
ids=tok(["The history of computing spans"]*8, return_tensors="pt", padding=True).input_ids.cuda()  # B=8
with torch.no_grad():
    for _ in range(20):  # decode steps: each GEMM M=8 (<=64), N/K=4096+ (big projections)
        out=m.generate(ids, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize(); print("B=8 forward done", tuple(out.shape))
