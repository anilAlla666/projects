#!/usr/bin/env python3
# PREMISE: is the GPU spatially underutilized during Mistral-7B decode at B=8/32/64?
# Sustained eager generate; an external `nvidia-smi dmon -s u` samples sm% concurrently.
# We print clear START/STOP markers per batch so the dmon log can be sliced by wall-clock.
import os, time, torch
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
M="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16,device_map="cuda").eval()
base="The history of computing spans several distinct eras, each shaped by"
for B in (8,32,64):
    ids=tok([base]*B,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): model.generate(ids,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)  # warm
    torch.cuda.synchronize()
    print(f"PREMISE_START B={B} t={time.time():.2f}",flush=True)
    t0=time.time()
    with torch.no_grad():
        for _ in range(6): model.generate(ids,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt=time.time()-t0
    print(f"PREMISE_STOP B={B} t={time.time():.2f} tok/s={B*64*6/dt:.0f}",flush=True)
    time.sleep(2)  # gap so dmon shows idle between batches
