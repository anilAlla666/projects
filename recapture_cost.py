#!/usr/bin/env python3
# G-O5 decision orientation: (a) per-recapture cost (capture+instantiate latency) vs burst cadence,
# (b) replay latency per bucket, (c) does cu13/torch expose conditional graph nodes?
import os, time, torch
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
M="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16,device_map="cuda").eval()

def prefill(B):
    ids=tok(["The history of computing spans several distinct eras, each shaped by"]*B,return_tensors="pt").input_ids.cuda()
    plen=ids.shape[1]
    c=StaticCache(config=model.config,max_batch_size=B,max_cache_len=plen+200,device="cuda",dtype=torch.float16)
    with torch.no_grad(): o=model(input_ids=ids,cache_position=torch.arange(plen,device="cuda"),past_key_values=c,use_cache=True,return_dict=True)
    return c, o.logits[:,-1:].argmax(-1), plen

print("=== (a) capture+instantiate latency (the per-RECAPTURE cost) + (b) replay latency, per bucket ===")
for B in (8,32,64,128):
    try:
        c,nt,plen=prefill(B)
    except RuntimeError as e:
        print(f"  B={B}: OOM prefill -> skip"); continue
    iid=nt.clone(); cp=torch.tensor([plen],device="cuda")
    outl=torch.empty(B,1,model.config.vocab_size,device="cuda",dtype=torch.float16)
    side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(4):
            with torch.no_grad(): o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
            outl.copy_(o.logits); iid.copy_(outl.argmax(-1)); cp+=1
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    # TIME the capture+instantiate (what a membership-change recapture costs)
    t0=time.time(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(): o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
        outl.copy_(o.logits); iid.copy_(outl.argmax(-1))
    torch.cuda.synchronize(); cap_ms=(time.time()-t0)*1000
    # replay latency
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(64): cp+=1; g.replay()
    torch.cuda.synchronize(); rep_ms=(time.time()-t0)/64*1000
    print(f"  B={B:>3}: capture+instantiate={cap_ms:7.1f} ms | replay={rep_ms:6.2f} ms/step ({1000/rep_ms:.0f} steps/s)")
    del g, c; torch.cuda.empty_cache()

print()
print("=== (c) cu13/torch conditional/dynamic graph node support? ===")
import torch.cuda
cands=['graph_task_group','conditional','cond','if_else','while_loop','CUDAGraph']
for n in dir(torch.cuda.graphs):
    if any(k in n.lower() for k in ['cond','dynamic','if','while','task']): print("  torch.cuda.graphs.",n)
print("  torch.cuda.CUDAGraph methods:", [m for m in dir(torch.cuda.CUDAGraph) if not m.startswith('__')])
try:
    import ctypes; libcudart=ctypes.CDLL("libcudart.so")
    has=[s for s in ['cudaGraphAddNode','cudaGraphConditionalHandleCreate','cudaGraphAddNode_v2'] if hasattr(libcudart,s)]
    print("  libcudart conditional-node symbols present:", has)
except Exception as e: print("  (libcudart probe failed:",e,")")
