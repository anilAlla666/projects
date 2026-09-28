#!/usr/bin/env python3
# Is the inc-4 B=8 fault (p2 @22, margin 1.36) a real large per-step logit divergence, or cross-row contamination?
# Measure row-2's per-step logits in a B=8 wave vs p2 solo (same greedy path while tokens match). And co-tenant
# invariance: is p2's output identical across two different B=8 compositions (invariant=numerical, not contamination)?
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer
q=WaveServer(CipherPager(),"/home/ubuntu/models/Qwen2-7B",0xC2); m=q.m; dev="cuda"
POOL=["The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty"]
rows=[q.tok(p,return_tensors="pt").input_ids[0][:16] for p in POOL]; Pc=min(r.shape[0] for r in rows); rows=[r[:Pc] for r in rows]
G=48; V=q.V
def clampi(t): return int(max(0,min(V-1,int(t))))
# manual EAGER batched decode capturing per-step logits for a target row (no graph -> isolate batched-attention numerics)
@torch.no_grad()
def batched_eager(batch_rows, target):
    B=len(batch_rows); P=batch_rows[0].shape[0]; Lc=P+G+8
    pids=torch.stack([r.to(dev) for r in batch_rows])
    c=StaticCache(config=m.config,max_cache_len=Lc)
    lg=m(pids,cache_position=torch.arange(P,device=dev),past_key_values=c,use_cache=True).logits[:,-1]
    toks=[]; logs=[]
    for i in range(G):
        logs.append(lg[target].float().clone()); nt=lg.argmax(dim=-1); toks.append(clampi(nt[target]))
        lg=m(nt.view(B,1),cache_position=torch.tensor([P+i],device=dev),past_key_values=c,use_cache=True).logits[:,-1]
    return toks,logs
# p2 solo (B=1)
st,sl=batched_eager([rows[2]],0)
# p2 in B=8 composition A (rows 0..7, p2 at index 2)
at,al=batched_eager(rows,2)
# p2 in B=8 composition B (different co-tenants: rows 2,5,3,7,1,6,4,0 -> p2 at index 0)
compB=[rows[2],rows[5],rows[3],rows[7],rows[1],rows[6],rows[4],rows[0]]
bt,bl=batched_eager(compB,0)
fd=next((i for i,(x,y) in enumerate(zip(at,st)) if x!=y),-1)
print(f"[p2] solo vs B=8: 1stdiff@{fd}", flush=True)
if fd>=0:
    print(f"  per-step logit max|Δ| (solo vs B=8) up to fd: "+" ".join(f"{float((al[i]-sl[i]).abs().max()):.3f}" for i in range(min(fd+2,len(sl)))), flush=True)
    print(f"  at fd={fd}: |Δ(solo,B8)|={float((al[fd]-sl[fd]).abs().max()):.3f}  solo top1-2 margin={float(sl[fd].topk(2).values[0]-sl[fd].topk(2).values[1]):.3f}", flush=True)
# co-tenant invariance: p2 output identical across comps A and B?
inv=(at==bt); fdab=next((i for i,(x,y) in enumerate(zip(at,bt)) if x!=y),-1)
print(f"[co-tenant] p2 output identical across two B=8 compositions: {inv} (1stdiff@{fdab})  -> {'NOT contamination (deterministic B=8-vs-B=1 numerics)' if inv else 'CONTAMINATION (output depends on co-tenants)'}", flush=True)
sys.stdout.flush(); os._exit(0)
