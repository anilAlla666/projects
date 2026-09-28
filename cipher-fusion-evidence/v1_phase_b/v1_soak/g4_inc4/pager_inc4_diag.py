#!/usr/bin/env python3
# Diagnose the inc-4 K=4 failure: real swap-debt re-emergence vs harness bug.
# (1) are solos DISTINCT across models? (misroute 'tie' red flag) (2) qwen2 batched-vs-solo NO churn -> FAULT? (batched
# bug) (3) qwen2 batched-vs-solo AFTER churn (evict/restore + other model) -> FAULT? (swap debt re-emergence).
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer, classify
G=48
pager=CipherPager()
q=WaveServer(pager,"/home/ubuntu/models/Qwen2-7B",0xC2)
rowq=q.tok("The recipe calls for two cups of flour a pinch of salt and three large", return_tensors="pt").input_ids[0][:16]
sq,mq=q.solo(rowq,G)
l=WaveServer(pager,"/home/ubuntu/models/Llama-3.1-8B",0xC3)
rowl=l.tok("The recipe calls for two cups of flour a pinch of salt and three large", return_tensors="pt").input_ids[0][:16]
sl,ml=l.solo(rowl,G)
print(f"(1) DISTINCT solos? qwen2 first8={sq[:8]}  llama first8={sl[:8]}  same={sq[:8]==sl[:8]}", flush=True)

# precompute qwen2 solos for 8 distinct POOL prompts (the inc-4 heterogeneous-wave condition)
POOL=["The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty"]
qrows=[q.tok(p,return_tensors="pt").input_ids[0][:16] for p in POOL]; Pc=min(r.shape[0] for r in qrows); qrows=[r[:Pc] for r in qrows]
qsolos=[q.solo(r,80) for r in qrows]
gens=[22,35,48,60,72,18,80,40]   # heterogeneous gen_lens (short regime), like inc-4

# (2) qwen2 batched B=8 HETEROGENEOUS prompts + varied gens, NO churn
out,_=q.serve_wave(qrows,gens); f2=0
for b in range(8):
    k,fd,mg=classify(out[b], qsolos[b][0][:gens[b]], qsolos[b][1][:gens[b]]); f2+=(k=="FAULT"); print(f"(2) B=8 hetero[no-churn] p{b} gen{gens[b]}: {k}@{fd} m={mg:.3f}", flush=True)

# (3) same after churn (evict+restore qwen2 with llama active)
q.base.evict(); q.base.restore()
out2,_=q.serve_wave(qrows,gens); f3=0
for b in range(8):
    k,fd,mg=classify(out2[b], qsolos[b][0][:gens[b]], qsolos[b][1][:gens[b]]); f3+=(k=="FAULT"); print(f"(3) B=8 hetero[AFTER churn] p{b} gen{gens[b]}: {k}@{fd} m={mg:.3f}", flush=True)
print(f"\n[DIAG] solos-distinct={sq[:8]!=sl[:8]}; B8-hetero-no-churn-faults={f2}/8; B8-hetero-after-churn-faults={f3}/8", flush=True)
print(f"  -> {'BATCHED B=8 hetero bug (not churn)' if f2>0 else ('SWAP DEBT at B=8 after churn' if f3>0 else 'both clean -- inc-4 fault is elsewhere (wave composition?)')}", flush=True)
sys.stdout.flush(); os._exit(0)
