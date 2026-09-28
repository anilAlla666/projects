#!/usr/bin/env python3
# Minimal 2-model swap isolation: does another model's load/capture/eager/evict between A's capture and A's
# restore+serve corrupt A's captured graph? Try with and without explicit per-graph memory pools.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
pager=CipherPager()
A=PagerGraphModel(pager,"/home/ubuntu/models/Qwen2-7B",0xC2).load(); A.capture(PROMPT,N); soloA=A.eager_burst(N); A.evict()
print("[setup] A(Qwen2) loaded,captured,evicted", flush=True)
B=PagerGraphModel(pager,"/home/ubuntu/models/Llama-3.1-8B",0xC3).load(); B.capture(PROMPT,N); soloB=B.eager_burst(N); B.evict()
print("[setup] B(Llama) loaded,captured,evicted", flush=True)
def tryserve(eng,tag,solo):
    try:
        eng.restore(); g=eng.serve_burst(N)
        mt=sum(a==b for a,b in zip(g,solo)); print(f"  [{tag}] OK {mt}/{N} {'KL=0' if mt==N else 'DIVERGES@'+str(next((i for i,(x,y) in enumerate(zip(g,solo)) if x!=y),-1))}", flush=True); return g
    except Exception as ex:
        print(f"  [{tag}] FAULT: {type(ex).__name__} {str(ex)[:90]}", flush=True); return None
# restore A and serve (A was evicted while B was loaded+captured)
tryserve(A,"restore A -> serve A", soloA)
tryserve(B,"restore B -> serve B", soloB)
sys.stdout.flush(); os._exit(0)
