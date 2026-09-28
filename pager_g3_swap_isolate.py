#!/usr/bin/env python3
# Isolate the router OOB: which step of capture->eager_burst->evict->restore->serve_burst breaks (ONE model)?
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
NAME=sys.argv[1] if len(sys.argv)>1 else "Qwen2-7B"
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
e=PagerGraphModel(CipherPager(), f"/home/ubuntu/models/{NAME}").load()
e.capture(PROMPT, N)
def tryserve(tag):
    try:
        g=e.serve_burst(N); print(f"  [{tag}] serve_burst OK, first5={g[:5]}", flush=True); return g
    except Exception as ex:
        print(f"  [{tag}] serve_burst FAULT: {type(ex).__name__} {str(ex)[:80]}", flush=True); return None
print(f"[{NAME}] router order: capture -> eager_burst -> evict -> restore -> serve_burst (FIRST serve)", flush=True)
# match the router EXACTLY: eager_burst (solo), evict, restore, then the FIRST serve_burst
_=e.eager_burst(N)
e.evict()
import torch as _t; f0=_t.cuda.mem_get_info()[0]
e.restore(); f1=_t.cuda.mem_get_info()[0]
print(f"  restore brought back {(f1-f0)/(1<<30):.2f}GB... wait f0 is post-evict", flush=True)
gC=tryserve("router-order: eager->evict->restore->serve(first)")
sys.stdout.flush(); os._exit(0)
