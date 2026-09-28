#!/usr/bin/env python3
# Isolate: serve_burst -> evict -> restore -> serve_burst AGAIN (re-serve after a 2nd page cycle). Does the
# SECOND serve fault? (the router re-serves models across repeated evict/restore swaps.)
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
e=PagerGraphModel(CipherPager(),"/home/ubuntu/models/Qwen2-7B",0xC2).load()
e.capture(PROMPT,N); solo=e.eager_burst(N); e.evict()
RECAP=os.environ.get("RECAP","0")=="1"
def serve(tag):
    try:
        e.restore()
        if RECAP: e.capture(PROMPT,N)   # FIX: re-capture the graph on restore (weights bit-identical per cksum)
        ck=e.region_cksum()
        g=e.serve_burst(N); mt=sum(a==b for a,b in zip(g,solo))
        print(f"  [{tag}] cksum={hex(ck)} -> {mt}/{N} {'KL=0' if mt==N else 'DIVERGES@'+str(next((i for i,(x,y) in enumerate(zip(g,solo)) if x!=y),-1))}", flush=True); return True
    except Exception as ex:
        print(f"  [{tag}] FAULT {type(ex).__name__} {str(ex)[:80]}", flush=True); return False
# serve #1 (restore from setup-evict)
serve("serve#1 (restore->serve)")
e.evict()
serve("serve#2 (evict->restore->serve)")
e.evict()
serve("serve#3 (evict->restore->serve)")
sys.stdout.flush(); os._exit(0)
