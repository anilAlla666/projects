#!/usr/bin/env python3
# INCREMENT-3b PROBE-FIRST: NAME the corrupt carried object after a diverging cross-model swap.
# Established (do NOT re-run): NOT pager (cksum stable + eager-fresh 48/48 correct); cache-aliasing not sole cause.
# Decisive sub-tests at the divergence:
#   (A) compare CARRIED cache KV[0:P] bytes vs a FRESH prefill's KV[0:P] -> is the static prefill-KV corrupt?
#   (B) rebuild-GRAPH-only over the EXISTING cache (NO re-prefill) -> does serve_burst restore KL=0? (graph was corrupt)
#   (C) re-capture (re-prefill + rebuild graph; the known masking fix) -> KL=0 (control)
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
pager=CipherPager(); models={}; solos={}
for name,path,key in SPECS:
    e=PagerGraphModel(pager,path,model_key=key).load(); e.capture(PROMPT,N); solos[name]=e.eager_burst(N); e.evict()
    models[name]=e
resident=[]; K=2
def make_resident(name):
    if name in resident: return
    if len(resident)>=K: models[resident.pop(0)].evict()
    models[name].restore(); resident.append(name)
def serve(name):
    make_resident(name); e=models[name]
    try:
        g=e.serve_burst(N); return sum(a==b for a,b in zip(g,solos[name]))
    except Exception: return -1
# drive churn until qwen2 diverges
seq=["qwen2","llama","tiny","qwen2"]
mt=None
for name in seq: mt=serve(name); print(f"  serve {name}: {mt}/{N}", flush=True)
import time
e=models["qwen2"]; P=e.Plen
print(f"\n[qwen2 diverged at {mt}/{N}] -- pinning the corrupt carried object:", flush=True)

# (B FIRST, no intervening eager): rebuild GRAPH ONLY over the existing cache (no re-prefill) -> serve
@torch.no_grad()
def rebuild_graph_only():
    e.sin.fill_(e._clamp(e.t1)); e.spos.fill_(e.Plen)
    e.graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(e.graph):
        o=e.m(e.sin, cache_position=e.spos, past_key_values=e.cache, use_cache=True); e.slog=o.logits
t0=time.time(); rebuild_graph_only(); gb_ms=(time.time()-t0)*1000
try:
    g=e.serve_burst(N); mtB=sum(a==b for a,b in zip(g,solos["qwen2"]))
    print(f"  (B) rebuild-GRAPH-only (reuse cache, {gb_ms:.0f}ms): serve {mtB}/{N}  -> {'RESTORES KL=0 (GRAPH was the corrupt object; minimal fix = rebuild graph over the intact cache, no re-prefill)' if mtB==N else 'still DIVERGES'}", flush=True)
except Exception as ex:
    mtB=-1; print(f"  (B) rebuild-GRAPH-only: FAULT {type(ex).__name__} -> graph-only insufficient", flush=True)

# (A) carried cache KV[0:P] vs fresh prefill KV[0:P]  (direct bytes) -- confirms the KV was intact
@torch.no_grad()
def fresh_prefill_cache():
    pids=e.tok(PROMPT, return_tensors="pt").input_ids.cuda(); Pp=pids.shape[1]
    c=StaticCache(config=e.m.config, max_cache_len=e.Lc)
    e.m(pids, cache_position=torch.arange(Pp,device="cuda"), past_key_values=c, use_cache=True)
    return c, Pp
fc,Pp=fresh_prefill_cache(); maxdiff=0.0
for i in range(len(e.cache.layers)):
    maxdiff=max(maxdiff, float((e.cache.layers[i].keys[:,:,0:Pp,:]-fc.layers[i].keys[:,:,0:Pp,:]).abs().max()),
                         float((e.cache.layers[i].values[:,:,0:Pp,:]-fc.layers[i].values[:,:,0:Pp,:]).abs().max()))
print(f"  (A) carried cache KV[0:P] vs fresh prefill: max|Δ|={maxdiff:.4f}  -> {'CACHE-KV CORRUPT' if maxdiff>0.05 else 'CACHE-KV INTACT (confirms GRAPH was the object)'}", flush=True)
sys.stdout.flush(); os._exit(0)
