#!/usr/bin/env python3
# DECISIVE (advisor): does the StaticCache REALLOCATE its KV buffers under cross-model churn? Record each model's
# per-layer cache.keys/values data_ptr AT CAPTURE; compare at the diverging serve (NON-re-capture path). If the ptr
# DIFFERS -> the captured graph holds a DANGLING pointer (realloc) -> fix = PIN the buffers (persistent graphs).
# If STABLE -> not a pointer bug. Also log max_cache_len (the index_copy self_dim_size).
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
pager=CipherPager(); models={}; solos={}; capt_ptrs={}
def ptrs(e): return [(l.keys.data_ptr(), l.values.data_ptr(), l.keys.shape[2]) for l in e.cache.layers]
for name,path,key in SPECS:
    e=PagerGraphModel(pager,path,model_key=key).load(); e.capture(PROMPT,N)
    capt_ptrs[name]=ptrs(e)                      # record AT CAPTURE
    solos[name]=e.eager_burst(N); e.evict(); models[name]=e
resident=[]; K=2
def make_resident(name):
    if name in resident: return
    if len(resident)>=K: models[resident.pop(0)].evict()
    models[name].restore(); resident.append(name)
def cmp_ptrs(name):
    now=ptrs(models[name]); cap=capt_ptrs[name]
    moved=sum(1 for (a,_,_),(b,_,_) in zip(now,cap) if a!=b) + sum(1 for (_,a,_),(_,b,_) in zip(now,cap) if a!=b)
    dimchg=sum(1 for (_,_,a),(_,_,b) in zip(now,cap) if a!=b)
    return moved, dimchg, now[0][0], cap[0][0], now[0][2]
for name in ["qwen2","llama","tiny","qwen2"]:
    make_resident(name); e=models[name]
    moved,dimchg,now0,cap0,mcl=cmp_ptrs(name)
    try:
        g=e.serve_burst(N); mt=sum(a==b for a,b in zip(g,solos[name]))
        print(f"  serve {name}: {mt}/{N} {'KL=0' if mt==N else 'DIVERGES'} | cache.keys[0] capture={hex(cap0)} now={hex(now0)} {'MOVED' if now0!=cap0 else 'stable'} | ptrs-moved={moved} dim-changed={dimchg} max_cache_len={mcl}", flush=True)
    except Exception as ex:
        moved,dimchg,now0,cap0,mcl=cmp_ptrs(name)
        print(f"  serve {name}: FAULT | cache.keys[0] capture={hex(cap0)} now={hex(now0)} {'MOVED' if now0!=cap0 else 'stable'} | ptrs-moved={moved} dim-changed={dimchg}", flush=True)
        break
print(f"\n[VERDICT] {'DANGLING POINTER (StaticCache reallocated buffers under churn -> graph baked stale ptr; FIX=pin buffers, persistent graphs)' if moved>0 else 'POINTERS STABLE (not a realloc/dangling bug; mechanism is something else)'}", flush=True)
sys.stdout.flush(); os._exit(0)
