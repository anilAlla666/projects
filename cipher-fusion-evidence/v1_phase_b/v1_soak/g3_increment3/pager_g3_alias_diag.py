#!/usr/bin/env python3
# DISCRIMINATING RUN (advisor): is the multi-model swap OOB a cache aliasing/use-after-free, or a position bug?
# Non-re-capture router serving a swap sequence; log spos + each model's StaticCache KV data_ptr/shape before each
# serve. Outcomes: (a) two models' cache ranges OVERLAP / a model's cache ptr/shape changed -> ALIASING;
# (b) spos OOB -> position bug. CUDA_LAUNCH_BLOCKING=1, CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
pager=CipherPager(); models={}; solos={}
def kvinfo(e):
    k=e.cache.layers[0].keys; return k.data_ptr(), tuple(k.shape), e.cache.max_cache_len
for name,path,key in SPECS:
    e=PagerGraphModel(pager,path,model_key=key).load(); e.capture(PROMPT,N); solos[name]=e.eager_burst(N); e.evict()
    models[name]=e
# log each model's captured-cache KV range after setup (BEFORE any restore) -> overlap check
print("[setup] captured-cache KV pointers per model:", flush=True)
ptrs={}
for name in models:
    dp,shp,mcl=kvinfo(models[name]); ptrs[name]=dp
    print(f"  {name}: layers[0].keys data_ptr={hex(dp)} shape={shp} max_cache_len={mcl}", flush=True)
# pairwise proximity (a 7B cache layer key ~ B*H*Lc*D*2 bytes; overlap if ptrs very close)
names=list(models)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        d=abs(ptrs[names[i]]-ptrs[names[j]])
        print(f"  |ptr({names[i]})-ptr({names[j]})| = {d} bytes ({'SAME/OVERLAP-RISK' if d< (4*131*64*2*8) else 'distinct'})", flush=True)

resident=[]; K=2
def make_resident(name):
    if name in resident: return
    if len(resident)>=K:
        v=resident.pop(0); models[v].evict()
    models[name].restore(); resident.append(name)   # NO re-capture (diagnostic)
def serve(name):
    make_resident(name); e=models[name]
    dp,shp,mcl=kvinfo(e); print(f"  serve {name}: spos={e.spos.item()} cache.keys data_ptr={hex(dp)} shape[2]={shp[2]} max_cache_len={mcl}", flush=True)
    try:
        g=e.serve_burst(N); mt=sum(a==b for a,b in zip(g,solos[name])); print(f"    -> {mt}/{N} {'KL=0' if mt==N else 'DIVERGES'}", flush=True)
    except Exception as ex:
        dp2,shp2,_=kvinfo(e); print(f"    -> FAULT spos={e.spos.item()} (max_cache_len={mcl}, spos in-bounds={e.spos.item()<mcl}) cache.keys NOW data_ptr={hex(dp2)} shape={shp2}", flush=True)
        raise
for name in ["qwen2","llama","tiny","qwen2","llama","tiny","qwen2"]:
    serve(name)
sys.stdout.flush(); os._exit(0)
