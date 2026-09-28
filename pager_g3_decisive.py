#!/usr/bin/env python3
# DECISIVE RUN (advisor): split weights-VA from cache-contents.
# (1) per-model base_va + live_cksum BEFORE EACH serve -> does ANY model's weight cksum DRIFT from load-time? (pager bug)
# (2) after a diverging swap, EAGER decode of the restored model on a FRESH cache (no graph) -> correct or wrong?
#     eager-fresh correct => weights/VA fine, issue is carried graph/static-KV (fix = re-prefill only).
#     eager-fresh wrong   => page_in not restoring weights under churn (PAGER bug).
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
pager=CipherPager(); models={}; solos={}; ck_load={}
for name,path,key in SPECS:
    e=PagerGraphModel(pager,path,model_key=key).load()
    ck_load[name]=e.region_cksum()                      # cksum WHILE RESIDENT at load (before capture/evict)
    e.capture(PROMPT,N); solos[name]=e.eager_burst(N); e.evict()
    models[name]=e
def clampi(t,V): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def eager_fresh(e):   # eager decode, FRESH cache, NO captured graph -> tests weights/VA only
    m=e.m; dev="cuda"; V=e.V; pids=e.tok(PROMPT, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; Lc=P+N+64
    c=StaticCache(config=m.config, max_cache_len=Lc)
    nt=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    out=[]
    for i in range(N):
        out.append(clampi(nt,V))
        nt=m(torch.tensor([[clampi(nt,V)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
resident=[]; K=2
def make_resident(name):
    if name in resident: return
    if len(resident)>=K: models[resident.pop(0)].evict()
    models[name].restore(); resident.append(name)
def allcksum(tag):
    s=[]
    for nm in models:
        if models[nm].pg.stats(models[nm].rid).state==2:   # only resident regions have valid cksum
            ck=models[nm].region_cksum(); s.append(f"{nm}={'OK' if ck==ck_load[nm] else 'DRIFT'}")
    print(f"  [{tag}] resident-cksum: {' '.join(s)}", flush=True)
seq=["qwen2","llama","tiny","qwen2","llama","tiny"]
for r,name in enumerate(seq):
    make_resident(name); e=models[name]
    allcksum(f"r{r} before serve {name}")
    try:
        g=e.serve_burst(N); mt=sum(a==b for a,b in zip(g,solos[name]))
        print(f"  r{r} serve {name}: graph {mt}/{N} {'KL=0' if mt==N else 'DIVERGES'}", flush=True)
        if mt<N:
            ef=eager_fresh(e); emt=sum(a==b for a,b in zip(ef,solos[name]))
            print(f"    >>> EAGER-FRESH (no graph) {emt}/{N}: {'CORRECT -> weights/VA FINE, carried-graph/KV is the bug (re-prefill fix)' if emt==N else 'WRONG -> PAGER page_in not restoring weights under churn (pager bug)'}", flush=True)
    except Exception as ex:
        print(f"  r{r} serve {name}: FAULT {type(ex).__name__}", flush=True)
        ef=eager_fresh(e); emt=sum(a==b for a,b in zip(ef,solos[name]))
        print(f"    >>> EAGER-FRESH (no graph) {emt}/{N}: {'CORRECT -> graph/KV bug' if emt==N else 'WRONG -> PAGER bug'}", flush=True)
        break
sys.stdout.flush(); os._exit(0)
