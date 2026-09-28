#!/usr/bin/env python3
# SM-PACK lever sizing: does filling the idle GPU with concurrent tenants lift AGGREGATE throughput?
# Decisive economic test (replaces occupancy proxies): N Mistral-7B replicas, B=8 each, eager decode.
#  - SERIAL: tenants run one-after-another (no packing) -> aggregate == solo (baseline).
#  - CONCURRENT: tenants interleaved on N non-default streams, NO host sync mid-loop -> overlap fills idle.
# concurrent_wall/solo_wall ~1 => perfect fill (N x aggregate); ~N => no overlap.
# Discriminator: sample power+gpu% during concurrent — saturation at ~150W/low-util = LAUNCH-CPU ceiling
# (GPU still idle, needs graph/MPS), NOT GPU-full. Correctness: each tenant's tokens == solo (Mem #11).
import os, time, threading, ctypes
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
import pynvml; pynvml.nvmlInit(); H=pynvml.nvmlDeviceGetHandleByIndex(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
M="/home/ubuntu/models/Mistral-7B-v0.1"
B=int(os.environ.get("B","8")); K=int(os.environ.get("K","64")); MAXN=int(os.environ.get("MAXN","4"))
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token; tok.padding_side="left"
PROMPT="The history of computing spans several distinct eras, each shaped by"
ids0=tok([PROMPT]*B,return_tensors="pt").input_ids.cuda(); plen=ids0.shape[1]

class Sampler:
    def __init__(s): s.on=False; s.p=[]; s.u=[]
    def _l(s):
        while s.on:
            try: s.p.append(pynvml.nvmlDeviceGetPowerUsage(H)/1000.0); s.u.append(pynvml.nvmlDeviceGetUtilizationRates(H).gpu)
            except: pass
            time.sleep(0.02)
    def __enter__(s): s.on=True; s.t=threading.Thread(target=s._l); s.t.start(); return s
    def __exit__(s,*a): s.on=False; s.t.join()
    def stat(s): return (float(np.mean(s.p)) if s.p else 0, float(np.max(s.u)) if s.u else 0)

print(f"loading up to {MAXN} Mistral-7B replicas...", flush=True)
reps=[]; streams=[]
for i in range(MAXN):
    try:
        m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16,device_map="cuda").eval()
        reps.append(m); streams.append(torch.cuda.Stream())
        print(f"  replica {i} loaded ({torch.cuda.memory_allocated()/2**30:.1f} GiB used)", flush=True)
    except RuntimeError as e:
        print(f"  replica {i} OOM -> capping N at {i}", flush=True); break
N_AVAIL=len(reps)

def new_tenant(t):
    cache=StaticCache(config=reps[t].config,max_batch_size=B,max_cache_len=plen+K+4,device="cuda",dtype=torch.float16)
    return {"ids":ids0.clone(),"cpos":torch.arange(plen,device="cuda"),"cache":cache,"toks":[],"prefilled":False}

def step(t,st):
    with torch.cuda.stream(streams[t]):
        with torch.no_grad():
            o=reps[t](input_ids=st["ids"],cache_position=st["cpos"],past_key_values=st["cache"],use_cache=True,return_dict=True)
        nxt=o.logits[:,-1:].argmax(-1)
        st["ids"]=nxt; st["toks"].append(nxt[:,0])           # on-stream, no host sync
        st["cpos"]=st["cpos"][-1:]+1 if not st["prefilled"] else st["cpos"]+1
        st["prefilled"]=True

def run(n, mode):
    sts=[new_tenant(t) for t in range(n)]
    torch.cuda.synchronize()
    with Sampler() as smp:
        t0=time.time()
        if mode=="serial":
            for t in range(n):
                for _ in range(K): step(t,sts[t])
        elif mode=="concurrent":         # single-thread interleave on N streams
            for _ in range(K):
                for t in range(n): step(t,sts[t])
        elif mode=="threads":            # N host threads -> more host issue parallelism
            def worker(t):
                for _ in range(K): step(t,sts[t])
            ths=[threading.Thread(target=worker,args=(t,)) for t in range(n)]
            for th in ths: th.start()
            for th in ths: th.join()
        torch.cuda.synchronize(); dt=time.time()-t0
    W,util=smp.stat()
    agg=n*K*B/dt
    toks=[torch.stack(s["toks"],1).cpu().numpy() for s in sts]   # [B,K] per tenant
    return dt,agg,W,util,toks

# warm up tenant-0 ONCE (pay one-time init: cuBLAS handles, autotune) so the solo baseline isn't cold-biased
_=run(1,"serial")
sdt,sagg,sW,sutil,solo_toks=run(1,"serial")
ref=solo_toks[0]
print(f"[solo N=1 warm] tok/s={sagg:.0f} wall={sdt:.2f}s meanW={sW:.0f} maxutil={sutil}%", flush=True)
print(f"{'N':>2} {'mode':>11} {'agg tok/s':>10} {'wall':>6} {'meanW':>6} {'maxutil':>7} {'vs_solo':>8} {'correct':>8}")
print(f"{1:>2} {'solo':>11} {sagg:>10.0f} {sdt:>6.2f} {sW:>6.0f} {sutil:>7} {'1.00x':>8} {'ref':>8}")
for n in range(2, N_AVAIL+1):
    for mode in ["serial","concurrent","threads"]:
        dt,agg,W,util,toks=run(n,mode)
        correct=all((toks[t]==ref).all() for t in range(n))   # each tenant == solo ref (co-residence unperturbed)
        print(f"{n:>2} {mode:>11} {agg:>10.0f} {dt:>6.2f} {W:>6.0f} {util:>7} {agg/sagg:>7.2f}x {str(correct):>8}", flush=True)
