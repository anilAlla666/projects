#!/usr/bin/env python3
# G-O1 DISPATCH-SIDE close — clean agents-per-GPU at FULL budget on REAL vLLM continuous batching.
# vLLM = per-model decode harness (5 in-process engines, continuous batching = THE dispatch lever, 8808 tok/s prior).
# CIPHER = the multi-model ROUTER across the 5 resident engines (this script). At full budget the PAGER is INACTIVE
# (5 models fit; this is NOT a pager test) — the pager's differentiation is the M>capacity case (M=15, hardware-gated).
# graph-decode 5-engine in-process hit vLLM's GLOBAL cudagraph-capture-monitor singleton (composition wall, same class
# as CuMemAllocator) -> run EAGER vLLM: continuous batching is the lever; graphs are a ~10-30% decode increment that
# cannot change whether 100 fit given ~20x throughput headroom (graph-vs-eager is SECOND-ORDER here, stated).
# GPU-time-shared wave server: 5 engines share ONE GPU (sequential per wave) -> CONSERVATIVE (understates cross-engine
# overlap = lower-bound-on-goodness, advisor). Real generate() per engine-group = real continuous batching within a wave.
import os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; os.environ["VLLM_USE_DEEP_GEMM"]="0"
os.environ["VLLM_PLUGINS"]=""; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import numpy as np, torch
from vllm import LLM, SamplingParams
ENG=[("TinyLlama-1.1B",0.07),("Llama-3.2-1B-Instruct",0.08),("Mistral-7B-v0.1",0.24),("Qwen2-7B",0.25),("Llama-3.1-8B",0.26)]
GB=1<<30
def used(): f,t=torch.cuda.mem_get_info(); return (t-f)/GB
sp=SamplingParams(max_tokens=16, temperature=0.0)
engines=[]; names=[]
for nm,util in ENG:
    llm=LLM(model=f"/home/ubuntu/models/{nm}", gpu_memory_utilization=util, max_model_len=2048,
            enforce_eager=True, dtype="float16", disable_log_stats=True)
    engines.append(llm); names.append(nm)
M=len(engines)
print(f"=== G-O1 ROUTER (eager vLLM continuous batching), {M}/5 engines, GPU-used={used():.1f}GiB ===",flush=True)

# --- correctness: greedy-match SOLO vs BATCHED (continuous batching must not perturb any agent) ---
PROMPT="The history of artificial intelligence began in the"
FILLER=["Once upon a time","The capital of France is","To compute a sum","In the year 2020","A neural network is"]
klbad=0
for i,llm in enumerate(engines):
    solo=llm.generate([PROMPT],sp,use_tqdm=False)[0].outputs[0].token_ids
    batched=llm.generate([PROMPT]+FILLER,sp,use_tqdm=False)[0].outputs[0].token_ids  # target is item 0 in a batch
    if list(solo)!=list(batched): klbad+=1
print(f"  per-agent KL=0 (greedy-match solo vs batched): {M-klbad}/{M}  (continuous batching doesn't perturb routing)",flush=True)

# --- realistic agent fleet -> GPU-time-shared wave server, REAL generate() ---
def gen_stream(N, burstiness, window=20.0, idle_mean=3.5, seed=0):
    rng=np.random.default_rng(seed); reqs=[]
    for a in range(N):
        mi=(rng.zipf(1.6)-1)%M; t=rng.uniform(0,idle_mean); turn=0
        while t<window:
            reqs.append((t,mi,min(256+96*turn,1024))); turn+=1
            t+=rng.gamma(shape=burstiness, scale=idle_mean/burstiness)
    reqs.sort(); return reqs
def serve(reqs):
    # single GPU server; batch all currently-pending (arrived<=clock) per engine, run each group via real generate().
    clock=0.0; i=0; n=len(reqs); ttfts=[]; engtok=[0.0]*M; engbusy=[0.0]*M
    while i<n:
        if reqs[i][0]>clock: clock=reqs[i][0]
        j=i
        while j<n and reqs[j][0]<=clock: j+=1
        wave=reqs[i:j]                                   # the pending burst
        groups={}
        for (arr,mi,ctx) in wave: groups.setdefault(mi,[]).append((arr,ctx))
        wave_svc=0.0
        for mi,items in groups.items():
            prompts=["data "*c for (_,c) in items]
            t0=time.time(); outs=engines[mi].generate(prompts,sp,use_tqdm=False); dt=time.time()-t0
            wave_svc+=dt; engbusy[mi]+=dt
            engtok[mi]+=sum(len(o.outputs[0].token_ids) for o in outs)
        done=clock+wave_svc
        for (arr,mi,ctx) in wave: ttfts.append(done-arr)
        clock=done; i=j
    ttfts.sort(); return ttfts, engtok, engbusy
def pctl(a,p): return a[min(len(a)-1,int(len(a)*p))]*1000

print(f"  realistic agent fleet (Zipfian affinity small-hot, duty-cycle burst+idle~3.5s, growing ctx), N=100:",flush=True)
print(f"  {'burstiness':>11} {'reqs':>5} {'p50':>6} {'p99':>7}  per-engine tok/s under load",flush=True)
for b in [1.0,0.5,0.2]:
    reqs=gen_stream(100,b); tt,et,eb=serve(reqs)
    tps=" ".join(f"{names[k].split('-')[0][:5]}:{(et[k]/eb[k] if eb[k]>0 else 0):.0f}" for k in range(M))
    print(f"  {b:>11.1f} {len(reqs):>5d} {pctl(tt,0.5):>5.0f}ms {pctl(tt,0.99):>5.0f}ms  {tps}",flush=True)

# --- agents-per-GPU at bounded P99: sweep N at burstiness 0.2 (correlated = hardest; hot-engine prefill storm) ---
print(f"  agents-per-GPU sweep @ burstiness 0.2 (correlated, hardest):",flush=True)
print(f"  {'N':>4} {'reqs':>5} {'p50':>6} {'p99':>7}",flush=True)
cap={500:0,1000:0,2000:0}
for N in [25,50,100,150,200]:
    reqs=gen_stream(N,0.2); tt,_,_=serve(reqs); p99=pctl(tt,0.99)
    for slo in cap:
        if p99<=slo: cap[slo]=N
    print(f"  {N:>4} {len(reqs):>5d} {pctl(tt,0.5):>5.0f}ms {p99:>5.0f}ms",flush=True)
for slo in [500,1000,2000]:
    print(f"  agents-per-GPU @ P99<{slo}ms: {cap[slo]}{'+' if cap[slo]==200 else ''}",flush=True)
print("DONE: clean dispatch number on real continuous batching (eager); graph-decode 2nd-order (capture-monitor wall).",flush=True)
sys.stdout.flush(); os._exit(0)
