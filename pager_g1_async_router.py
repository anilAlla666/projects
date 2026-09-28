#!/usr/bin/env python3
# INCREMENT-1 (compose, NOT reimplement): CIPHER asyncio ROUTER (the dispatch loop CIPHER owns) over vLLM
# AsyncLLMEngine x3 in-process (per-model continuous batching = vLLM's lever; eager — graph-decode is increment-2,
# blocked in-process by vLLM's cudagraph-monitor singleton). Payoff = the REAL async p50/p99 SPLIT my conservative
# sum-over-engines wave server understated: does the HOT small model's agents get served sub-second (p50) while only
# the big-model tail is slow (p99)? That split is the decision-relevant number for Anil's engine call.
# vLLM = per-model engine (validator); CIPHER = the cross-model async router. Pager inactive at full budget.
import os, sys, time, asyncio
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; os.environ["VLLM_USE_DEEP_GEMM"]="0"
os.environ["VLLM_PLUGINS"]=""; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
ENG=[("TinyLlama-1.1B",0.07),("Mistral-7B-v0.1",0.24),("Llama-3.1-8B",0.26)]   # 1 small-hot + 2 big tail
sp=SamplingParams(max_tokens=16, temperature=0.0)
engines=[]; names=[]
for nm,util in ENG:
    args=AsyncEngineArgs(model=f"/home/ubuntu/models/{nm}", gpu_memory_utilization=util, max_model_len=2048,
                         enforce_eager=True, dtype="float16", disable_log_stats=True)
    engines.append(AsyncLLMEngine.from_engine_args(args)); names.append(nm)
M=len(engines)
print(f"=== INCREMENT-1 async router: {M} in-process AsyncLLMEngines (eager continuous batching) ===",flush=True)

async def gen_ids(eng, prompt, rid):
    final=None
    async for out in eng.generate(prompt, sp, request_id=rid): final=out
    return list(final.outputs[0].token_ids)

async def one_req(eng, prompt, rid, arrival, t0, out, mi):
    d=arrival-(time.time()-t0)
    if d>0: await asyncio.sleep(d)
    final=None
    async for o in eng.generate(prompt, sp, request_id=rid): final=o
    out.append((mi, time.time()-(t0+arrival)))

PROMPT="The history of artificial intelligence began in the"
FILLER=["Once upon a time","The capital of France is","A neural network is","In the year 2020","To sum two numbers"]
def prompt_ctx(c): return "data "*c

async def main():
    # correctness: greedy-match SOLO vs UNDER-LOAD (continuous batching must not perturb)
    klbad=0
    for i,eng in enumerate(engines):
        solo=await gen_ids(eng, PROMPT, f"solo{i}")
        tasks=[asyncio.create_task(gen_ids(eng, PROMPT, f"tgt{i}"))]+[asyncio.create_task(gen_ids(eng,f,f"fl{i}_{k}")) for k,f in enumerate(FILLER)]
        res=await asyncio.gather(*tasks)
        if res[0]!=solo: klbad+=1
    print(f"  per-agent KL=0 (greedy-match solo vs under-load): {M-klbad}/{M}",flush=True)

    def stream(N, burstiness, window=15.0, idle_mean=3.5, seed=0):
        rng=np.random.default_rng(seed); reqs=[]
        for a in range(N):
            mi=(rng.zipf(1.6)-1)%M; t=rng.uniform(0,idle_mean); turn=0
            while t<window:
                reqs.append((t,mi,min(256+96*turn,1024))); turn+=1
                t+=rng.gamma(shape=burstiness, scale=idle_mean/burstiness)
        reqs.sort(); return reqs
    def pctl(a,p): a=sorted(a); return a[min(len(a)-1,int(len(a)*p))]*1000 if a else 0
    print(f"  realistic agent fleet (Zipfian small-hot, duty-cycle, growing ctx), REAL-TIME async replay:",flush=True)
    print(f"  {'N':>4} {'b':>4} {'reqs':>5} {'p50':>7} {'p99':>7} | per-engine p50/p99 (ms)",flush=True)
    for (N,b) in [(100,0.5),(100,0.2),(200,0.2)]:
        reqs=stream(N,b); out=[]; t0=time.time()
        tasks=[asyncio.create_task(one_req(engines[mi], prompt_ctx(ctx), f"r{N}_{b}_{i}", arr, t0, out, mi))
               for i,(arr,mi,ctx) in enumerate(reqs)]
        await asyncio.gather(*tasks)
        alll=[l for _,l in out]
        per=" ".join(f"{names[k].split('-')[0][:5]}:{pctl([l for m,l in out if m==k],0.5):.0f}/{pctl([l for m,l in out if m==k],0.99):.0f}" for k in range(M))
        print(f"  {N:>4} {b:>4} {len(reqs):>5} {pctl(alll,0.5):>6.0f}ms {pctl(alll,0.99):>6.0f}ms | {per}",flush=True)
    print("DONE: real async p50/p99 split (eager continuous batching). graph-decode (increment-2) = the sub-second lever, blocked in-process.",flush=True)

asyncio.run(main())
sys.stdout.flush(); os._exit(0)
