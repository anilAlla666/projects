#!/usr/bin/env python3
# INCREMENT-1 CONTROL: the symmetric skeptic check the advisor flagged. The first async run used prompts "data "*ctx
# (identical repeated token) -> all requests share a long prefix -> vLLM enable_prefix_caching=True (default) ELIDES
# prefill after the first request = a FAVORABLE artifact (real agents have DISTINCT contexts + pay full prefill).
# Control: enable_prefix_caching=FALSE + DISTINCT random token-id prompts per request (full prefill every turn).
# Also vary burst length (16 vs 128 decode tokens) -- the other load-bearing envelope assumption. Verdict: if p99
# stays sub-second with prefill un-elided + realistic burst, the finding is real; if it jumps, prefill-elision was it.
import os, sys, time, asyncio
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"; os.environ["VLLM_USE_DEEP_GEMM"]="0"
os.environ["VLLM_PLUGINS"]=""; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt
ENG=[("TinyLlama-1.1B",0.07),("Mistral-7B-v0.1",0.24),("Llama-3.1-8B",0.26)]
engines=[]; names=[]
for nm,util in ENG:
    args=AsyncEngineArgs(model=f"/home/ubuntu/models/{nm}", gpu_memory_utilization=util, max_model_len=2048,
                         enforce_eager=True, dtype="float16", disable_log_stats=True,
                         enable_prefix_caching=False)   # <-- CONTROL: no prefill elision
    engines.append(AsyncLLMEngine.from_engine_args(args)); names.append(nm)
M=len(engines)
print(f"=== INCREMENT-1 CONTROL: {M} async engines, prefix_caching=OFF, DISTINCT token prompts ===",flush=True)

async def one_req(eng, ids, sp, rid, arrival, t0, out, mi):
    d=arrival-(time.time()-t0)
    if d>0: await asyncio.sleep(d)
    final=None
    async for o in eng.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=rid): final=o
    out.append((mi, time.time()-(t0+arrival)))

async def main():
    def stream(N, burstiness, window=15.0, idle_mean=3.5, seed=0):
        rng=np.random.default_rng(seed); reqs=[]
        for a in range(N):
            mi=(rng.zipf(1.6)-1)%M; t=rng.uniform(0,idle_mean); turn=0
            while t<window:
                ctx=min(256+96*turn,1024)
                ids=rng.integers(10,30000,size=ctx).tolist()   # DISTINCT context per request -> full prefill
                reqs.append((t,mi,ids)); turn+=1
                t+=rng.gamma(shape=burstiness, scale=idle_mean/burstiness)
        reqs.sort(key=lambda r:r[0]); return reqs
    def pctl(a,p): a=sorted(a); return a[min(len(a)-1,int(len(a)*p))]*1000 if a else 0
    print(f"  {'N':>4} {'b':>4} {'maxtok':>6} {'reqs':>5} {'p50':>7} {'p99':>7} | per-engine p50/p99 (ms)",flush=True)
    for (N,b,mt) in [(100,0.5,16),(100,0.5,128)]:
        sp=SamplingParams(max_tokens=mt, temperature=0.0)
        reqs=stream(N,b); out=[]; t0=time.time()
        tasks=[asyncio.create_task(one_req(engines[mi], ids, sp, f"r{N}_{b}_{mt}_{i}", arr, t0, out, mi))
               for i,(arr,mi,ids) in enumerate(reqs)]
        await asyncio.gather(*tasks)
        alll=[l for _,l in out]
        per=" ".join(f"{names[k].split('-')[0][:5]}:{pctl([l for m,l in out if m==k],0.5):.0f}/{pctl([l for m,l in out if m==k],0.99):.0f}" for k in range(M))
        print(f"  {N:>4} {b:>4} {mt:>6} {len(reqs):>5} {pctl(alll,0.5):>6.0f}ms {pctl(alll,0.99):>6.0f}ms | {per}",flush=True)
    print("DONE control: prefix-caching OFF + distinct prompts (+128-tok burst). Compare vs the prefix-cached run (171/479).",flush=True)
asyncio.run(main())
sys.stdout.flush(); os._exit(0)
