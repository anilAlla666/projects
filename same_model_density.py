#!/usr/bin/env python3
# PART A: stock vLLM same-model density — N concurrent same-model requests via continuous batching + APC
# (the path vLLM ALREADY delivers). Aggregate tok/s + tok/W + GPU util at N=4/16/50/100. Trap-guard baseline.
import os, time, threading, sys
import numpy as np
os.environ.setdefault("VLLM_USE_DEEP_GEMM","0"); os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
def main():
    import pynvml; pynvml.nvmlInit(); H=pynvml.nvmlDeviceGetHandleByIndex(0)
    from vllm import LLM, SamplingParams
    MODEL=os.environ.get("MODEL","/home/ubuntu/models/TinyLlama-1.1B")
    NEW=int(os.environ.get("NEW","64"))
    class P:
        def __init__(s): s.on=False; s.p=[]; s.u=[]
        def _l(s):
            while s.on:
                try: s.p.append(pynvml.nvmlDeviceGetPowerUsage(H)/1000.0); s.u.append(pynvml.nvmlDeviceGetUtilizationRates(H).gpu)
                except: pass
                time.sleep(0.02)
        def __enter__(s): s.on=True; s.t=threading.Thread(target=s._l); s.t.start(); return s
        def __exit__(s,*a): s.on=False; s.t.join()
        def mean(s): return (float(np.mean(s.p)) if s.p else 0, float(np.mean(s.u)) if s.u else 0)
    # agent prompts: shared instruction prefix (exercises APC) + distinct tails (realistic agents)
    PRE="You are a helpful assistant. Answer concisely and accurately. Question: "
    tails=["explain how a GPU schedules warps","summarize the causes of inflation","what is a transformer attention head",
           "describe how TCP congestion control works","outline the water cycle","what makes a good database index"]
    llm=LLM(model=MODEL, dtype="bfloat16", max_num_seqs=128, gpu_memory_utilization=0.85,
            enable_prefix_caching=True, disable_log_stats=True)
    sp=SamplingParams(temperature=0.0, max_tokens=NEW, ignore_eos=True)
    print(f"model={MODEL} NEW={NEW} (vLLM defaults: continuous batching + APC on)")
    print(f"{'N':>4} {'agg tok/s':>10} {'meanW':>6} {'gpu%':>5} {'tok/s/W':>8} {'wall':>6}")
    for N in (4,16,50,100):
        prompts=[PRE+tails[i%len(tails)]+f" (variant {i})" for i in range(N)]
        llm.generate(prompts, sp, use_tqdm=False)  # warm/APC fill
        with P() as pw:
            t0=time.time(); outs=llm.generate(prompts, sp, use_tqdm=False); dt=time.time()-t0
        nout=sum(len(o.outputs[0].token_ids) for o in outs); tps=nout/dt
        W,u=pw.mean()
        print(f"{N:>4} {tps:>10.0f} {W:>6.0f} {u:>5.0f} {tps/W if W else 0:>8.3f} {dt:>6.2f}", flush=True)
if __name__=="__main__": main()
