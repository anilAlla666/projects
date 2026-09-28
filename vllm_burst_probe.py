#!/usr/bin/env python3
# G-O1 on REAL dispatch: measure vLLM graph-decode burst_t (NOT HF generate). The 4.7s/64-tok was an HF-generate
# strawman; vLLM uses CUDA-graph decode (no Python-per-token). Confirm burst_t drops toward ~150-300ms ->
# agents/GPU = (burst+idle)/burst_t jumps from ~2 to the real number. + same-model continuous-batching throughput.
import os, sys, time
os.environ["VLLM_USE_DEEP_GEMM"]="0"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
os.environ["VLLM_PLUGINS"]=""
import torch
def used(): f,t=torch.cuda.mem_get_info(); return (t-f)/(1<<30)
if __name__=="__main__":
    from vllm import LLM, SamplingParams
    t0=time.time()
    # graphs ON (enforce_eager=False is default) -> CUDA-graph decode
    llm=LLM(model="/home/ubuntu/models/Mistral-7B-v0.1", dtype="float16", gpu_memory_utilization=0.55,
            max_model_len=2048, enforce_eager=False)
    load_t=time.time()-t0
    prompt="The history of artificial intelligence began in the"
    # warmup (triggers graph capture)
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)
    # (1) single-agent 64-token decode burst on graph-decode
    sp=SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    ts=[]
    for _ in range(3):
        t=time.time(); o=llm.generate([prompt], sp, use_tqdm=False); ts.append(time.time()-t)
    burst=min(ts); ntok=len(o[0].outputs[0].token_ids)
    print(f"RESULT vLLM graph-decode 64-tok burst: {burst*1000:.0f}ms ({ntok} tok) -> {ntok/burst:.0f} tok/s/agent  (HF-generate was 4172ms/15tok)",flush=True)
    for idle in [2,3,5]:
        print(f"RESULT   agents/GPU = (burst+idle)/burst at idle={idle}s -> ~{(burst+idle)/burst:.0f}  (HF-generate gave ~2)",flush=True)
    # (2) same-model continuous batching: N concurrent requests, aggregate throughput (vLLM's G1∩G3 lever)
    for N in [1,8,32,64]:
        sp2=SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
        t=time.time(); outs=llm.generate([prompt]*N, sp2, use_tqdm=False); dt=time.time()-t
        tot=sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"RESULT   same-model continuous-batch N={N:2d}: {tot} tok in {dt*1000:.0f}ms -> {tot/dt:.0f} aggregate tok/s",flush=True)
    print(f"RESULT load_time={load_t:.0f}s used_HBM={used():.1f}GiB (graphs on)",flush=True)
    sys.stdout.flush(); os._exit(0)
