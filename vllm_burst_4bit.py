#!/usr/bin/env python3
# Convert the distinct-model 4-bit agent count from a 4x hand-wave to MEASURED: vLLM graph-decode burst_t at 4-bit
# (bitsandbytes). 4-bit reads 1/4 the bytes/tok -> decode faster -> smaller burst -> more distinct-model agents.
import os, sys, time
os.environ["VLLM_USE_DEEP_GEMM"]="0"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING"); os.environ["VLLM_PLUGINS"]=""
import torch
def used(): f,t=torch.cuda.mem_get_info(); return (t-f)/(1<<30)
if __name__=="__main__":
    from vllm import LLM, SamplingParams
    try:
        llm=LLM(model="/home/ubuntu/models/Mistral-7B-v0.1", dtype="float16", quantization="bitsandbytes",
                gpu_memory_utilization=0.55, max_model_len=2048, enforce_eager=False)
    except Exception as e:
        print(f"4-bit vLLM load FAILED: {str(e)[:200]}",flush=True)
        print("FALLBACK: fp16 burst was 383ms; 4-bit reads 1/4 bytes/tok -> decode ~2-4x faster -> burst ~100-190ms (ESTIMATE)",flush=True)
        sys.stdout.flush(); os._exit(0)
    p="The history of artificial intelligence began in the"
    llm.generate([p], SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)  # warmup/capture
    sp=SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
    ts=[]
    for _ in range(3):
        t=time.time(); o=llm.generate([p], sp, use_tqdm=False); ts.append(time.time()-t)
    burst=min(ts); n=len(o[0].outputs[0].token_ids)
    print(f"RESULT vLLM 4-bit graph-decode 64-tok burst: {burst*1000:.0f}ms -> {n/burst:.0f} tok/s/agent (fp16 was 383ms)",flush=True)
    for idle in [2,3,5]:
        print(f"RESULT   distinct-model agents/GPU at idle={idle}s -> ~{(burst+idle)/burst:.0f}",flush=True)
    print(f"RESULT used_HBM={used():.1f}GiB (4-bit, one model + KV + graphs)",flush=True)
    sys.stdout.flush(); os._exit(0)
