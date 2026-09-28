#!/usr/bin/env python3
# Minimal REAL vLLM eager decode driver (real model, real KV cache, continuous batching, real attention).
# Used for: (a) discovery (which cuBLAS entry fires), (b) the detector/injection/0-FP runs, (c) tok/s.
# enforce_eager so the cuBLAS interceptor sees every steady-state decode GEMM (cudagraph would replay them).
import sys, os, time, json
MODE=sys.argv[1] if len(sys.argv)>1 else "decode"   # decode | bench
EAGER = os.environ.get("RV_EAGER","1")=="1"
MODEL = os.environ.get("RV_MODEL","mistralai/Mistral-7B-v0.1")
B = int(os.environ.get("RV_BATCH","8")); OUT=int(os.environ.get("RV_OUT","64"))
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=0.85, max_model_len=2048,
          dtype="float16", disable_log_stats=True)
prompts=["The history of computing began "*4]*B
sp=SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)   # warmup (also lets the shim see steady-state decode)
if MODE=="bench":
    best=None
    for _ in range(2):
        t0=time.perf_counter(); o=llm.generate(prompts,sp,use_tqdm=False); dt=time.perf_counter()-t0
        gt=sum(len(x.outputs[0].token_ids) for x in o); tps=gt/dt
        best=tps if best is None or tps>best else best
    print("BENCH",json.dumps({"model":MODEL,"eager":EAGER,"batch":B,"out":OUT,"decode_tok_s":round(best,1)}))
else:
    o=llm.generate(prompts,sp,use_tqdm=False)
    print("DECODE done, tokens:",sum(len(x.outputs[0].token_ids) for x in o))
