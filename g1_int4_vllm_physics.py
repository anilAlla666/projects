#!/usr/bin/env python3
# PHYSICS kill-shot via vLLM (GPTQ-Marlin, the proven fast int4 path on this box; no new installs).
# Question: is int4 batch-1 decode bandwidth-floored (~4x fp16's 8.2ms/tok)? Or dequant-bound (STOP signal)?
# Method: per-token decode latency isolated by differencing two greedy single-seq bursts:
#   per_tok = (t_long - t_short)/(N_long - N_short)   -> cancels prefill + fixed overhead.
# vLLM uses cuda-graph decode (the engine's target regime), so this is the achievable int4 graph-decode floor.
import os, sys, time, json
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
os.environ["VLLM_USE_DEEP_GEMM"]="0"; os.environ["VLLM_PLUGINS"]=""
from vllm import LLM, SamplingParams

path = sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/Llama-3.1-8B-Instruct-w4a16"
quant = sys.argv[2] if len(sys.argv)>2 else "gptq_marlin"   # pass "none" for fp16 baseline
qkw = {} if quant=="none" else {"quantization": quant}
llm = LLM(model=path, dtype="float16", gpu_memory_utilization=0.6, max_model_len=2048,
          enforce_eager=False, disable_log_stats=True, **qkw)
prompt = "The history of artificial intelligence began in the 1950s, when researchers first"

def burst(n, reps=3):
    sp = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)
    llm.generate([prompt], sp)  # warmup / graph capture
    best = 1e9
    for _ in range(reps):
        t0=time.time(); o=llm.generate([prompt], sp); dt=time.time()-t0
        best=min(best,dt); toks=len(o[0].outputs[0].token_ids)
    return best, toks

t16,_  = burst(16)
t128,n = burst(128)
per_tok_ms = (t128 - t16)/(128-16)*1000.0
import glob as _g
wbytes = sum(os.path.getsize(f) for f in _g.glob(os.path.join(path,"*.safetensors")))
floor_ms = wbytes/2.5e12*1000.0
fp16 = 8.2
res = {"model":path,"t16_ms":round(t16*1000,1),"t128_ms":round(t128*1000,1),
       "int4_decode_per_tok_ms":round(per_tok_ms,2),"int4_weight_floor_ms":round(floor_ms,2),
       "fp16_per_tok_ms":fp16,"int4_vs_fp16_speedup":round(fp16/per_tok_ms,2),"engine":f"vllm-{quant}"}
print("\n=== INT4 vLLM GPTQ-Marlin batch-1 decode physics ===",flush=True)
print(f"  t16={res['t16_ms']:.0f}ms  t128={res['t128_ms']:.0f}ms",flush=True)
print(f"  int4 decode per-tok = {res['int4_decode_per_tok_ms']}ms  (differenced, prefill-cancelled)",flush=True)
print(f"  int4 weight-bandwidth floor ~{floor_ms:.2f}ms/tok | fp16 measured {fp16}ms/tok",flush=True)
print(f"  int4 vs fp16 per-tok speedup = {res['int4_vs_fp16_speedup']}x  "
      f"({'BANDWIDTH-FLOORED (physics CONFIRMED)' if res['int4_vs_fp16_speedup']>=2.5 else 'DEQUANT-BOUND? sub-2.5x -> scrutinize STOP signal'})",flush=True)
outf=f"/home/ubuntu/g1_vllm_physics_{quant}.json"
json.dump(res, open(outf,"w"), indent=2)
print(f"  [written] {outf}",flush=True)
sys.stdout.flush(); os._exit(0)
