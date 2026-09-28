#!/usr/bin/env python3
# STEP 1 (gating fact): do 5 in-process vLLM engines (3 big 7-8B + 2 small) COEXIST at full budget? Proven config was
# 2 small eager @ util 0.25. Here: 5 distinct, EAGER first (cheapest, less mem than graph capture), cumulative
# gpu_memory_utilization schedule (vLLM measures free mem at each init -> util must be cumulative). Confirm 5/5 init
# + one greedy generate() each (coherent). NO router yet. If OOM -> lower utils. fp16 weights ~47GiB / 80.
import os, time, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"   # in-process engine (not subprocess)
os.environ["VLLM_USE_DEEP_GEMM"]="0"               # disable FP8 DeepGEMM path (not installed; unrelated to this test)
os.environ["VLLM_PLUGINS"]=""                       # CLEAN vLLM: do NOT auto-load the CIPHER substrate hooks into vLLM
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from vllm import LLM, SamplingParams
# (path, PER-ENGINE gpu_memory_utilization = this engine's OWN footprint fraction = (weights+~2-3GiB KV)/80).
# vLLM 0.20: util is the engine's own target footprint and must be <= CURRENT free; NOT cumulative. Small per-engine
# utils -> each gets ~2-3 GiB KV; total ~71 GiB across 5 (weights 47 + 5 KV + ctx) fits in 79 with margin.
ENG=[("/home/ubuntu/models/TinyLlama-1.1B",        0.07),
     ("/home/ubuntu/models/Llama-3.2-1B-Instruct", 0.08),
     ("/home/ubuntu/models/Mistral-7B-v0.1",       0.24),
     ("/home/ubuntu/models/Qwen2-7B",              0.25),
     ("/home/ubuntu/models/Llama-3.1-8B",          0.26)]
GB=1<<30
def used():
    free,tot=torch.cuda.mem_get_info(); return (tot-free)/GB
EAGER=os.environ.get("EAGER","1")=="1"             # EAGER=0 -> graph-decode (enforce_eager=False)
print(f"=== coexistence probe: {'EAGER' if EAGER else 'GRAPH-DECODE'} ===", flush=True)
sp=SamplingParams(max_tokens=16, temperature=0.0)
engines=[]; names=[]
for path,util in ENG:
    nm=os.path.basename(path)
    t=time.time()
    try:
        llm=LLM(model=path, gpu_memory_utilization=util, max_model_len=2048, enforce_eager=EAGER,
                dtype="float16", disable_log_stats=True)
    except Exception as e:
        print(f"  ENGINE {nm} FAILED to init at util={util}: {repr(e)[:200]}", flush=True)
        print(f"  -> coexistence wall at engine #{len(engines)+1}/{len(ENG)} (used={used():.1f}GiB)", flush=True)
        sys.stdout.flush(); os._exit(1)
    engines.append(llm); names.append(nm)
    print(f"  [{len(engines)}/5] {nm:<26} init {time.time()-t:.0f}s  util={util}  GPU-used={used():.1f}GiB", flush=True)
print(f"=== 5/5 ENGINES COEXIST (eager), GPU-used={used():.1f}GiB / 80 ===", flush=True)
# greedy generate per engine (coherence + determinism)
for nm,llm in zip(names,engines):
    o=llm.generate(["The history of artificial intelligence began in the"], sp, use_tqdm=False)
    txt=o[0].outputs[0].text.replace(chr(10)," ")[:60]
    print(f"  {nm:<26} greedy-> '{txt}'", flush=True)
print("STEP1 PASS: 5 in-process engines coexist + generate (eager). Next: flip graph-decode, then router.", flush=True)
sys.stdout.flush(); os._exit(0)
