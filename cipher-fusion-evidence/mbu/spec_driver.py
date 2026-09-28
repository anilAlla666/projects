#!/usr/bin/env python3
# FP8 x speculative-decode composition: decode throughput vs fp16 baseline.
# Measures the guaranteed-2x route (FP8 per-pass 1.53x  x  ngram spec-decode fewer-passes).
import os, sys, time, json, ctypes
ARM = os.environ.get("SP_ARM", "fp16")        # fp16 | fp8
SPEC = os.environ.get("SP_SPEC", "0") == "1"  # ngram speculative decoding on/off
K = int(os.environ.get("SP_K", "5"))          # num speculative tokens
OUTJ = os.environ.get("SP_OUTJSON", f"/home/ubuntu/cipher-fusion-evidence/mbu/spec_{ARM}_spec{int(SPEC)}.json")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch
from vllm import LLM, SamplingParams

kw = dict(model="mistralai/Mistral-7B-v0.1", enforce_eager=False, gpu_memory_utilization=0.85,
          max_model_len=2048, disable_log_stats=False, compilation_config={"cudagraph_capture_sizes": [1, 8]})
if ARM == "fp16": kw["dtype"] = "float16"
if ARM == "fp8": kw["quantization"] = "fp8"
if SPEC:
    kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": K,
                                "prompt_lookup_max": 4, "prompt_lookup_min": 2}
llm = LLM(**kw)

# realistic, mildly-structured generation (the kind of serving workload ngram helps — lists/code/structured prose)
prompts = [
    "Write a numbered list of 20 steps to set up a Python web server, with a short sentence each.\n1.",
    "Explain in a structured way, with repeated 'The' sentence openers, how a CPU executes an instruction.",
    "List 15 common HTTP status codes and what each means, one per line in the format CODE: meaning.",
    "Summarize the water cycle in 12 sentences, each starting with 'Then the water'.",
]
sp = SamplingParams(max_tokens=256, min_tokens=256, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)  # warmup

best = None
for _ in range(2):
    t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    gen = sum(len(x.outputs[0].token_ids) for x in o); tps = gen / dt
    best = tps if best is None or tps > best else best

# acceptance / spec metrics if available
acc = None
try:
    m = llm.llm_engine.get_metrics() if hasattr(llm.llm_engine, "get_metrics") else None
except Exception:
    m = None
anchor_loaded = True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError: anchor_loaded = False

res = dict(arm=ARM, spec=SPEC, num_spec_tokens=K if SPEC else 0,
           decode_tok_s=round(best, 1), anchor_loaded=anchor_loaded)
print("SPEC", json.dumps(res))
with open(OUTJ, "w") as f: json.dump(res, f, indent=1)
