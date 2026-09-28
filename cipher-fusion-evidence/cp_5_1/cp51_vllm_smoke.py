#!/usr/bin/env python3
"""CP 5.1 step 1 — vLLM engine-init + generate smoke test.

Discriminates: does the installed vLLM engine-init cleanly on this pod
(torch 2.11 / CUDA 13.0) after the ml_dtypes NumPy-2 ABI repair?
No CIPHER substrate loaded here — this is the unmodified-vLLM baseline check.
"""
import os, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# deep_gemm (FP8 JIT GEMM backend) is not installed on this pod; vLLM 0.20.2
# gates its warmup on GPU capability, not package availability, so it hard-
# raises. Our bf16/fp16 models never use FP8 kernels — disable it cleanly.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import vllm
from vllm import LLM, SamplingParams

print(f"vllm {vllm.__version__}", flush=True)

t0 = time.time()
llm = LLM(
    model="/home/ubuntu/models/TinyLlama-1.1B",
    enforce_eager=True,
    gpu_memory_utilization=0.30,
    max_model_len=2048,
)
print(f"engine-init OK in {time.time() - t0:.1f}s", flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=24, seed=1234)
prompts = ["The capital of France is", "2 + 2 ="]
out = llm.generate(prompts, sp)
for o in out:
    print(f"PROMPT={o.prompt!r}  ->  {o.outputs[0].text!r}", flush=True)

print("SMOKE OK", flush=True)
