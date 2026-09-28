#!/usr/bin/env python3
"""CP 5.2 Step 3 — offload correctness-gate worker (one vLLM run, one process).

Runs a deterministic greedy decode of several concurrent requests and emits
the output token IDs as a JSON line tagged ``CIPHER_GATE_RESULT``.

Env knobs (the gate driver sets these):
  CIPHER_GATE_MODEL   model path (default TinyLlama-1.1B)
  CIPHER_GATE_BLOCKS  num_gpu_blocks_override — set small to force preemption;
                      unset/0 = let vLLM size the pool (no-preempt baseline)
  CIPHER_KV_ALLOC     CP 5.1 buffer ownership   (default 1)
  CIPHER_KV_OFFLOAD   CP 5.2 snapshot/restore   (default 1)

Prefix caching is disabled — the gate exercises offload, not prefix-cache
interaction (CP 5.2 Step 1 memo: a deliberate gate-scoping choice).
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import vllm
from vllm import LLM, SamplingParams

MODEL = os.environ.get("CIPHER_GATE_MODEL", "/home/ubuntu/models/TinyLlama-1.1B")
BLOCKS = int(os.environ.get("CIPHER_GATE_BLOCKS", "0"))
OFFLOAD = os.environ.get("CIPHER_KV_OFFLOAD", "1") not in ("0", "off", "no", "")

print(f"[gate-worker] vllm {vllm.__version__}  model={MODEL}  "
      f"blocks_override={BLOCKS or 'none'}  offload={'ON' if OFFLOAD else 'OFF'}",
      flush=True)

kw = dict(
    model=MODEL,
    enforce_eager=True,
    gpu_memory_utilization=0.55,
    # 480 so one full sequence fits inside the 32-block num_gpu_blocks_override
    # pool (vLLM reserves a null block → 31 usable × 16 = 496 token slots;
    # 480 clears it for both TinyLlama and Mistral-7B). vLLM refuses to start
    # if the pool can't hold one max_model_len request. Generations here are
    # ~268 tokens, well under 480.
    max_model_len=480,
    enable_prefix_caching=False,   # gate exercises offload, not prefix cache
)
if BLOCKS > 0:
    kw["num_gpu_blocks_override"] = BLOCKS

t0 = time.time()
llm = LLM(**kw)
print(f"[gate-worker] engine-init OK in {time.time() - t0:.1f}s", flush=True)

# Several concurrent requests with long generations — under a small
# num_gpu_blocks_override they contend for the KV pool and force preemption.
sp = SamplingParams(temperature=0.0, max_tokens=256, seed=1234)
prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Write a short poem about the sea:",
    "In a distant galaxy, a lone explorer discovered",
    "The history of computing began when",
    "A recipe for a good day starts with",
]
t1 = time.time()
out = llm.generate(prompts, sp)
print(f"[gate-worker] generate OK in {time.time() - t1:.1f}s", flush=True)

result = {"model": MODEL, "blocks_override": BLOCKS, "offload": OFFLOAD,
          "prompts": []}
for o in out:
    result["prompts"].append({
        "prompt": o.prompt,
        "token_ids": list(o.outputs[0].token_ids),
        "n_tokens": len(o.outputs[0].token_ids),
    })

print("CIPHER_GATE_RESULT " + json.dumps(result), flush=True)
print("[gate-worker] DONE", flush=True)
