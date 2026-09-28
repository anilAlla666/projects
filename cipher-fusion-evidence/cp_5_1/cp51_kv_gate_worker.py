#!/usr/bin/env python3
"""CP 5.1 Step 3 — correctness-gate worker (one vLLM run, one process).

Runs a deterministic greedy decode and emits the raw output token IDs as a
single JSON line tagged ``CIPHER_GATE_RESULT``. The substrate state is decided
entirely by the ``CIPHER_KV_ALLOC`` env var, which the cipher-vllm-kv plugin's
register() reads — so substrate-on vs substrate-off is a clean per-process
switch with byte-identical Python otherwise.

The gate driver (cp51_kv_gate.py) runs this twice and diffs the token IDs.
Token IDs, not decoded text: a tokenizer roundtrip can mask a 1-token divergence.
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

MODE = "ON" if os.environ.get("CIPHER_KV_ALLOC", "1") not in ("0", "off", "no", "") else "OFF"
MODEL = os.environ.get("CIPHER_GATE_MODEL", "/home/ubuntu/models/TinyLlama-1.1B")

print(f"[gate-worker] vllm {vllm.__version__}  substrate={MODE}  model={MODEL}", flush=True)

t0 = time.time()
llm = LLM(
    model=MODEL,
    enforce_eager=True,          # removes the cudagraph confound for the gate
    gpu_memory_utilization=0.30,
    max_model_len=2048,
)
print(f"[gate-worker] engine-init OK in {time.time() - t0:.1f}s", flush=True)

# Greedy, fixed seed, fixed max_tokens — fully deterministic decode.
sp = SamplingParams(temperature=0.0, max_tokens=64, seed=1234)
prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Write a short poem about the sea:",
    "In a distant galaxy, a lone explorer discovered",
]
out = llm.generate(prompts, sp)

result = {"mode": MODE, "model": MODEL, "prompts": []}
for o in out:
    result["prompts"].append({
        "prompt": o.prompt,
        "token_ids": list(o.outputs[0].token_ids),
        "text": o.outputs[0].text,
    })

print("CIPHER_GATE_RESULT " + json.dumps(result), flush=True)
print("[gate-worker] DONE", flush=True)
