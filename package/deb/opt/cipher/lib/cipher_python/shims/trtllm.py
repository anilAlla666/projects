"""TensorRT-LLM fusion shim.

TRT-LLM compiles models to optimized engines ahead of time, so the
fusion is largely already baked in by TensorRT itself. The CIPHER
fused RMSNorm path doesn't apply at the same hot-path level. Stub for
now; future v2 work would intercept TRT-LLM's `Engine.execute()`
boundary and apply CIPHER policies to the engine-level batch (rather
than per-kernel).
"""
from __future__ import annotations

import sys


def apply(rt, model=None):
    print(f"[CIPHER PY trtllm] no-op shim (TRT-LLM has built-in fusion; "
          f"CIPHER applies at engine-execute boundary, see v2 roadmap)",
          file=sys.stderr)
    return {"applied": True, "framework": "trtllm",
            "patched_class": None,
            "reason": "trt-llm has its own fusion; cipher is no-op at kernel level"}
