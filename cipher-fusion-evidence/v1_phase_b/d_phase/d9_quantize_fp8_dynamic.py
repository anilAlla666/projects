#!/usr/bin/env python3
# D.9 — produce the PRE-REGISTERED FP8 method as a checkpoint: per-CHANNEL weight + per-TOKEN
# dynamic activation (llm-compressor FP8_DYNAMIC, no calibration needed). vLLM loads it natively
# (compressed-tensors). This is the lossless-grade scheme (arXiv 2411.02355), single-impl for
# both quality and MFU.
import sys
try:
    from llmcompressor import oneshot
except Exception:
    from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
OUT="/home/ubuntu/models/Mistral-7B-FP8-dynamic"
recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=MODEL, recipe=recipe, output_dir=OUT)
print("WROTE", OUT, flush=True)
