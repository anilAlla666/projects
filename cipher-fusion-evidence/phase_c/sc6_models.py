#!/usr/bin/env python3
"""Track 2 SC6 — shared harness constants (model registry + helpers).

SC6 runs the N=4 weight-sharing verification on two models:
  - TinyLlama-1.1B  — the small-model edge; weights (~2.06 GiB) are a small
                      fraction of a CUDA context, so it UNDERSTATES the
                      substrate-value % (kept for SC3/SC4/SC5 continuity).
  - Mistral-7B-v0.1 — the production-representative mid-size model; weights
                      (~14 GiB) dominate, so the savings approach the
                      asymptote N*W/(N*W+(N+1)*C).
"""
import subprocess

import torch

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"

# the single canonical prompt — bit-identical is a binary property, one
# prompt proves it (SC6 design memo §7 decision 2).
CANONICAL_PROMPT = ("The history of computing spans several distinct eras, "
                    "each defined by")

MODELS = {
    "TinyLlama": {"path": "/home/ubuntu/models/TinyLlama-1.1B",
                  "dtype": torch.float16},
    "Mistral-7B": {"path": "/home/ubuntu/models/Mistral-7B-v0.1",
                   "dtype": torch.bfloat16},
}

DT = {torch.float16: "float16", torch.bfloat16: "bfloat16",
      torch.float32: "float32"}

N_CONSUMERS = 4          # the N in "N=4 verification"


def gpu_fb_mib():
    """Whole-GPU framebuffer in use (nvidia-smi memory.used)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().split("\n")[0].strip())
