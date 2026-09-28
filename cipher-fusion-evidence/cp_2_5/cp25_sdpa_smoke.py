#!/usr/bin/env python3
# CP 2.5 — SDPA-only smoke. Exercises the attn substrate's 3 GOT-patched
# ATen SDPA ::call slots under CUDA_INJECTION64_PATH-only (no LD_PRELOAD).
# Test C (cp25_smoke.py) is matmul-only and leaves tramp_calls=0; this smoke
# closes that gap. PASS = [cipher-attn] exit totals tramp_calls > 0.
import sys, torch
import torch.nn.functional as F
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), file=sys.stderr)
q = torch.randn(2, 8, 256, 64, dtype=torch.float16, device="cuda")
k = torch.randn(2, 8, 256, 64, dtype=torch.float16, device="cuda")
v = torch.randn(2, 8, 256, 64, dtype=torch.float16, device="cuda")
for _ in range(20):
    o = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
print("sdpa done, o.sum=", float(o.float().sum()), file=sys.stderr)
