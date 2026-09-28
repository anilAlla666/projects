#!/usr/bin/env python3
# CP 2.5 gate (d) — Fix A smoke: Marlin in the primary context, injection-only.
# A Marlin-ELIGIBLE GEMM (M<=64 -> not passthrough) with CIPHER_MARLIN=on,
# CUDA_INJECTION64_PATH only. Fix A pins the quant/repack sequence to the
# device-0 primary context; the prior bug hung here (testA_fixA_hung.log).
# PASS = Marlin handles the GEMM and the process exits cleanly (no hang).
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), file=sys.stderr)
a = torch.randn(32, 4096, dtype=torch.float16, device="cuda")   # M=32 <= 64
w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(20):
    c = (a @ w)
torch.cuda.synchronize()
print("marlin smoke done, c.sum=", float(c.float().sum()), file=sys.stderr)
