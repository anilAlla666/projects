"""T4.5.2 accuracy check: single Linear, compare FP16 vs Marlin outputs.

Run with CIPHER_MARLIN=on / off and compare y values.
Marlin should be within ~12% relative error per the prior INT4 noise floor.
"""
import os, sys, time
import torch
import torch.nn as nn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.set_device(0)

lin = nn.Linear(2048, 2048, bias=False).half().cuda()
x = torch.randn(1, 2048, device='cuda', dtype=torch.float16)

# Warm calls past the stability threshold (count=4) THEN measure
ys = []
for i in range(8):
    with torch.no_grad():
        y = lin(x)
    torch.cuda.synchronize()
    ys.append(y.clone())

# Compare last (Marlin if enabled) to first (FP16 since below threshold)
y0 = ys[0].float()  # forward 1, definitely FP16 (count=1, below threshold=4)
yN = ys[-1].float() # forward 8, Marlin if enabled

# Cosine similarity + relative error
import math
dot = float((y0 * yN).sum())
n0 = float((y0 ** 2).sum() ** 0.5)
nN = float((yN ** 2).sum() ** 0.5)
cos = dot / (n0 * nN + 1e-12)
diff = (y0 - yN)
rel_err = float((diff ** 2).sum() ** 0.5) / (float((y0 ** 2).sum() ** 0.5) + 1e-12)
print(f"y0 (FP16)   first 3: {[round(float(v),4) for v in y0[0,:3]]}")
print(f"yN (Marlin) first 3: {[round(float(v),4) for v in yN[0,:3]]}")
print(f"cosine_sim: {cos:.5f}")
print(f"rel_err:    {rel_err:.5f}")
print(f"PASS"     if cos > 0.95 and rel_err < 0.15 else
      "MARGINAL" if cos > 0.85 and rel_err < 0.30 else
      "FAIL")
