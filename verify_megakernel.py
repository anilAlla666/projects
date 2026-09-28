#!/usr/bin/env python3
"""Verify the Step-2 megakernels write correct outputs (not silently 0/NaN).

For each of cipher_int4_silu_mul_gemv and cipher_int4_down_residual_gemv:
  - Allocate output as NaN sentinel
  - Call the kernel
  - Check no NaNs and rel-err vs reference is in INT4 noise range
"""
import os, ctypes
import warnings; warnings.filterwarnings("ignore")
ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

for fn, restype, argtypes in [
    ("cipher_substitute_v2_init", ctypes.c_int, []),
    ("cipher_weight_compress_init", ctypes.c_int, []),
    ("cipher_weight_compress_observe", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t]),
    ("cipher_weight_compress_quantize", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("cipher_weight_compress_lookup_T", ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
         ctypes.POINTER(ctypes.c_int)]),
    ("cipher_weight_compress_int4_silu_mul", ctypes.c_int,
        [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_void_p]),
    ("cipher_weight_compress_int4_down_residual", ctypes.c_int,
        [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.c_void_p]),
]:
    f = getattr(rt, fn)
    if restype is not None: f.restype = restype
    f.argtypes = argtypes

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()

import torch
torch.manual_seed(7)


def quant_lookup(W):
    K, N = W.shape
    for _ in range(1001):
        rt.cipher_weight_compress_observe(W.data_ptr(), W.numel()*2)
    rc = rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)
    assert rc == 1
    bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
    br = ctypes.c_int(); bc = ctypes.c_int()
    rc = rt.cipher_weight_compress_lookup_T(W.data_ptr(),
        ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
    assert rc == 1
    return bT.value, bs.value


sp = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

# silu_mul megakernel: M=1, K=4096, N=14336 (Mistral MLP shape)
M, K, N = 1, 4096, 14336
x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
Wg = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
Wu = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
bgT, bgs = quant_lookup(Wg)
buT, bus = quant_lookup(Wu)

# fp16 reference
gate_ref = (x @ Wg).float()
up_ref   = (x @ Wu).float()
silu_ref = (gate_ref * torch.sigmoid(gate_ref)) * up_ref

out = torch.full((M, N), float("nan"), dtype=torch.float16, device="cuda")
rc = rt.cipher_weight_compress_int4_silu_mul(
    x.data_ptr(), bgT, bgs, buT, bus, out.data_ptr(),
    M, N, K, sp)
torch.cuda.synchronize()
nan = torch.isnan(out).sum().item()
print(f"silu_mul: rc={rc} NaNs={nan}/{M*N}")
assert nan == 0 and rc == 1
diff = (out.float() - silu_ref).abs()
rel  = diff.mean().item() / silu_ref.abs().mean().item()
print(f"  vs fp16 ref: mean_abs_diff={diff.mean().item():.4f} "
      f"rel_err={rel:.3f}  (INT4 noise expected ~0.12)")
assert 0.05 < rel < 0.30, "out of expected INT4 noise band"
print("  ✓ silu_mul megakernel writes a real INT4 result")

# down_residual: M=1, K=14336, N=4096
M2, K2, N2 = 1, 14336, 4096
x2 = torch.randn(M2, K2, dtype=torch.float16, device="cuda") * 0.05
Wd = torch.randn(K2, N2, dtype=torch.float16, device="cuda") * 0.05
res = torch.randn(M2, N2, dtype=torch.float16, device="cuda") * 0.1
bdT, bds = quant_lookup(Wd)

down_ref = (x2 @ Wd).float()
out_ref = down_ref + res.float()

out2 = torch.full((M2, N2), float("nan"), dtype=torch.float16, device="cuda")
rc = rt.cipher_weight_compress_int4_down_residual(
    x2.data_ptr(), bdT, bds, res.data_ptr(), out2.data_ptr(),
    M2, N2, K2, sp)
torch.cuda.synchronize()
nan = torch.isnan(out2).sum().item()
print(f"down_residual: rc={rc} NaNs={nan}/{M2*N2}")
assert nan == 0 and rc == 1
diff = (out2.float() - out_ref).abs()
rel  = diff.mean().item() / out_ref.abs().mean().item()
print(f"  vs fp16 ref: mean_abs_diff={diff.mean().item():.4f} "
      f"rel_err={rel:.3f}  (residual lifts denominator so noise <<0.12)")
assert rel < 0.30, "out of expected band"
print("  ✓ down_residual megakernel writes a real INT4 result")
print("\nBoth megakernels verified: rc=1, no NaNs, INT4-noise-level rel-err.")
