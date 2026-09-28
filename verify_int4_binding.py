#!/usr/bin/env python3
"""Definitively check the cipher_weight_compress_int4_gemv binding.

Tests:
  A. "Broken" binding: argtypes 4×ptr + 2×int + 1×ptr (matches existing repo).
     Call passes (A, B_T, B_scales, C, N, K, stream).
  B. "Fixed" binding: argtypes 4×ptr + 3×int + 1×ptr.
     Call passes (A, B_T, B_scales, C, M=1, N, K, stream).

For each: rc, output mean abs, output mean abs vs fp16 reference.

If my Step-5 claim is right:
  - Binding A: rc=0, output stays at initial-empty values (garbage)
  - Binding B: rc=1, output close to fp16 reference (within INT4 quant noise ~12%)
"""
import os, sys, ctypes
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

# Init
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()

rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype  = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype  = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype  = ctypes.c_int

# Two distinct ctypes function objects (so each can have different argtypes)
gemv_broken = rt.cipher_weight_compress_int4_gemv
gemv_broken.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int,
                                                ctypes.c_void_p]
gemv_broken.restype = ctypes.c_int

# Re-resolve a separate binding for fixed
gemv_fixed = ctypes.CFUNCTYPE(ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
sym = ctypes.cast(rt.cipher_weight_compress_int4_gemv, ctypes.c_void_p)
gemv_fixed_fn = gemv_fixed(sym.value)

import torch
torch.manual_seed(0)

# Mistral q_proj-like shape: x[1, 4096] @ W[4096, 4096] -> [1, 4096]
M, K, N = 1, 4096, 4096
x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
W = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05

# Quantize W
for _ in range(1001):
    rt.cipher_weight_compress_observe(W.data_ptr(), W.numel()*2)
rc = rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)
assert rc == 1, "quantize failed"

bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
br = ctypes.c_int(); bc = ctypes.c_int()
rc = rt.cipher_weight_compress_lookup_T(W.data_ptr(),
    ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
assert rc == 1
print(f"weight quantized: rows={br.value} cols={bc.value} bT_ptr={hex(bT.value)}")

# Reference fp16 matmul
ref = (x @ W).float()
print(f"ref:    mean_abs={ref.abs().mean().item():.6f}  max_abs={ref.abs().max().item():.4f}")

stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

# Sentinel value to detect untouched memory
sentinel = float("nan")

# A: Broken binding — what existing repo uses
out_a = torch.full((M, N), sentinel, dtype=torch.float16, device="cuda")
rc_a = gemv_broken(x.data_ptr(), bT.value, bs.value, out_a.data_ptr(),
                    N, K, stream)   # 7 args, no M; existing repo pattern
torch.cuda.synchronize()
nan_count_a = torch.isnan(out_a).sum().item()
print(f"\nBINDING A (broken-as-existing): rc={rc_a}, NaNs={nan_count_a}/{M*N}")
if nan_count_a < M*N:
    finite_a = out_a[~torch.isnan(out_a)].float()
    diff_a = (finite_a - ref.flatten()[:finite_a.numel()]).abs()
    print(f"  finite mean_abs={finite_a.abs().mean().item():.6f}")
    print(f"  diff vs ref:  mean={diff_a.mean().item():.6f}  max={diff_a.max().item():.4f}")
    print(f"  rel error:    mean={(diff_a.mean()/ref.abs().mean()).item():.4f}")
else:
    print(f"  output untouched (all NaN sentinel) — confirms broken path returned without writing")

# B: Fixed binding — pass M=1
out_b = torch.full((M, N), sentinel, dtype=torch.float16, device="cuda")
rc_b = gemv_fixed_fn(x.data_ptr(), bT.value, bs.value, out_b.data_ptr(),
                     M, N, K, stream)   # 8 args
torch.cuda.synchronize()
nan_count_b = torch.isnan(out_b).sum().item()
print(f"\nBINDING B (fixed M,N,K):       rc={rc_b}, NaNs={nan_count_b}/{M*N}")
if nan_count_b < M*N:
    finite_b = out_b[~torch.isnan(out_b)].float()
    diff_b = (finite_b - ref.flatten()[:finite_b.numel()]).abs()
    print(f"  finite mean_abs={finite_b.abs().mean().item():.6f}")
    print(f"  diff vs ref:  mean={diff_b.mean().item():.6f}  max={diff_b.max().item():.4f}")
    print(f"  rel error:    mean={(diff_b.mean()/ref.abs().mean()).item():.4f}")
else:
    print(f"  output untouched")

# Diagnosis
print()
print("="*60)
if rc_a == 0 and nan_count_a == M*N:
    print("DIAGNOSIS: existing 7-arg binding silently returns 0 (no write).")
    print("  All prior 'INT4 GEMV' calls in run_mistral_*.py would have fallen")
    print("  back to fp16 cuBLAS. The repo never measured INT4 in production.")
elif rc_a == 1 and nan_count_a == 0:
    finite_a = out_a[~torch.isnan(out_a)].float()
    rel_a = ((finite_a - ref.flatten()).abs().mean() / ref.abs().mean()).item()
    if rel_a < 0.20:
        print(f"DIAGNOSIS: existing 7-arg binding ALSO works (rel_err={rel_a:.3f}).")
        print("  My Step-5 claim that the binding was broken is WRONG.")
    else:
        print(f"DIAGNOSIS: existing 7-arg binding 'returns rc=1' but produces garbage")
        print(f"  (rel_err={rel_a:.3f} >> INT4 noise floor ~0.12).")
else:
    print(f"DIAGNOSIS: A returned rc={rc_a}, NaN={nan_count_a}/{M*N}.")
    print("  Inconclusive — need more investigation.")
