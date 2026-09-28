#!/usr/bin/env python3
"""Step-2 megakernel correctness test.
Verifies cipher_weight_compress_int4_silu_mul and cipher_weight_compress_int4_down_residual
match the existing two-kernel reference (gate INT4 + up INT4 + silu·mul vs gate+up
done separately and combined via cipher_fused_silu_mul).
"""
import os, sys, ctypes, random
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

import torch
torch.manual_seed(7); random.seed(7)

for fn, restype, argtypes in [
    ("cipher_substitute_v2_init",        ctypes.c_int, []),
    ("cipher_weight_compress_init",      ctypes.c_int, []),
    ("cipher_fusion_kernels_init",       ctypes.c_int, []),
    ("cipher_weight_compress_observe",   ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t]),
    ("cipher_weight_compress_quantize",  ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    ("cipher_weight_compress_lookup_T",  ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
         ctypes.POINTER(ctypes.c_int)]),
    ("cipher_weight_compress_int4_gemv", ctypes.c_int,
        [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]),
    ("cipher_weight_compress_int4_silu_mul", ctypes.c_int,
        [ctypes.c_void_p]*6 + [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p]),
    ("cipher_weight_compress_int4_down_residual", ctypes.c_int,
        [ctypes.c_void_p]*5 + [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p]),
    ("cipher_fused_silu_mul", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int, ctypes.c_void_p]),
]:
    f = getattr(rt, fn)
    if restype is not None: f.restype = restype
    f.argtypes = argtypes

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()
rt.cipher_fusion_kernels_init()


def quant_lookup(w_T):
    """Quantize a (K,N) fp16 weight, return (B_T, scales) device pointers."""
    for _ in range(1001):
        rt.cipher_weight_compress_observe(w_T.data_ptr(), w_T.numel()*2)
    K, N = w_T.shape
    rc = rt.cipher_weight_compress_quantize(w_T.data_ptr(), K, N)
    assert rc == 1, "quantize failed"
    bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
    br = ctypes.c_int(); bc = ctypes.c_int()
    rc = rt.cipher_weight_compress_lookup_T(w_T.data_ptr(),
        ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
    assert rc == 1
    return bT.value, bs.value


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def test_silu_mul_megakernel():
    M = 1
    K = 4096
    N = 14336   # gate/up output dim for Mistral
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    Wg = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    Wu = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    bgT, bgs = quant_lookup(Wg)
    buT, bus = quant_lookup(Wu)

    # Reference: two separate INT4 GEMVs + cipher_fused_silu_mul
    gate = torch.empty(M, N, dtype=torch.float16, device="cuda")
    up   = torch.empty(M, N, dtype=torch.float16, device="cuda")
    out  = torch.empty(M, N, dtype=torch.float16, device="cuda")
    s = stream_ptr()
    rc = rt.cipher_weight_compress_int4_gemv(x.data_ptr(), bgT, bgs,
        gate.data_ptr(), M, N, K, s); assert rc == 1
    rc = rt.cipher_weight_compress_int4_gemv(x.data_ptr(), buT, bus,
        up.data_ptr(),   M, N, K, s); assert rc == 1
    rc = rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(),
        out.data_ptr(), gate.numel(), s); assert rc == 1
    torch.cuda.synchronize()

    # Megakernel
    out_meg = torch.empty(M, N, dtype=torch.float16, device="cuda")
    rc = rt.cipher_weight_compress_int4_silu_mul(x.data_ptr(),
        bgT, bgs, buT, bus, out_meg.data_ptr(), M, N, K, s)
    assert rc == 1
    torch.cuda.synchronize()

    # Both go through the same (lossy) INT4 quant of Wg/Wu — should match
    # bit-exact except for fp32 reduction round-off.
    diff = (out - out_meg).float().abs()
    rel  = diff.max().item() / out.float().abs().max().item()
    print(f"  silu_mul megakernel: max_abs_diff={diff.max().item():.4e} "
          f"max_abs={out.float().abs().max().item():.4f} rel={rel:.2e}")
    assert rel < 1e-2, f"silu_mul megakernel diverges: rel={rel}"
    print("  ✓ silu_mul megakernel matches ref (max rel<1e-2)")


def test_down_residual_megakernel():
    M = 1
    K = 14336
    N = 4096
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05
    Wd = torch.randn(K, N, dtype=torch.float16, device="cuda") * 0.05
    res = torch.randn(M, N, dtype=torch.float16, device="cuda") * 0.1
    bdT, bds = quant_lookup(Wd)

    # Reference: INT4 GEMV + add residual
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    s = stream_ptr()
    rc = rt.cipher_weight_compress_int4_gemv(x.data_ptr(), bdT, bds,
        out.data_ptr(), M, N, K, s); assert rc == 1
    out_ref = out + res
    torch.cuda.synchronize()

    # Megakernel
    out_meg = torch.empty(M, N, dtype=torch.float16, device="cuda")
    rc = rt.cipher_weight_compress_int4_down_residual(x.data_ptr(),
        bdT, bds, res.data_ptr(), out_meg.data_ptr(), M, N, K, s)
    assert rc == 1
    torch.cuda.synchronize()

    diff = (out_ref - out_meg).float().abs()
    rel  = diff.max().item() / out_ref.float().abs().max().item()
    print(f"  down_residual megakernel: max_abs_diff={diff.max().item():.4e} "
          f"max_abs={out_ref.float().abs().max().item():.4f} rel={rel:.2e}")
    assert rel < 1e-2, f"down_residual megakernel diverges: rel={rel}"
    print("  ✓ down_residual megakernel matches ref (max rel<1e-2)")


print("Step 2 megakernel correctness:")
test_silu_mul_megakernel()
test_down_residual_megakernel()
print("All megakernel correctness checks PASSED.")
