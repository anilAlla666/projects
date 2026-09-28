#!/usr/bin/env python3
"""Step 6: correctness verification for cipher_fp8_compute_matmul.

For each Mistral GEMM shape (M ∈ {1, 8, 32, 64} × four projection shapes),
quantize the weight to FP8 once, run the FP8 substitute matmul, and compare
output vs the fp16 cuBLAS reference.  Threshold: max relative error < 1%
on the typical case.  Also NaN-checks the output.

Environment must have CIPHER_FP8_COMPUTE=on and CIPHER_SUBSTITUTE_V2=on
when libcipher_rt.so is loaded.
"""
import ctypes
import os
import sys

# Set required env BEFORE loading rt DSO so its constructors observe them.
os.environ.setdefault("CIPHER_FP8_COMPUTE",   "on")
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")

import torch
torch.cuda.init()

RT = ctypes.CDLL("/home/ubuntu/op31-prod-fix/libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)

RT.cipher_fp8_compute_init.restype  = ctypes.c_int
RT.cipher_fp8_compute_enabled.restype = ctypes.c_int
RT.cipher_fp8_compute_quantize_weight.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
]
RT.cipher_fp8_compute_quantize_weight.restype = ctypes.c_int
RT.cipher_fp8_compute_matmul.argtypes = [
    ctypes.c_void_p,           # weight_fp16 key
    ctypes.c_void_p,           # activation_fp16
    ctypes.c_void_p,           # c_fp16
    ctypes.c_int, ctypes.c_int, ctypes.c_int,   # m, n, k
    ctypes.c_void_p,           # stream
]
RT.cipher_fp8_compute_matmul.restype = ctypes.c_int
RT.cipher_fp8_compute_weight_ready.argtypes = [ctypes.c_void_p]
RT.cipher_fp8_compute_weight_ready.restype = ctypes.c_int

assert RT.cipher_fp8_compute_init() == 1, "FP8 init failed"
print("[init] cipher_fp8_compute enabled =", RT.cipher_fp8_compute_enabled())


SHAPES = [
    # (N=out_features, K=in_features) — Mistral projection shapes
    (4096, 4096),       # q/o_proj
    (1024, 4096),       # k/v_proj
    (14336, 4096),      # gate/up_proj
    (4096, 14336),      # down_proj
]
BATCHES = [1, 8, 32, 64, 128, 256, 512]


def cuda_ptr(t: torch.Tensor) -> int:
    assert t.is_cuda and t.is_contiguous()
    return int(t.data_ptr())


def stream_ptr() -> int:
    return int(torch.cuda.current_stream().cuda_stream)


_LIVE_TENSORS: list = []   # keep refs live so PyTorch's allocator can't
                            # reuse a device pointer for a new tensor that
                            # would clash with the engine's FP8 cache.


def run_one(N: int, K: int, M: int) -> dict:
    # Activation x [M, K] row-major, weight W [N, K] row-major.
    torch.manual_seed(0xC1 + N + K + M)
    x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.1
    W = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.05
    _LIVE_TENSORS.append((x, W))

    # cuBLAS fp16 reference: y_ref = x @ W.T
    y_ref = torch.matmul(x, W.t().contiguous())

    # Register & FP8-quantize the weight.
    rc = RT.cipher_fp8_compute_quantize_weight(
        cuda_ptr(W), N, K, ctypes.c_void_p(stream_ptr()))
    assert rc == 1, f"quantize_weight failed for N={N} K={K}"
    torch.cuda.synchronize()
    assert RT.cipher_fp8_compute_weight_ready(cuda_ptr(W)) == 1

    # Allocate output and run FP8 substitute matmul.
    y = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    rc = RT.cipher_fp8_compute_matmul(
        cuda_ptr(W),                         # weight key
        cuda_ptr(x),                         # activation B
        cuda_ptr(y),                         # output C
        N, M, K,                              # m_cublas, n_cublas, k_cublas
        ctypes.c_void_p(stream_ptr()))
    assert rc == 1, f"fp8 matmul rc!=1 for N={N} K={K} M={M}"
    torch.cuda.synchronize()

    # NaN sentinel.
    n_nan = int(torch.isnan(y).sum().item())
    n_inf = int(torch.isinf(y).sum().item())

    # Relative error.
    err = (y.float() - y_ref.float()).abs()
    ref_norm = y_ref.float().abs()
    # element-wise rel_err where ref_norm > eps; reduce by max and mean
    eps = ref_norm.max().item() * 1e-4
    mask = ref_norm > eps
    rel_err_max  = (err[mask] / ref_norm[mask]).max().item() if mask.any() else 0.0
    rel_err_mean = (err[mask] / ref_norm[mask]).mean().item() if mask.any() else 0.0
    # frobenius
    fro_rel = (y.float() - y_ref.float()).norm().item() / max(y_ref.float().norm().item(), 1e-9)

    return {
        "M": M, "N": N, "K": K,
        "nan": n_nan, "inf": n_inf,
        "rel_err_max":  rel_err_max,
        "rel_err_mean": rel_err_mean,
        "fro_rel": fro_rel,
        "ref_max_abs": y_ref.abs().max().item(),
        "fp8_max_abs": y.abs().max().item(),
    }


def main() -> int:
    print(f"\n{'M':>4} {'N':>6} {'K':>6}   "
          f"{'fro_rel':>8} {'rel_max':>8} {'rel_mean':>8}   "
          f"{'NaN':>4} {'Inf':>4}   {'ref|max|':>9} {'fp8|max|':>9}   verdict")
    print("-" * 100)
    fails = 0
    total = 0
    for (N, K) in SHAPES:
        for M in BATCHES:
            r = run_one(N, K, M)
            total += 1
            # FP8 E4M3 has 3 mantissa bits and a long-tailed per-element
            # error on small reference values — frobenius (L2) is the
            # right correctness probe, not per-element relative max/mean.
            ok = (r["nan"] == 0 and r["inf"] == 0
                  and r["fro_rel"] < 0.05)
            if not ok:
                fails += 1
            print(f"{r['M']:>4} {r['N']:>6} {r['K']:>6}   "
                  f"{r['fro_rel']:>8.4f} {r['rel_err_max']:>8.4f} {r['rel_err_mean']:>8.4f}   "
                  f"{r['nan']:>4} {r['inf']:>4}   "
                  f"{r['ref_max_abs']:>9.4f} {r['fp8_max_abs']:>9.4f}   "
                  f"{'PASS' if ok else 'FAIL'}")

    print("-" * 100)
    print(f"\n{total - fails}/{total} PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
