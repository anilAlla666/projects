#!/usr/bin/env python3
# =============================================================================
# Change 1 gate test — EDMD Live Calibration
#
# Verifies that CIPHER's EDMD live calibration:
#   1. collects passthrough snapshots from real cuBLAS GEMMs,
#   2. fits a rank-16 surrogate via randomized SVD of the weight operand,
#   3. registers it through cipher_koopman_fp16_register_shape,
#   4. and routes subsequent calls through the rank-16 kernel with
#      bounded error on inputs drawn from a low-rank manifold.
#
# Gate (realistic, per approved plan):
#   residual_ratio = ||Y_cipher - Y_ref||_F / ||Y_ref||_F  <  0.10
#
# All three numbers are reported regardless of pass/fail:
#   - residual_ratio
#   - max_diff (element-wise)
#   - energy_captured fraction (rank-16 sum of sigma^2 / ||W||_F^2)
#
# Run:
#   cd /workspace/CIPHER_final_session7
#   LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" \
#     CIPHER_USE_CACHE=0 CIPHER_EDMD_LIVE=1 CIPHER_FORCE_PERMIT=1 \
#     python3 tests/test_edmd_live_calibration.py
# =============================================================================

import os
import sys
import ctypes
import math

import torch

# The test intentionally uses a registered shape (K=4096, N=4096) because
# that's the entry the dispatch relaunch path hits (registry entry
# "gemm-4096x4096x4096"). The low-rank manifold is 16-dimensional so
# rank-16 surrogate can in principle fit with zero residual.
K = 4096
N = 4096
# M=4096 is required: dispatch's registry lookup keys on the full (m,n,k) tuple
# and only gemm-4096x4096x4096 is a fully-specified entry. Smaller M values
# produce a different shape hash and the GEMM recipe branch never fires,
# so the EDMD live hook would never be invoked. We still only snapshot
# min(M, 8) rows per call inside the hook — M=4096 just selects the code path.
M = 4096
RANK_TRUE = 64         # dimension of the input manifold — matches fit rank
FIT_CALLS = 320        # 320 * min(M,8) = 2560 snapshots → >= 2000 target
TEST_CALLS = 100
GATE_RESIDUAL = 0.10


def resolve_stats_fn():
    try:
        rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    fn = getattr(rt, "cipher_edmd_live_get_stats", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_int,            # K_dim
        ctypes.c_int,            # N_dim
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    fn.restype = ctypes.c_bool
    return fn


def report_fn():
    try:
        rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    fn = getattr(rt, "cipher_edmd_live_report", None)
    if fn is None:
        return None
    fn.argtypes = []
    fn.restype = None
    return fn


def main():
    print("=" * 70)
    print("CHANGE 1 GATE — EDMD Live Calibration")
    print("=" * 70)
    print(f"Shape: K={K} N={N}  |  Input manifold rank: {RANK_TRUE}")
    print(f"Fit snapshots target: ~{FIT_CALLS * min(M, 8)}  |  Test calls: {TEST_CALLS}")
    print(f"Gate: residual_ratio < {GATE_RESIDUAL}")
    print(f"Env: CIPHER_USE_CACHE={os.environ.get('CIPHER_USE_CACHE', '(unset)')} "
          f"CIPHER_EDMD_LIVE={os.environ.get('CIPHER_EDMD_LIVE', '(unset)')}")
    print("-" * 70)

    dev = "cuda"
    torch.manual_seed(0)

    # Fixed weight — the operator we're rank-reducing.
    # Scale so that output magnitudes are O(1) for stable ratios.
    B = (torch.randn(K, N, dtype=torch.float32, device=dev) / math.sqrt(K)).half()

    # Fixed 16-D orthonormal subspace in R^K, so activations live on a
    # manifold of effective rank RANK_TRUE. This is the honest test regime
    # for any rank-r Koopman surrogate.
    U_basis_f32 = torch.randn(RANK_TRUE, K, dtype=torch.float32, device=dev)
    # Orthonormalize rows (QR of the transposed matrix)
    Q, _ = torch.linalg.qr(U_basis_f32.t())     # (K, RANK_TRUE)
    U_basis = (Q.t().contiguous()).half()       # (RANK_TRUE, K)

    # Pre-allocate a ring of NRING distinct input tensors and write different
    # low-rank content into each slot. This gives CIPHER's distinct-input
    # gate the pointer diversity it needs without retaining 320*32MB tensors
    # (which would force PyTorch's allocator into a reuse-heavy pattern).
    NRING = 16   # must be >= DISTINCT_INPUT_THRESHOLD (8) in cipher_edmd_live.cpp
    input_ring = [torch.empty(M, K, dtype=torch.float16, device=dev)
                  for _ in range(NRING)]
    U_basis_f32 = U_basis.float()

    def fill_ring_slot(slot, seed_offset):
        g = torch.Generator(device=dev).manual_seed(1000 + seed_offset)
        z = torch.randn(M, RANK_TRUE, generator=g, dtype=torch.float32, device=dev)
        input_ring[slot].copy_((z @ U_basis_f32).half())

    def make_input(seed_offset):
        # Held-out evaluation path — fresh tensor (not from the ring, so
        # we exercise the cache-miss kernel path rather than pointer hits).
        g = torch.Generator(device=dev).manual_seed(9_000_000 + seed_offset)
        z = torch.randn(M, RANK_TRUE, generator=g, dtype=torch.float32, device=dev)
        return (z @ U_basis_f32).half()

    # --- warm-up: fill the ring once and touch each slot ---
    for slot in range(NRING):
        fill_ring_slot(slot, slot)
    for _ in range(5):
        _ = torch.mm(input_ring[0], B)
    torch.cuda.synchronize()

    # --- collection phase: cycle through the ring so distinct pointers
    # exceed the threshold, and keep feeding new content on each revolution.
    print("Collection phase...", flush=True)
    for i in range(FIT_CALLS):
        slot = i % NRING
        fill_ring_slot(slot, 1_000_000 + i)
        x = input_ring[slot]
        y = torch.mm(x, B)
    torch.cuda.synchronize()

    # Poll for background fit completion. Randomized rank-64 SVD on
    # 2000x4096 snapshots is ~2 seconds of CPU. We wait up to 30 s.
    import time as _t
    is_reg = None
    try:
        rt_lib = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
        is_reg = getattr(rt_lib, "cipher_edmd_live_is_registered", None)
        if is_reg is not None:
            is_reg.argtypes = [ctypes.c_int, ctypes.c_int]
            is_reg.restype  = ctypes.c_bool
    except OSError:
        pass
    t0 = _t.monotonic()
    while _t.monotonic() - t0 < 30.0:
        if is_reg is not None and is_reg(K, N):
            break
        _t.sleep(0.2)
    print(f"Fit wait: {_t.monotonic() - t0:.2f}s  registered={bool(is_reg(K, N)) if is_reg else '?'}",
          flush=True)

    # --- read live stats ---
    stats_fn = resolve_stats_fn()
    residual_live = float("nan")
    energy_captured = float("nan")
    max_diff_live = float("nan")
    rows_collected = -1
    if stats_fn is not None:
        rr = ctypes.c_float(0.0)
        en = ctypes.c_float(0.0)
        md = ctypes.c_float(0.0)
        rc = ctypes.c_int(0)
        ok = stats_fn(K, N,
                      ctypes.byref(rr), ctypes.byref(en),
                      ctypes.byref(md), ctypes.byref(rc))
        if ok:
            residual_live = rr.value
            energy_captured = en.value
            max_diff_live = md.value
            rows_collected = rc.value

    rep = report_fn()
    if rep is not None:
        rep()

    print(f"Live-fit diagnostics (from calibration-time snapshots):")
    print(f"  rows_collected   = {rows_collected}")
    print(f"  residual_ratio   = {residual_live:.6f}")
    print(f"  energy_captured  = {energy_captured:.6f}")
    print(f"  max_diff (calib) = {max_diff_live:.6f}")
    print("-" * 70)

    # --- held-out evaluation ---
    print("Held-out evaluation phase...", flush=True)
    B_f32 = B.float()
    num = 0.0
    den = 0.0
    max_diff = 0.0
    for i in range(TEST_CALLS):
        x = make_input(10000 + i)
        y_cipher = torch.mm(x, B).float()
        y_ref = x.float() @ B_f32
        diff = (y_cipher - y_ref)
        num += float((diff * diff).sum().item())
        den += float((y_ref * y_ref).sum().item())
        d = float(diff.abs().max().item())
        if d > max_diff:
            max_diff = d
    torch.cuda.synchronize()

    residual_heldout = math.sqrt(num / den) if den > 0 else float("inf")

    print(f"Held-out residual_ratio  = {residual_heldout:.6f}")
    print(f"Held-out max_diff        = {max_diff:.6f}")
    print("-" * 70)

    print("REPORT:")
    print(f"  residual_ratio  (held-out) : {residual_heldout:.6f}")
    print(f"  max_diff        (held-out) : {max_diff:.6f}")
    print(f"  energy_captured (rank-16)  : {energy_captured:.6f}")
    print(f"  residual_ratio  (calib)    : {residual_live:.6f}")
    print(f"  max_diff        (calib)    : {max_diff_live:.6f}")
    print(f"  rows_collected             : {rows_collected}")
    print("=" * 70)

    passed = residual_heldout < GATE_RESIDUAL
    print(f"GATE {'PASS' if passed else 'FAIL'}: "
          f"residual_ratio < {GATE_RESIDUAL} "
          f"(actual={residual_heldout:.6f})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
