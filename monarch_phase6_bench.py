"""Phase 6 — wall-clock Monarch matvec speedup vs dense matmul on H100.

Tests three Monarch implementations:
  1. einsum (current monarch_matvec) — convenient, may have launch overhead
  2. bmm (batched matrix multiply) — what an NVRTC fused kernel would target
  3. dense matmul (baseline)

Reports per-shape (Mistral linears) at ranks 1, 2, 4, 8, 16:
  - dense_us, mono_einsum_us, mono_bmm_us
  - speedup_bmm vs dense
  - bytes_dense, bytes_mono, bytes_ratio  (HBM read estimates)

Also includes the rotation overhead (Q^T x or Q y, m x m matmul) which is the
mandatory cost of using ProcrustesGPT-rotated Monarch.
"""
import time
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monarch_phase1 import monarch_factorize


def monarch_matvec_bmm(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor,
                       p: int, q_n: int, q_m: int) -> torch.Tensor:
    """Monarch matvec via torch.bmm. A: (p^2, R), B: (R, q_n*q_m), x: (m=p*q_m,)."""
    R = A.shape[1]
    A_brc = A.reshape(p, p, R).permute(2, 0, 1).contiguous()  # (R, p, p)
    B_brc = B.reshape(R, q_n, q_m)                            # (R, q_n, q_m)
    x_resh = x.reshape(p, q_m)                                # (p, q_m)
    # u_r[p, q_n] = sum_l x[p, l] * B[r, q_n, l] over l
    # bmm needs (R, p, q_m) @ (R, q_m, q_n)
    x_batched = x_resh.unsqueeze(0).expand(R, -1, -1).contiguous()      # (R, p, q_m)
    u = torch.bmm(x_batched, B_brc.transpose(1, 2))                     # (R, p, q_n)
    # v_r[p, q_n] = sum_k A[r, p, k] * u[r, k, q_n]
    v = torch.bmm(A_brc, u)                                             # (R, p, q_n)
    y = v.sum(dim=0)                                                    # (p, q_n)
    return y.reshape(p * q_n)


def monarch_matvec_einsum(A, B, x, p, q_n, q_m):
    R = A.shape[1]
    A_brc = A.reshape(p, p, R).permute(2, 0, 1).contiguous()
    B_brc = B.reshape(R, q_n, q_m)
    x_resh = x.reshape(p, q_m)
    u = torch.einsum('pm,rnm->rpn', x_resh, B_brc)
    v = torch.einsum('rik,rkn->rin', A_brc, u)
    return v.sum(dim=0).reshape(p * q_n)


def time_matvec(fn, *args, n_iter=1000, warmup=20):
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # microseconds


def main():
    device = "cuda"
    dtype = torch.float16
    p = 64
    shapes = [
        ("q_proj/o_proj", 4096, 4096),
        ("k_proj/v_proj", 1024, 4096),
        ("gate/up_proj",  14336, 4096),
        ("down_proj",     4096, 14336),
    ]

    print(f"=== Monarch wall-clock benchmark on H100 ===")
    print(f"All shapes use blocking p={p}. Times in microseconds, 1000 iters median.")
    print()
    print(f"  {'shape':<22} {'rank':>5}  {'dense':>8}  {'einsum':>8}  {'bmm':>8}  "
          f"{'speedup':>8}  {'bytes_dense':>12}  {'bytes_mono':>12}  {'bw_ratio':>9}")
    print("  " + "-" * 110)

    rotation_cost_us = {}  # m -> cost of Q^T x for m x m

    for name, n, m in shapes:
        if n % p != 0 or m % p != 0:
            print(f"  {name}: blocking p={p} doesn't divide both — skip")
            continue
        q_n = n // p; q_m = m // p
        W = torch.randn(n, m, device=device, dtype=dtype)
        x = torch.randn(m, device=device, dtype=dtype)

        # Dense baseline
        dense_us = time_matvec(lambda: W @ x)
        bytes_dense = n * m * 2  # fp16

        # Pre-compute Monarch factors (don't time the SVD; that's a one-time cost at calibration)
        W32 = W.to(torch.float32)

        for R in [1, 2, 4, 8, 16]:
            A, B = monarch_factorize(W32, p, q_n, q_m, R)
            A_h = A.to(torch.float16); B_h = B.to(torch.float16)
            ein_us = time_matvec(lambda: monarch_matvec_einsum(A_h, B_h, x, p, q_n, q_m))
            bmm_us = time_matvec(lambda: monarch_matvec_bmm   (A_h, B_h, x, p, q_n, q_m))
            speedup = dense_us / bmm_us
            # Memory: A is (p^2, R), B is (R, q_n*q_m), each fp16
            bytes_mono = (A_h.numel() + B_h.numel()) * 2
            bw_ratio = bytes_dense / bytes_mono
            print(f"  {name:<22} {R:>5}  {dense_us:>8.1f}  {ein_us:>8.1f}  {bmm_us:>8.1f}  "
                  f"{speedup:>7.2f}x  {bytes_dense:>12}  {bytes_mono:>12}  {bw_ratio:>8.1f}x")

        # Rotation overhead (Q^T x) — this is part of every forward call when using rotated Monarch
        # Use the smaller side for rotation.
        rot_dim = min(n, m)
        if rot_dim not in rotation_cost_us:
            Q_full = torch.randn(rot_dim, rot_dim, device=device, dtype=torch.float16)
            x_for_rot = torch.randn(rot_dim, device=device, dtype=torch.float16)
            rotation_cost_us[rot_dim] = time_matvec(lambda: Q_full @ x_for_rot)
        print(f"  {'  rotation (Q^Tx)':<22} {'-':>5}  {rotation_cost_us[rot_dim]:>8.1f}  {'-':>8}  {'-':>8}  "
              f"({'mandatory':>7})  {rot_dim*rot_dim*2:>12}  {'-':>12}  {'-':>9}")
        print()


if __name__ == "__main__":
    main()
