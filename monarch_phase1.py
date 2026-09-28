"""Phase 1 — Monarch factorization math validation.

A Monarch matrix at rank R approximates a dense W as a sum of R Kronecker-like
terms structured to match the (p, q_n) x (p, q_m) blocking of W:

    W[i*q_n + j, k*q_m + l] ~= sum_r A[i, k, r] * B[r, j, l]

For matvec y = W @ x with x reshaped to (p, q_m), the rank-R Monarch matvec is:
    u_r = x_resh @ B[r].T          # (p, q_n)  — applies right factor B per rank
    v_r = A[..., r] @ u_r          # (p, q_n)  — applies left factor A per rank
    y    = sum_r v_r reshaped       # (n,)

Per-rank FLOPs: p*q_n*q_m (right) + p*p*q_n (left) = p*q_n*(p + q_m)
Dense matmul FLOPs: p*q_n * p*q_m
Reduction at rank=1, p=q_n=q_m: ~p^2 * (1+1) / p^3 = 2/p  (for p=64: 32x fewer ops)

Validation: at rank = min(p^2, q_n*q_m) — full rank — Monarch matches dense bit-perfectly.
At lower ranks, rel_err scales as expected.
"""
import torch


def monarch_factorize(W: torch.Tensor, p: int, q_n: int, q_m: int, rank: int):
    """Factor W [n=p*q_n, m=p*q_m] into Monarch factors A, B at given rank."""
    n, m = W.shape
    assert n == p * q_n, f"n={n} != p*q_n = {p}*{q_n}"
    assert m == p * q_m, f"m={m} != p*q_m = {p}*{q_m}"
    # Reshape: (n, m) -> (p, q_n, p, q_m); permute -> (p, p, q_n, q_m); reshape -> (p^2, q_n*q_m)
    W4 = W.reshape(p, q_n, p, q_m)
    F  = W4.permute(0, 2, 1, 3).reshape(p * p, q_n * q_m)  # row index = (i, k); col index = (j, l)
    # SVD; symmetric absorbed sqrt of singular values into A and B
    U, S, Vh = torch.linalg.svd(F, full_matrices=False)
    R = min(rank, len(S))
    sqrt_S = S[:R].sqrt()
    A = U[:, :R] * sqrt_S.unsqueeze(0)            # (p^2, R)
    B = Vh[:R, :] * sqrt_S.unsqueeze(1)           # (R, q_n*q_m)
    return A, B


def monarch_reconstruct(A: torch.Tensor, B: torch.Tensor,
                        p: int, q_n: int, q_m: int) -> torch.Tensor:
    """Reconstruct full W approximation from Monarch factors."""
    F = A @ B  # (p^2, q_n*q_m)
    W4 = F.reshape(p, p, q_n, q_m).permute(0, 2, 1, 3).reshape(p * q_n, p * q_m)
    return W4


def monarch_matvec(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor,
                   p: int, q_n: int, q_m: int) -> torch.Tensor:
    """y = W_monarch @ x using batched matmuls. x shape (m,) = (p*q_m,)."""
    R = A.shape[1]
    A_brc = A.reshape(p, p, R).permute(2, 0, 1).contiguous()  # (R, p, p)  a_r[i, k]
    B_brc = B.reshape(R, q_n, q_m)                            # (R, q_n, q_m) b_r[j, l]
    x_resh = x.reshape(p, q_m)                                # (p, q_m)
    # Step 1: u_r[p, q_n] = sum_l x_resh[p, l] * B_brc[r, q_n, l]  -- one bmm across R ranks
    u = torch.einsum('pm,rnm->rpn', x_resh, B_brc)            # (R, p, q_n)
    # Step 2: v_r[i, j] = sum_k A_brc[r, i, k] * u_r[k, j]
    v = torch.einsum('rik,rkn->rin', A_brc, u)                # (R, p, q_n)
    y = v.sum(dim=0)                                          # (p, q_n)
    return y.reshape(p * q_n)


def monarch_matvec_naive(A, B, x, p, q_n, q_m):
    """Reference implementation (rank-by-rank loop) for cross-checking the bmm/einsum path."""
    R = A.shape[1]
    A_brc = A.reshape(p, p, R)                                # a_r[i, k]
    B_brc = B.reshape(R, q_n, q_m)                            # b_r[j, l]
    x_resh = x.reshape(p, q_m)
    y = torch.zeros(p, q_n, device=x.device, dtype=x.dtype)
    for r in range(R):
        a_r = A_brc[:, :, r]
        b_r = B_brc[r]
        u_r = x_resh @ b_r.T
        v_r = a_r @ u_r
        y = y + v_r
    return y.reshape(p * q_n)


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float64  # use fp64 so the validation isn't masked by SVD numerics

    # ---------- Test 1: full-rank reconstruction ----------
    print("Test 1: full-rank reconstruction must equal W exactly (fp64)")
    print("-" * 70)
    for n, m, p in [(4096, 4096, 64), (4096, 4096, 32), (4096, 4096, 128),
                     (1024, 4096, 64), (4096, 14336, 64), (14336, 4096, 64)]:
        q_n = n // p
        q_m = m // p
        if p * q_n != n or p * q_m != m:
            continue
        W = torch.randn(n, m, device=device, dtype=dtype)
        full_rank = min(p * p, q_n * q_m)
        A, B = monarch_factorize(W, p, q_n, q_m, full_rank)
        W_rec = monarch_reconstruct(A, B, p, q_n, q_m)
        rec_err = (W - W_rec).norm() / W.norm()

        x = torch.randn(m, device=device, dtype=dtype)
        y_dense = W @ x
        y_mono = monarch_matvec(A, B, x, p, q_n, q_m)
        matvec_err = (y_dense - y_mono).norm() / y_dense.norm()

        # Cross-check naive vs einsum
        y_naive = monarch_matvec_naive(A, B, x, p, q_n, q_m)
        cross_err = (y_naive - y_mono).norm() / y_dense.norm()

        ok = "PASS" if (rec_err < 1e-5 and matvec_err < 1e-5 and cross_err < 1e-5) else "FAIL"
        print(f"  W={n}x{m}  p={p:>3}  q_n={q_n:>4} q_m={q_m:>4}  R={full_rank:>5}  "
              f"reconstruct_err={rec_err:.2e}  matvec_err={matvec_err:.2e}  "
              f"cross_err={cross_err:.2e}  {ok}")

    # ---------- Test 2: low-rank approximation error ----------
    print("\nTest 2: low-rank Monarch approximation error (random fp32 W, 4096x4096, p=64)")
    print("-" * 70)
    n = m = 4096; p = 64; q_n = q_m = 64
    W = torch.randn(n, m, device=device, dtype=dtype)
    x = torch.randn(m, device=device, dtype=dtype)
    y_dense = W @ x
    print(f"  {'rank':>6}  {'reconstruct_err':>16}  {'matvec_err':>12}  {'param_savings':>14}  {'flop_savings':>13}")
    full_params = n * m
    full_flops  = n * m
    for R in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
        A, B = monarch_factorize(W, p, q_n, q_m, R)
        W_rec = monarch_reconstruct(A, B, p, q_n, q_m)
        y_mono = monarch_matvec(A, B, x, p, q_n, q_m)
        rec_err = (W - W_rec).norm() / W.norm()
        mv_err = (y_dense - y_mono).norm() / y_dense.norm()
        m_params = A.numel() + B.numel()
        m_flops  = R * (p * q_n * q_m + p * p * q_n)
        print(f"  {R:>6}  {rec_err.item():>16.6f}  {mv_err.item():>12.6f}  "
              f"{full_params/m_params:>13.2f}x  {full_flops/m_flops:>12.2f}x")

    # ---------- Test 3: low-rank on a low-rank-ish W (separable structure should fit) ----------
    print("\nTest 3: rank-1 W constructed from outer product (p=64, q=64)")
    print("       Expected: rank-1 Monarch should reconstruct very well")
    print("-" * 70)
    a_outer = torch.randn(p * p, device=device, dtype=dtype)
    b_outer = torch.randn(q_n * q_m, device=device, dtype=dtype)
    F_lr = torch.outer(a_outer, b_outer)   # (p^2, q_n*q_m)
    W_lr = F_lr.reshape(p, p, q_n, q_m).permute(0, 2, 1, 3).reshape(n, m)
    x = torch.randn(m, device=device, dtype=dtype)
    y_dense = W_lr @ x
    for R in [1, 2, 4]:
        A, B = monarch_factorize(W_lr, p, q_n, q_m, R)
        y_mono = monarch_matvec(A, B, x, p, q_n, q_m)
        mv_err = (y_dense - y_mono).norm() / y_dense.norm()
        print(f"  R={R}: matvec_err={mv_err.item():.2e}  (rank-1 W should fit at R=1)")


if __name__ == "__main__":
    main()
