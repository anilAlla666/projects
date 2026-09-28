"""ACTION 1: fp32-inner ProjectedLinear at k=499 decode-only.

Reuses calibrate(), compute_qs(), free_run_baseline(), teacher_forced_run(),
compute_metrics(), patch_model(), unpatch_model() from projection_quality.

Single change: ProjectedLinear.forward upcasts to fp32 for the (x @ Q) @ M.t()
chain, then casts back to fp16. Tests whether the 22.5% top1 gap at k=499 was
caused by fp16 chained-matmul accumulation or by the algorithm itself.

Runs ONLY the k=499 decode-only diagnostic (no v1/v2/sweep).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

import projection_quality as pq

# Override ProjectedLinear forward to use fp32 inner accumulation.
class ProjectedLinearFp32(nn.Module):
    def __init__(self, orig: nn.Linear, Q: torch.Tensor, dtype=torch.float16):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.k = Q.shape[1]
        device = orig.weight.device
        with torch.no_grad():
            W = orig.weight.data.to(torch.float32)  # [out, in]
            Qf = Q.to(device=device, dtype=torch.float32)
            M = W @ Qf  # [out, k]
        # Store in fp32 (NOT fp16) so the inner products run in fp32 with no upcast cost.
        self.register_buffer("Q", Qf)        # fp32
        self.register_buffer("M", M)         # fp32
        self.bias = orig.bias

    def forward(self, x):
        # x is fp16 (Mistral activation). Upcast for the inner matmul chain,
        # cast back at the end.
        x32 = x.to(torch.float32)
        z = x32 @ self.Q          # [..., k]  fp32
        y = z @ self.M.t()        # [..., out] fp32
        y = y.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y


# Monkey-patch ProjectedLinear in pq module so existing patch_model uses fp32 version.
pq.ProjectedLinear = ProjectedLinearFp32


def main():
    print("=== ACTION 1: fp32-inner ProjectedLinear, k=499, decode-only ===", flush=True)
    print(f"baseline (fp16-inner) result from prior run: top1=0.7750  mean_kl=1.2827", flush=True)
    print(f"hypothesis: fp32 inner accumulation closes the 22.5% gap if it was numerics, not algorithm", flush=True)
    print(flush=True)

    model, tok = pq.load_model()

    print("[sanity] verify ProjectedLinearFp32 matches dense W@x at full rank (small test)", flush=True)
    torch.manual_seed(0)
    n, m = 1024, 4096
    orig = nn.Linear(m, n, bias=False).to("cuda").half()
    X = (torch.randn(500, m, device="cuda", dtype=torch.float32) * 0.1)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Q_full = Vh.t().contiguous()
    proj = ProjectedLinearFp32(orig, Q_full).to("cuda")
    x_test = X[0:1].to(torch.float16)
    y_full = orig(x_test)
    y_proj = proj(x_test)
    err = (y_full.float() - y_proj.float()).norm() / y_full.float().norm()
    print(f"[sanity]   full-rank rel_err vs dense: {err:.2e}  (was ~5e-4 in fp16)", flush=True)

    # Calibration (5 prompts x 100 decode tokens — matches projection_quality.py)
    captures = pq.calibrate(model, tok)
    qs = pq.compute_qs(captures, k_max=pq.K_MAX)
    del captures

    # Eval baseline (free run)
    eval_ids = tok(pq.EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"\n[baseline] eval prompt={pq.EVAL_PROMPT!r} prompt_len={eval_ids.shape[1]}", flush=True)
    baseline = pq.free_run_baseline(model, eval_ids, pq.N_EVAL_DECODE)
    base_text = tok.decode(baseline["decoded_tokens"])
    print(f"[baseline] free-running text: {base_text[:200]!r}", flush=True)

    # ONLY the k=499 decode-only diagnostic
    res_diag = pq.run_projected_variant(
        model, tok, qs, pq.K_PROFILES["v4_uniform_512"], eval_ids, baseline,
        "diag_k499_fp32_decode_only", decode_only=True
    )
    pq.print_metrics("diag_k499_fp32_decode_only", res_diag, pq.K_PROFILES["v4_uniform_512"])

    m = res_diag["metrics"]
    print(f"\n=== RESULT ===", flush=True)
    print(f"  fp16-inner (prior run): top1=0.7750  mean_kl=1.2827  p99_kl=15.6", flush=True)
    print(f"  fp32-inner (this run):  top1={m['top1_match']:.4f}  mean_kl={m['mean_kl']:.4f}  p99_kl={m['p99_kl']:.4f}", flush=True)

    if m['top1_match'] >= 0.95:
        print(f"\n  --> SUBSPACE PATH REOPENS. fp16 chain drift was the problem; fp32 inner restores it.", flush=True)
    elif m['top1_match'] >= 0.85:
        print(f"\n  --> Marginal improvement. fp32 helps but algorithm has additional ceiling.", flush=True)
    else:
        print(f"\n  --> Algorithm doesn't compose at 224-linear chain even in fp32.", flush=True)


if __name__ == "__main__":
    main()
