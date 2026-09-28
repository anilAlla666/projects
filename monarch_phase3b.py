"""Phase 3 (refactored) — Procrustes rotation for Monarch, with side-aware rotation.

Key optimization: for W shape (n, m), rotate whichever side is SMALLER:
  - If m <= n: rotate input. Q is m x m. W' = W Q. Forward: y = W' Q^T x.
  - If m >  n: rotate output. Q is n x n. W' = Q^T W. Forward: y = Q W' x.

This keeps Procrustes SVD at most max(n, m) x max(n, m) = 4096 x 4096 for Mistral
(was 14336 x 14336 = 136s per SVD before, now 1.2s).

Initial pass: rank=4 only (the rank the user wants for Phase 5). Wider sweep
runs as a follow-up if rank=4 results are promising.
"""
import os, sys, time, json
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monarch_phase1 import monarch_factorize, monarch_reconstruct

MODEL = "mistralai/Mistral-7B-v0.1"
RANK_PRIMARY = 4
QUALITY_BAR = 0.07
N_ITER = 5
P_BLOCK = 64

LINEARS = [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"),
    ("v_proj", "self_attn"), ("o_proj", "self_attn"),
    ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp"),
]


def baseline_err(W, p, q_n, q_m, R):
    A, B = monarch_factorize(W, p, q_n, q_m, R)
    M = monarch_reconstruct(A, B, p, q_n, q_m)
    return ((W - M).norm() / W.norm()).item()


def procrustes_rotation(W: torch.Tensor, p: int, R: int, n_iter: int = N_ITER):
    """Find rotation Q minimizing low-rank Monarch error.

    Auto-selects input vs output rotation based on which side is smaller.
    Returns (Q, side, final_err, err_history) where side ∈ {'input', 'output'}.

    Forward semantics:
      - side='input':  y = W Q (Q^T x)        --  pre-divide x by Q
      - side='output': y = Q (Q^T W) x = Q W' x  --  post-multiply by Q
    """
    n, m = W.shape
    use_output = m > n  # rotate the smaller side

    if not use_output:
        side = "input"
        # Standard formulation: W' = W Q
        q_n = n // p
        q_m = m // p
        Q = torch.eye(m, device=W.device, dtype=W.dtype)
        best_err = float("inf")
        best_Q = Q.clone()
        history = []
        for it in range(n_iter):
            W_rot = W @ Q
            A, B = monarch_factorize(W_rot, p, q_n, q_m, R)
            M = monarch_reconstruct(A, B, p, q_n, q_m)
            err = ((W_rot - M).norm() / W_rot.norm()).item()
            history.append(err)
            if err < best_err:
                best_err = err
                best_Q = Q.clone()
            WtM = W.T @ M  # (m, m)
            U, S, Vh = torch.linalg.svd(WtM)
            Q = U @ Vh
        return best_Q, side, best_err, history
    else:
        side = "output"
        # Output rotation: W' = Q^T W. Frobenius norm preserved by orthogonal Q.
        q_n = n // p
        q_m = m // p
        Q = torch.eye(n, device=W.device, dtype=W.dtype)
        best_err = float("inf")
        best_Q = Q.clone()
        history = []
        for it in range(n_iter):
            W_rot = Q.T @ W                       # (n, m)
            A, B = monarch_factorize(W_rot, p, q_n, q_m, R)
            M = monarch_reconstruct(A, B, p, q_n, q_m)  # (n, m)
            err = ((W_rot - M).norm() / W_rot.norm()).item()
            history.append(err)
            if err < best_err:
                best_err = err
                best_Q = Q.clone()
            # Find Q minimizing ||Q^T W - M||_F = ||W - Q M||_F
            # Solution: SVD(W M^T) = U S V^T, Q = U V^T
            WMt = W @ M.T   # (n, n)
            U, S, Vh = torch.linalg.svd(WMt)
            Q = U @ Vh
        return best_Q, side, best_err, history


def main():
    print(f"[load] {MODEL}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    n_layers = len(model.model.layers)
    print(f"\n[phase3b] sweeping {n_layers} layers x {len(LINEARS)} linears at rank=4, p={P_BLOCK}", flush=True)
    print(f"[phase3b] each weight: baseline_err + procrustes_rotation (n_iter={N_ITER})", flush=True)
    print(f"[phase3b] uses input rotation when m<=n, output rotation when m>n (so SVD stays 4096x4096)", flush=True)

    results = []
    t_start = time.time()

    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, parent_name in LINEARS:
            parent = getattr(layer, parent_name)
            mod = getattr(parent, name)
            W = mod.weight.detach().to(torch.float32)
            n, m = W.shape
            if n % P_BLOCK != 0 or m % P_BLOCK != 0:
                continue
            q_n = n // P_BLOCK
            q_m = m // P_BLOCK

            t_w = time.perf_counter()
            before = baseline_err(W, P_BLOCK, q_n, q_m, RANK_PRIMARY)
            Q, side, after, hist = procrustes_rotation(W, P_BLOCK, RANK_PRIMARY, n_iter=N_ITER)
            t_w = time.perf_counter() - t_w
            ratio = before / after if after > 1e-9 else float("inf")
            results.append({
                "layer": li, "name": name, "shape": [n, m],
                "side": side, "before": before, "after": after,
                "improvement": ratio, "history": hist, "time_s": t_w,
            })
        elapsed = time.time() - t_start
        eta = elapsed / (li + 1) * (n_layers - li - 1)
        # Per-layer summary
        layer_results = [r for r in results if r["layer"] == li]
        layer_avg_before = sum(r["before"] for r in layer_results) / len(layer_results)
        layer_avg_after  = sum(r["after"]  for r in layer_results) / len(layer_results)
        print(f"[layer {li:>2}/{n_layers}] avg_before={layer_avg_before:.4f}  "
              f"avg_after={layer_avg_after:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
              flush=True)

    # ---- Aggregate ----
    print()
    print("=" * 96, flush=True)
    print(f"AGGREGATE — Procrustes rotation at rank={RANK_PRIMARY}, p={P_BLOCK}, n_iter={N_ITER}", flush=True)
    print("=" * 96, flush=True)
    befores = [r["before"] for r in results]
    afters  = [r["after"]  for r in results]
    n = len(results)
    print(f"  n_weights={n}", flush=True)
    print(f"  avg before: {sum(befores)/n:.4f}", flush=True)
    print(f"  avg after:  {sum(afters)/n:.4f}", flush=True)
    print(f"  min after:  {min(afters):.4f}", flush=True)
    print(f"  max after:  {max(afters):.4f}", flush=True)
    print(f"  pass <7%:   {sum(1 for e in afters if e < 0.07)}/{n}", flush=True)
    print(f"  pass <15%:  {sum(1 for e in afters if e < 0.15)}/{n}", flush=True)
    print(f"  pass <30%:  {sum(1 for e in afters if e < 0.30)}/{n}", flush=True)

    # Per-name aggregates
    print(f"\n  {'name':<10}  {'n':>3}  {'avg_before':>10}  {'avg_after':>10}  {'best_after':>11}")
    by_name = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)
    for name, rs in by_name.items():
        ab = sum(r["before"] for r in rs) / len(rs)
        aa = sum(r["after"]  for r in rs) / len(rs)
        ba = min(r["after"]  for r in rs)
        print(f"  {name:<10}  {len(rs):>3}  {ab:>10.4f}  {aa:>10.4f}  {ba:>11.4f}", flush=True)

    # Sample table
    print(f"\nSample weights (layer 0/15/31, all linears):", flush=True)
    print(f"  {'L':>3} {'name':<10} {'shape':>14} {'side':>7} {'before':>9} {'after':>9} {'ratio':>9}", flush=True)
    for r in results:
        if r["layer"] in (0, 15, 31):
            shape_s = f"{r['shape'][0]}x{r['shape'][1]}"
            print(f"  {r['layer']:>3} {r['name']:<10} {shape_s:>14} {r['side']:>7} "
                  f"{r['before']:>9.4f} {r['after']:>9.4f} {r['improvement']:>8.2f}x", flush=True)

    # Save
    out = "/home/ubuntu/op31-prod-fix/monarch_phase3b_results.json"
    with open(out, "w") as f:
        json.dump({
            "rank": RANK_PRIMARY, "p": P_BLOCK, "n_iter": N_ITER,
            "results": results,
        }, f, indent=2)
    print(f"\n[save] -> {out}", flush=True)

    # Decision
    avg_after = sum(afters) / n
    print(f"\n=== DECISION ===")
    if avg_after < QUALITY_BAR:
        print(f"avg after = {avg_after:.4f} < 7% bar  -->  PROCEED to Phase 4/5/6")
    elif avg_after < 0.15:
        print(f"avg after = {avg_after:.4f} between 7% and 15%  -->  marginal; try larger rank")
    else:
        print(f"avg after = {avg_after:.4f} >= 15%  -->  rotation does not help enough; STOP")


if __name__ == "__main__":
    main()
