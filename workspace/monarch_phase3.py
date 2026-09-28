"""Phase 3 — ProcrustesGPT-style rotation for Monarch factorization.

Find orthogonal Q such that W @ Q has more Monarch structure than W itself.
Iterative (alternating) algorithm:
  1. Hold Q fixed, compute Monarch approximation M = Monarch(W @ Q) at given rank.
  2. Hold M fixed, find Q orthogonal minimizing ||W Q - M||_F via SVD(W^T M) = U S V^T,
     Q = U V^T.
  3. Iterate to convergence.

Output: per-weight Q matrix and rotated-Monarch reconstruction error.

We start with UNIFORM weighting (no activation distribution). If this doesn't
bring error below 30% even at rank=4-8, the activation-weighted version won't
either. If uniform helps, Phase 3b can add activation weighting on top.
"""
import time
import json

import torch
from transformers import AutoModelForCausalLM

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monarch_phase1 import monarch_factorize, monarch_reconstruct

MODEL = "mistralai/Mistral-7B-v0.1"
RANKS = [1, 2, 4, 8, 16]
QUALITY_BAR = 0.07


def procrustes_rotation(W: torch.Tensor, p: int, q_n: int, q_m: int,
                        rank: int, n_iter: int = 8):
    """Iteratively find orthogonal Q minimizing ||W Q - Monarch(W Q)||_F at given rank.

    Returns Q (m, m) and the final approximation error.
    """
    n, m = W.shape
    Q = torch.eye(m, device=W.device, dtype=W.dtype)
    err_history = []
    best_err = float("inf")
    best_Q = Q.clone()
    for it in range(n_iter):
        W_rot = W @ Q  # (n, m)
        A, B = monarch_factorize(W_rot, p, q_n, q_m, rank)
        M = monarch_reconstruct(A, B, p, q_n, q_m)  # (n, m), Monarch approximation of W_rot
        # Error in rotated frame == error in original frame (rotation preserves Frobenius norm)
        err = ((W_rot - M).norm() / W_rot.norm()).item()
        err_history.append(err)
        if err < best_err:
            best_err = err
            best_Q = Q.clone()
        # Procrustes step: find Q' orthogonal s.t. W Q' ~= M.
        # Solve min over Q orthogonal of ||W Q - M||_F. Q = U V^T where U S V^T = SVD(W^T M).
        WtM = W.T @ M  # (m, m)
        try:
            U, S, Vh = torch.linalg.svd(WtM)
        except torch._C._LinAlgError:
            break
        Q = U @ Vh
    # Recompute error at best_Q
    W_rot = W @ best_Q
    A, B = monarch_factorize(W_rot, p, q_n, q_m, rank)
    M = monarch_reconstruct(A, B, p, q_n, q_m)
    final_err = ((W_rot - M).norm() / W_rot.norm()).item()
    return best_Q, final_err, err_history


def baseline_err(W, p, q_n, q_m, rank):
    A, B = monarch_factorize(W, p, q_n, q_m, rank)
    M = monarch_reconstruct(A, B, p, q_n, q_m)
    return ((W - M).norm() / W.norm()).item()


def quick_test():
    """Sanity check: does Procrustes rotation help on a few weights at rank=4?"""
    print("=" * 80)
    print("Quick smoke test: 1 weight per layer category, rank=4, p=64")
    print("=" * 80)
    print(f"  {'name':<10} {'shape':>14} {'p':>3} {'before':>10} {'after':>10} {'improvement':>12}")
    print("-" * 80)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    layer = model.model.layers[15]  # mid layer
    samples = [
        ("q_proj", layer.self_attn.q_proj.weight),
        ("k_proj", layer.self_attn.k_proj.weight),
        ("o_proj", layer.self_attn.o_proj.weight),
        ("gate_proj", layer.mlp.gate_proj.weight),
        ("down_proj", layer.mlp.down_proj.weight),
    ]
    for name, W_fp16 in samples:
        W = W_fp16.detach().to(torch.float32)
        n, m = W.shape
        p = 64
        # Pick valid blocking
        if n % p != 0 or m % p != 0:
            print(f"  {name}: bad blocking, skip")
            continue
        q_n = n // p
        q_m = m // p
        before = baseline_err(W, p, q_n, q_m, 4)
        Q, after, hist = procrustes_rotation(W, p, q_n, q_m, 4, n_iter=8)
        ratio = before / after if after > 0 else float("inf")
        hist_str = " ".join(f"{e:.3f}" for e in hist)
        print(f"  {name:<10} {n}x{m:<12} {p:>3} {before:>10.4f} {after:>10.4f} {ratio:>11.2f}x"
              f"  hist=[{hist_str}]")
    return model


def full_sweep(model):
    """Full sweep: best-blocking Procrustes for all 224 weights at all ranks."""
    print()
    print("=" * 80)
    print("Full sweep: 224 weights x ranks {1,2,4,8,16}, Procrustes rotation, p=64")
    print("=" * 80)

    results = []
    n_layers = len(model.model.layers)
    LINEARS = [
        ("q_proj", "self_attn"), ("k_proj", "self_attn"),
        ("v_proj", "self_attn"), ("o_proj", "self_attn"),
        ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp"),
    ]

    t_start = time.time()
    for li in range(n_layers):
        layer = model.model.layers[li]
        for name, parent_name in LINEARS:
            parent = getattr(layer, parent_name)
            mod = getattr(parent, name)
            W = mod.weight.detach().to(torch.float32)
            n, m = W.shape
            p = 64
            if n % p != 0 or m % p != 0:
                continue
            q_n = n // p
            q_m = m // p

            row = {"layer": li, "name": name, "shape": [n, m], "p": p,
                   "before": {}, "after": {}, "improvement": {}}
            for R in RANKS:
                before = baseline_err(W, p, q_n, q_m, R)
                _Q, after, _hist = procrustes_rotation(W, p, q_n, q_m, R, n_iter=5)
                row["before"][R] = before
                row["after"][R] = after
                row["improvement"][R] = before / after if after > 1e-9 else float("inf")
            results.append(row)
        if (li + 1) % 4 == 0:
            elapsed = time.time() - t_start
            print(f"  layer {li+1}/{n_layers}  elapsed={elapsed:.1f}s "
                  f"({(li+1)/n_layers*100:.0f}% done)")

    return results


def report(results):
    print()
    print("=" * 100)
    print("Aggregate stats: rotated Monarch rel_err (mean across 224 weights at p=64)")
    print("=" * 100)
    print(f"  {'rank':>6}  {'before_avg':>11}  {'after_avg':>11}  {'after_med':>11}  "
          f"{'after_p90':>11}  {'after_max':>11}  {'< 7%':>8}  {'<15%':>8}  {'<30%':>8}")
    for R in RANKS:
        befores = [r["before"][R] for r in results]
        afters  = [r["after"][R]  for r in results]
        afters_sorted = sorted(afters)
        n = len(afters_sorted)
        b_mean = sum(befores) / n
        a_mean = sum(afters)  / n
        a_med  = afters_sorted[n // 2]
        a_p90  = afters_sorted[int(0.9 * n)]
        a_max  = afters_sorted[-1]
        n7   = sum(1 for e in afters if e < 0.07)
        n15  = sum(1 for e in afters if e < 0.15)
        n30  = sum(1 for e in afters if e < 0.30)
        print(f"  {R:>6}  {b_mean:>11.4f}  {a_mean:>11.4f}  {a_med:>11.4f}  "
              f"{a_p90:>11.4f}  {a_max:>11.4f}  {n7:>4}/{n}  {n15:>4}/{n}  {n30:>4}/{n}")

    # Sample table
    print()
    print("=" * 100)
    print("Sample (selected layers): before vs after rotation at rank=4")
    print("=" * 100)
    print(f"  {'L':>3} {'name':<10} {'shape':>14} {'before@R=4':>12} {'after@R=4':>11} {'improvement':>12}")
    for r in results:
        if r["layer"] in (0, 5, 15, 25, 31):
            shape_s = f"{r['shape'][0]}x{r['shape'][1]}"
            print(f"  {r['layer']:>3} {r['name']:<10} {shape_s:>14}"
                  f"  {r['before'][4]:>12.4f}  {r['after'][4]:>11.4f}  {r['improvement'][4]:>11.2f}x")

    # Save raw
    out = "/home/ubuntu/op31-prod-fix/monarch_phase3_results.json"
    with open(out, "w") as f:
        ser = []
        for r in results:
            ser.append({
                "layer": r["layer"], "name": r["name"], "shape": r["shape"], "p": r["p"],
                "before": {str(k): v for k, v in r["before"].items()},
                "after":  {str(k): v for k, v in r["after"].items()},
                "improvement": {str(k): (v if v != float("inf") else None) for k, v in r["improvement"].items()},
            })
        json.dump({"ranks": RANKS, "results": ser}, f, indent=2)
    print(f"\n[save] -> {out}")


def main():
    model = quick_test()
    results = full_sweep(model)
    report(results)


if __name__ == "__main__":
    main()
