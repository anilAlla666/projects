"""Phase 2 — Raw Monarch quality on all 224 Mistral-7B linear weights.

For each weight, sweep blockings (p=32, 64, 128) and ranks (1, 2, 4, 8, 16) and
record the Frobenius reconstruction error. Pick the best blocking per weight.

Decision: at which rank does the average rel_err fall below 7% (the naive INT4
quality bar measured this session at 97.5% top-1)?
"""
import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
RANKS = [1, 2, 4, 8, 16]
BLOCKINGS = [32, 64, 128]
QUALITY_BAR = 0.07  # naive INT4 rel_err level
LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def factorize_topk(W: torch.Tensor, p: int, q_n: int, q_m: int, max_k: int):
    """Compute top-k right SVD of the Monarch-permuted matrix F. Returns (U_k, S_k, Vh_k)."""
    n, m = W.shape
    W4 = W.reshape(p, q_n, p, q_m)
    F = W4.permute(0, 2, 1, 3).reshape(p * p, q_n * q_m).contiguous()
    # torch.svd_lowrank gives top-q via randomized SVD; over-sample by 6 for accuracy
    q = min(max_k + 6, min(F.shape))
    U, S, V = torch.svd_lowrank(F, q=q, niter=4)
    return U[:, :max_k], S[:max_k], V[:, :max_k].T  # Vh[k, :]


def reconstruct_at_rank(U_k, S_k, Vh_k, R: int, p: int, q_n: int, q_m: int):
    """Truncate stored top-k SVD to rank R, reconstruct full W approximation."""
    U_R = U_k[:, :R]
    S_R = S_k[:R]
    Vh_R = Vh_k[:R, :]
    F_approx = (U_R * S_R.unsqueeze(0)) @ Vh_R   # (p^2, q_n*q_m)
    W_approx = F_approx.reshape(p, p, q_n, q_m).permute(0, 2, 1, 3).reshape(p * q_n, p * q_m)
    return W_approx


def rel_err(W: torch.Tensor, W_approx: torch.Tensor) -> float:
    return ((W - W_approx).norm() / W.norm()).item()


def valid_blocking(n: int, m: int, p: int) -> bool:
    return (n % p == 0) and (m % p == 0)


def main():
    print(f"[load] {MODEL}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    n_layers = len(model.model.layers)
    print(f"[load] done in {time.time()-t0:.1f}s, {n_layers} layers")

    results = []  # list of dicts per (layer, name, blocking)
    t_start = time.time()

    for li, layer in enumerate(model.model.layers):
        for name in LINEARS:
            parent = layer.self_attn if name in ATTN else layer.mlp
            mod = getattr(parent, name)
            W = mod.weight.detach().to(torch.float32)  # (out, in)
            n, m = W.shape

            for p in BLOCKINGS:
                if not valid_blocking(n, m, p):
                    continue
                q_n = n // p
                q_m = m // p
                max_R = max(RANKS)
                # Capacity check: SVD rank cap = min(p^2, q_n*q_m)
                max_R = min(max_R, p * p, q_n * q_m)
                try:
                    U_k, S_k, Vh_k = factorize_topk(W, p, q_n, q_m, max_R)
                except Exception as e:
                    print(f"  [warn] L{li}_{name} blocking p={p} failed: {e}")
                    continue
                errs = {}
                for R in RANKS:
                    if R > max_R:
                        errs[R] = None
                        continue
                    W_approx = reconstruct_at_rank(U_k, S_k, Vh_k, R, p, q_n, q_m)
                    errs[R] = rel_err(W, W_approx)
                results.append({
                    "layer": li, "name": name, "shape": (n, m),
                    "p": p, "q_n": q_n, "q_m": q_m,
                    "errs": errs,
                })
        if (li + 1) % 4 == 0:
            elapsed = time.time() - t_start
            done = (li + 1) * len(LINEARS) * len(BLOCKINGS)
            print(f"[sweep] layer {li+1}/{n_layers} done; {done} (weight, blocking) cells; elapsed {elapsed:.1f}s")

    # ---- Pick best blocking per (layer, name, rank) and aggregate ----
    print()
    print("=" * 110)
    print("TABLE: best blocking per weight. Best is min over blockings of err_at_rank_R for the LARGEST R "
          "(rank=16) — using rank=16 as the tiebreaker so the same blocking is used at all ranks.")
    print("=" * 110)
    print(f"{'L':>2} {'name':<10} {'shape':>14} {'best_p':>6} "
          + "  ".join(f"err@R={R:<3}" for R in RANKS))
    print("-" * 110)

    best_per_weight = {}  # (li, name) -> result row with best blocking
    for li in range(n_layers):
        for name in LINEARS:
            cands = [r for r in results if r["layer"] == li and r["name"] == name and r["errs"][RANKS[-1]] is not None]
            if not cands:
                continue
            best = min(cands, key=lambda r: r["errs"][RANKS[-1]])
            best_per_weight[(li, name)] = best
            if li in (0, 5, 10, 15, 20, 25, 31):  # print samples
                shape_str = f"{best['shape'][0]}x{best['shape'][1]}"
                err_strs = "  ".join(f"{best['errs'][R]:.4f}" if best['errs'][R] is not None else " --- "
                                      for R in RANKS)
                print(f"{li:>2} {name:<10} {shape_str:>14} {best['p']:>6} " + err_strs)

    # ---- Per-rank aggregate stats over the 224 best-blocking results ----
    print()
    print("=" * 70)
    print("Aggregate rel_err across all 224 weights at best blocking per weight")
    print("=" * 70)
    print(f"{'rank':>6}  {'mean':>8}  {'median':>8}  {'p10':>8}  {'p90':>8}  {'max':>8}  {'< 7%':>8}")
    bar_passes_per_rank = {}
    for R in RANKS:
        errs_R = [bp["errs"][R] for bp in best_per_weight.values() if bp["errs"][R] is not None]
        if not errs_R:
            continue
        errs_sorted = sorted(errs_R)
        n = len(errs_sorted)
        mean = sum(errs_sorted) / n
        median = errs_sorted[n // 2]
        p10 = errs_sorted[int(0.1 * n)]
        p90 = errs_sorted[int(0.9 * n)]
        mx = errs_sorted[-1]
        n_pass = sum(1 for e in errs_R if e < QUALITY_BAR)
        bar_passes_per_rank[R] = n_pass
        print(f"{R:>6}  {mean:>8.4f}  {median:>8.4f}  {p10:>8.4f}  {p90:>8.4f}  {mx:>8.4f}  {n_pass:>4}/{n}")

    # Decision
    print()
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    for R in RANKS:
        errs = [bp["errs"][R] for bp in best_per_weight.values() if bp["errs"][R] is not None]
        mean = sum(errs)/len(errs)
        verdict = "OK to gate" if mean < QUALITY_BAR else "fails 7% bar"
        print(f"  rank={R}: avg rel_err={mean:.4f} -> {verdict}")

    # Save
    out_path = "/home/ubuntu/op31-prod-fix/monarch_phase2_results.json"
    serial = []
    for r in results:
        serial.append({
            "layer": r["layer"], "name": r["name"], "shape": list(r["shape"]),
            "p": r["p"], "q_n": r["q_n"], "q_m": r["q_m"],
            "errs": {str(k): v for k, v in r["errs"].items()},
        })
    with open(out_path, "w") as f:
        json.dump({
            "ranks": RANKS, "blockings": BLOCKINGS,
            "quality_bar": QUALITY_BAR,
            "n_weights_total": len(best_per_weight),
            "results": serial,
            "bar_passes_per_rank": bar_passes_per_rank,
        }, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()
