"""Phase 4/5/6 driver — runs after monarch_phase3.py has produced
monarch_phase3_results.json.

Phase 4: aggregate the before/after rotation errors and decide gate.
Phase 5: end-to-end quality test on Mistral-7B if Phase 4 passes.
Phase 6: wall-clock Monarch matvec timing if Phase 5 passes.
"""
import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monarch_phase1 import monarch_factorize, monarch_reconstruct, monarch_matvec

PHASE3_PATH = "/home/ubuntu/op31-prod-fix/monarch_phase3_results.json"
QUALITY_BAR_GATE = 0.07   # avg < 7% at rank=4 to proceed to Phase 5
HARD_FAIL_BAR    = 0.15   # if avg >= 15% at rank=8 even with rotation, STOP


def phase4_decision():
    if not os.path.exists(PHASE3_PATH):
        print(f"[phase4] {PHASE3_PATH} missing — run monarch_phase3.py first")
        return None
    with open(PHASE3_PATH) as f:
        d = json.load(f)
    print()
    print("=" * 80)
    print("PHASE 4: rotated Monarch quality decision")
    print("=" * 80)
    by_rank_after = {}
    for r in d["results"]:
        for k, v in r["after"].items():
            by_rank_after.setdefault(int(k), []).append(v)
    print(f"  {'rank':>5}  {'avg_after':>10}  {'gate':>20}")
    decisions = {}
    for R in sorted(by_rank_after):
        vals = by_rank_after[R]
        avg = sum(vals) / len(vals)
        gate = "PROCEED to Phase 5" if avg < QUALITY_BAR_GATE else (
               f"avg >= 15% at R={R}, STOP" if avg >= HARD_FAIL_BAR else
               f"between 7-15%, marginal")
        print(f"  {R:>5}  {avg:>10.4f}  {gate}")
        decisions[R] = (avg, gate)
    # Top-line decision: avg at rank=4
    avg4 = sum(by_rank_after[4]) / len(by_rank_after[4]) if 4 in by_rank_after else None
    avg8 = sum(by_rank_after[8]) / len(by_rank_after[8]) if 8 in by_rank_after else None
    print()
    if avg4 is not None and avg4 < QUALITY_BAR_GATE:
        print(f"[phase4] avg rotated rel_err at R=4 = {avg4:.4f} < 7% — PROCEED to Phase 5")
        return {"go": True, "min_rank_for_bar": 4}
    if avg8 is not None and avg8 < QUALITY_BAR_GATE:
        print(f"[phase4] R=4 misses bar but R=8 (avg={avg8:.4f}) clears it — PROCEED to Phase 5 with R=8")
        return {"go": True, "min_rank_for_bar": 8}
    if avg8 is not None and avg8 >= HARD_FAIL_BAR:
        print(f"[phase4] avg rotated rel_err at R=8 = {avg8:.4f} >= 15% — STOP")
        return {"go": False, "reason": "rotation does not bring error below 15% even at R=8"}
    print(f"[phase4] no rank passes 7% bar; not running Phase 5 (would degrade quality)")
    return {"go": False, "reason": "no rank passes 7% bar"}


def phase5_run_e2e(min_rank):
    """End-to-end Mistral-7B replacement: weights with rotated_err < 7% at min_rank become
    rotated-Monarch; others stay dense. Greedy decode 200 tokens, compare to baseline."""
    print()
    print("=" * 80)
    print(f"PHASE 5: end-to-end with rotated Monarch at R={min_rank}")
    print("=" * 80)
    # Load Phase 3 results to know which weights pass
    with open(PHASE3_PATH) as f:
        d = json.load(f)
    pass_set = {(r["layer"], r["name"]) for r in d["results"]
                 if r["after"][str(min_rank)] < QUALITY_BAR_GATE}
    print(f"[phase5] {len(pass_set)} of 224 weights pass 7% bar at R={min_rank}; will substitute these only")
    # Stub: this requires implementing RotatedMonarchLinear and re-running model.
    # We'll only flesh this out if Phase 4 actually says go.
    print("[phase5] (e2e substitution code intentionally elided until Phase 4 gates pass; will implement after)")


def phase6_wallclock_bench():
    """Microbench: dense matmul vs Monarch matvec via torch.einsum / bmm at Mistral shapes."""
    print()
    print("=" * 80)
    print("PHASE 6: wall-clock Monarch matvec vs dense matmul")
    print("=" * 80)
    device = "cuda"
    dtype = torch.float16
    shapes = [
        ("q_proj/o_proj", 4096, 4096, 64),
        ("k_proj/v_proj", 1024, 4096, 64),
        ("gate/up_proj",  14336, 4096, 64),
        ("down_proj",     4096, 14336, 64),
    ]
    print(f"  {'shape':<20} {'dense_us':>10} {'mono_R=1_us':>12} {'mono_R=4_us':>12} {'mono_R=8_us':>12}  {'speedup_R=4':>11}")
    for name, n, m, p in shapes:
        if n % p != 0 or m % p != 0:
            continue
        q_n = n // p; q_m = m // p
        W = torch.randn(n, m, device=device, dtype=dtype)
        x = torch.randn(m, device=device, dtype=dtype)
        # Warmup
        for _ in range(20):
            _ = W @ x
        torch.cuda.synchronize()
        # Dense timing
        t0 = time.perf_counter()
        N = 1000
        for _ in range(N):
            _ = W @ x
        torch.cuda.synchronize()
        dense_us = (time.perf_counter() - t0) / N * 1e6
        # Monarch at various ranks (using fp32 W for SVD then convert)
        W32 = W.to(torch.float32)
        timings = {}
        for R in [1, 4, 8]:
            A, B = monarch_factorize(W32, p, q_n, q_m, R)
            A_h = A.to(torch.float16); B_h = B.to(torch.float16)
            for _ in range(20):
                _ = monarch_matvec(A_h, B_h, x, p, q_n, q_m)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                _ = monarch_matvec(A_h, B_h, x, p, q_n, q_m)
            torch.cuda.synchronize()
            timings[R] = (time.perf_counter() - t0) / N * 1e6
        sp4 = dense_us / timings[4] if timings.get(4) else 0
        print(f"  {name:<20} {dense_us:>10.1f} {timings.get(1, 0):>12.1f} {timings.get(4, 0):>12.1f} "
              f"{timings.get(8, 0):>12.1f}  {sp4:>10.2f}x")


def main():
    decision = phase4_decision()
    if decision is None:
        return
    if decision.get("go"):
        phase5_run_e2e(decision["min_rank_for_bar"])
    # Phase 6 runs regardless — even if quality fails, the wall-clock data tells us
    # whether the algorithm has any speedup ceiling worth optimizing toward.
    phase6_wallclock_bench()


if __name__ == "__main__":
    main()
