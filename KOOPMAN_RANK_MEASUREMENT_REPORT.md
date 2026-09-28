# Koopman / rank measurement — Mistral-7B, H100 SXM

**2026-06-04. READ-ONLY. No `.so` change** — anchor `cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2` unchanged. Part A
(offline frontier) replicates the shipped EDMD fit in PyTorch via forward hooks; Part B (live) drives the **deployed `.so`'s
real kernels** via direct C-API (`cipher_edmd_live_collect` → randomized-SVD fit → `cipher_koopman_fp16_ood_max_residual`,
`cipher_koopman_fp16_launch_shape`). Harness: `koopman_rank_sweep.py`, `koopman_partB_live.py`. Results:
`koopman_rank_results.json`, `koopman_partB_results.json`.

---

## TL;DR

The Koopman actuator is a **rank-r EDMD surrogate** (shipped `KR_RANK=64`) that replaces an FP16 GEMM `Y=XW` with a
data-driven reduced-rank map `Y≈X·Â_r` fit from real activation snapshots, fires only when the runtime residual passes an
**OOD gate (β=0.05)**. The rank question — *does r=64 work, and what rank would?* — resolves to a clean **earned negative**:

1. **The activation manifold is not low-rank enough.** Measured effective rank (of the real activations EDMD truncates):
   lm_head **844 @90% / 2630 @99%** energy (of 4096); down_proj **2409 / 5282** (of 14336); q_proj **895 / 2661**. r=64 is
   **~13–40× too small.**
2. **At r=64 the surrogate fails its own 5% OOD gate by ~20×** — held-out residual 0.43 (lm_head, offline) and **0.96 live**;
   it fails the gate at **every** rank swept, even with exact SVD. This is the mechanism behind the prior "ships correct,
   never fires" finding — **now measured on the live OOD kernel: residual 0.91–0.97 ≫ 0.05 → does not fire.**
3. **lm_head quality at r=64 is destroyed:** ΔPPL **+2,560%** (held-out), **+32,000%** (cross-corpus), top-1 0.33. To reach
   the project's +0.37% PPL bar you'd need r≈2048 (still +54% — **≈150× over the +0.37% bar**) — where the FLOP ceiling is
   only **1.8×**. (And even a tensor-core-optimized kernel could not rescue the approach: at the ranks where a speedup
   survives, the quality is unusable; at the ranks where quality survives, the speedup is gone.)
4. **The realized speedup is a slowdown.** The FLOP-ratio "speedup" `K·N/(r·(K+N))` is a theoretical ceiling (56.7× for
   lm_head @r=64). A cuBLAS two-GEMM best-case realizes only 3.6–11.7×. **The actual shipped `.cu` surrogate kernel runs
   at 0.04× — i.e. 25× *slower* than the cuBLAS GEMM it replaces** (q_proj 0.05× / 20× slower).

**Both legendary numbers are phantoms.** "LM-head 7.43×" has no located provenance (mechanism dossier 2026-05-30: "NOT
FOUND"); it would correspond to r≈489, where lm_head ΔPPL is in the hundreds of %. "rank ~1773" (a code-comment advisor
citation, no measurement) lands *inside* lm_head's measured 90→99% energy band — coincidentally near the truth, but it was
never measured. **Net: the rank-r Koopman surrogate cannot be simultaneously fast and accurate on these operators; at the
shipped r=64 it is both inaccurate and 20–25× slower, and it correctly never fires.**

---

## Setup & methodology

| | |
|---|---|
| GPU / model | H100 80GB HBM3 SXM; Mistral-7B-v0.1 **float16** (the actuator's only dtype — it gates on `CUDA_R_16F`) |
| Target ops | `lm_head` (K=4096,N=32000), `down_proj`@L16 (K=14336,N=4096), `q_proj`@L16 (K=4096,N=4096) |
| Shipped fit (faithfully replicated) | `cipher_edmd_live.cpp`: rand rank-r SVD of **X** (activations, not weights) → `Â = Vₓ·diag(1/σ)·Uₓᵀ·Y`; out=`X·Â`. `KR_RANK=64`, `TARGET_ROWS=2000`, OOD gate `β=0.05`. |
| Corpora | wikitext-2 train / held-out (same dist) / **code (cross-corpus OOD)** — held-out + cross-corpus are the load-bearing residuals (the actuator's gate uses *in-sample*, which understates error) |
| Exact vs randomized SVD | Part A uses exact `torch.linalg.svd` ⇒ residual is an **optimistic lower bound** vs the shipped randomized SVD (Part B confirms: shipped energy@r64 ≈ 0.69 matches) |
| Speedup | FLOP ratio `K·N/(r·(K+N))` = **theoretical ceiling**; reported alongside **measured wall-clock** (cuBLAS two-GEMM best-case in A; the real `.cu` kernel in B), locked 1200 MHz |

---

## Part A — offline rank frontier (faithful EDMD replication)

**Effective rank of the activation manifold** (the quantity that decides whether truncation to r works). *Participation
ratio* = `(Σσ²)² / Σσ⁴`, a spectrum-spread measure (low ⇒ few dominant modes); the load-bearing numbers are rank@90/99%
and the residuals below.

| op | participation ratio | rank@90% energy | rank@99% energy | (of K) |
|---|---|---|---|---|
| lm_head | 15.4 | **844** | **2630** | 4096 |
| down_proj | 175.7 | **2409** | **5282** | 14336 |
| q_proj | 25.7 | **895** | **2661** | 4096 |

**lm_head rank sweep** (residual = ‖Y−XÂ‖/‖Y‖; OOD gate needs held-out ≤ 0.05):

| r | energy | resid in / held / **cross** | OOD≤.05? | FLOP ceil | KL held | top-1 | **ΔPPL%** held / cross |
|--|--|--|--|--|--|--|--|
| 64 | 0.64 | 0.25 / **0.43** / 0.79 | ✗ | 56.7× | 3.35 | 0.33 | **+2,560 / +32,202** |
| 256 | 0.78 | 0.19 / 0.36 / 0.68 | ✗ | 14.2× | 2.29 | 0.47 | +856 / +4,890 |
| 512 | 0.85 | 0.15 / 0.32 / 0.61 | ✗ | 7.1× | 1.73 | 0.54 | +450 / +1,782 |
| 1024 | 0.92 | 0.11 / 0.27 / 0.51 | ✗ | 3.5× | 1.10 | 0.62 | +198 / +479 |
| 2048 | 0.98 | 0.06 / 0.20 / 0.37 | ✗ | 1.8× | 0.44 | 0.74 | +55 / +76 |

down_proj and q_proj are the same shape of result (down_proj worse: held-out residual 0.93 @r=64, 0.76 @r=2048). **No rank,
on any operator, passes the 5% OOD gate on held-out data** — even with exact SVD. Quality and speedup move in opposite
directions: the rank that would preserve lm_head PPL (≳2048) erases the FLOP advantage (≤1.8×).

**Wall-clock microbench (cuBLAS two-GEMM best-case, locked 1200 MHz) — realized ≪ FLOP ceiling:**

| op | M=1 (decode) | M=512 | M=2048 (prefill) | FLOP ceil @r64 |
|--|--|--|--|--|
| lm_head | 3.6× | 7.1× | 11.7× | 56.7× |
| down_proj | 1.9× | 4.0× | 8.1× | 49.8× |
| q_proj | **0.54×** | 1.4× | 4.7× | 32.0× |

Even the cuBLAS best-case realizes a fraction of the ceiling, and **at decode (M=1) q_proj is already a slowdown** — the
r=64 inner dimension is too small to be efficient, and you pay two kernel launches + an (m,r) intermediate.

---

## Part B — live shipped reality (deployed `.so`, real kernels)

Drove the shipped `cipher_edmd_live_collect` with 2,000 real activation rows → background randomized-SVD rank-64 fit →
registered, then exercised the real OOD kernel and surrogate kernel:

| op | registered | shipped energy@r64 | **runtime OOD resid** held / cross | **fires?** (β=.05) | kernel rel-err | **realized speedup** (FLOP ceil) |
|--|--|--|--|--|--|--|
| lm_head | ✓ (2000 rows) | 0.69 | **0.96 / 0.97** | **No** | 0.36 (top-1 0.27) | **0.04× = 25× slower** (56.7×) |
| q_proj | ✓ (2000 rows) | 0.69 | **0.91 / 0.91** | **No** | 0.50 | **0.05× = 20× slower** (32.0×) |

Three live confirmations: (1) the shipped randomized-SVD energy (0.69) matches the offline exact-SVD energy (~0.64) — the
offline frontier is a faithful, slightly optimistic replica. (2) The runtime OOD gate residual is **0.91–0.97 ≫ 0.05**, so
the actuator **provably does not fire** on real held-out activations — the live mechanism behind "never fires." (3) The
real `.cu` surrogate kernel's quality matches the offline residual (rel-err 0.36, top-1 0.27 ≈ offline 0.43 / 0.33) **and is
20–25× slower than cuBLAS** — the FLOP ceiling is not just unrealized, it is inverted by an unoptimized scalar kernel.

---

## Caveats / honest bounds

- **Exact-SVD residuals (Part A) are an optimistic lower bound**; the shipped randomized SVD is no better (Part B confirms).
- **In-sample vs held-out vs cross-corpus** are reported separately; the actuator's own gate uses *in-sample*, which is the
  most optimistic — held-out/cross are the deployment-relevant numbers, and both fail.
- **Layer 16 chosen as representative** for down_proj/q_proj; lm_head is unique (the headline). Other layers may differ
  modestly but the spectra are characteristic of transformer activations (high effective rank).
- **Part B counters** (`cipher_rt_koopman_calls_*`) are the matmul-*dispatch* path; this probe used the direct C-API, so
  those read 0 — engagement here is established by the OOD-gate residual (≫β) and confirmed by the prior reachability doc.
- The **runtime OOD residual (0.91–0.97) is a per-row-max input-subspace metric**, stricter than the offline output
  residual (0.43); both are ≫ the 0.05 gate, so the no-fire conclusion is robust to the metric choice.
- This measures the **rank/quality/speed** of the surrogate; it does not re-litigate the engine-composition question (that
  was the prior g1.3 finding: capture-illegal calibration feed).

---

## Independent verification

Four adversarial reviewers (fit-faithfulness, live-kernel skeptic, negative-robustness, provenance/completeness) checked it
against the shipped source. **Fit:** the offline replica computes the *same* map as `cipher_edmd_live.cpp` (Â = Vₓ·diag(1/σ)·Uₓᵀ·Y
from SVD of activations), held-out correctly materializes Â from train (no `UₓUₓᵀY` shortcut); all cells reproduced
(ΔPPL, FLOP ceilings, wall-clock ratios). **Live-kernel skeptic:** the 25× slowdown is **real compute, not an artifact** —
with `CIPHER_USE_CACHE=0` the generic scalar kernel (`cipher_koopman_fp16_decode_generic`, no tensor cores) actually runs;
the cuBLAS comparison is fair (same harness, locked clock); ~0.1 TFLOP/s vs cuBLAS ~263 TFLOP/s. **Robustness:** every
attempt to find a regime where r=64 would work (other operators, narrower corpus, decode M=1, a faster kernel) *reinforced*
the negative. **Phantoms** confirmed (7.43× "NOT FOUND" in prior dossier; "1773" an unsourced code comment). No core claim
refuted; only minor wording fixes (folded in above). Note: the live OOD residual (0.96, a per-row-max input-subspace
metric) and the offline residual (0.43, output-space) are *different metrics* — both ≫ the 0.05 gate, so the no-fire
conclusion is metric-robust.

## Verdict

The shipped rank-64 Koopman surrogate is an **earned negative on every axis**: the activation manifold needs hundreds to
thousands of modes (lm_head 844–2630), so r=64 captures too little, fails its own 5% OOD gate by ~20× (live residual 0.96),
and destroys lm_head quality (ΔPPL +2,560%); the rank that would preserve quality (≳2048) collapses the FLOP advantage to
≤1.8×; and the realized speedup is a **20–25× slowdown** because the shipped `.cu` kernel is unoptimized. The "7.43×" and
"~1773" are both unsubstantiated. The actuator correctly ships default-OFF and, by its OOD gate, correctly never fires on
real inference. This confirms and quantifies — across the full rank frontier, on the live kernels — the prior g1.3 and
reachability findings.
