# W14 Step 2 E ζ-deep 5-day extension: final closeout + Anil re-adjudication surface

**Date:** 2026-05-23
**Adjudication chain:** 2026-05-23 ζ-deep 3-day → 5-day extension → Path A → α + β plan
**Status:** **5-day box exhausted.** Five substrate fixes shipped. Architectural
            ceiling finding confirmed. Goal 4 v1 narrow-domain Koopman compute
            substitution does NOT deliver top-1 ≥ 99% on real TinyLlama
            workloads. Surfacing for Anil re-adjudication on goal-4 scope.

## 1. Five substrate fixes shipped (real engineering wins)

| Tag                                           | Commit  | Fix                                                                       |
|-----------------------------------------------|---------|---------------------------------------------------------------------------|
| week-14-step-2-h2-cgs2-fix                    | 3590f4d | rand_svd MGS loss-of-orthogonality (CGS2 / Giraud-Langou-Rozloznik 2005)  |
| week-14-step-2-zeta-day3-gpu-upload           | 42b05e9 | CPU-pointer-as-GPU-pointer bug in cipher_koopman_fp16_register_shape      |
| week-14-step-2-zeta-day1                      | cb6ea84 | POWER_ITERS 1→4 (rand_svd partial convergence improvement)                |
| week-14-step-2-zeta-day2-deterministic-svd    | 775abda | cusolver deterministic SVD as default; rand_svd as env-gated fallback     |
| week-14-step-2-zeta-day5-ood-detector         | 352c996 | runtime OOD auto-passthrough detector via residual_ratio check            |

All five committed standalone with their own tags. Each is independently
revertable. Substrate post-all-fixes is mathematically correct: sigmas match
torch SVD, V_T aligns with dominant singular subspace, calibration data
flows correctly through GPU pipeline, OOD detector prevents off-manifold
corruption.

## 2. Day 4 α-matched test (distribution-matched calibration)

Per Anil 5-day Day 4: warmup + measurement both use prompts from same
template distribution (build_prompts function). 22 warmup prompts (offset
1000-1022) × 100 token autoregressive decode; 100 measurement prompts
(offset 1-100) × 100 tokens teacher-forced.

Outcome with substrate at week-14-step-2-zeta-day2-deterministic-svd
(deterministic cuSOLVER SVD), all-shapes substitution enabled, no OOD
detector:
```
  vl shape: (100, 100, 32000)   cl shape: (100, 100, 32000)
  top-1 match rate:  0.00%        (gate ≥ 99%)
  top-5 overlap:     0.00%
  KL mean:           1.28e+01
  Pearson:           -0.063
  koopman handled (measurement): 15500
```

**Decision tree: α.3.C** — even matched-distribution calibration can't
recover the gate. The substrate's rank-64 Koopman doesn't span the
measurement activation manifold even when calibration and measurement
share prompt template distribution.

## 3. Day 5 β OOD detector + threshold sweep

Per Anil 5-day Day 5: implemented `cipher_koopman_fp16_ood_max_residual`
in cipher_block_sub_kernel.cu. Per input row: compute alpha = V_T · x,
then residual_ratio² = 1 - (||alpha||² / ||x||²). cipher_rt_koopman_engine
calls this before launch_shape; if max(residual_ratio) > threshold,
return CIPHER_RT_MATMUL_PASSTHROUGH so substrate runs real cuBLAS.

Threshold sweep results (α-matched workload, 100 prompts × 100 tokens):

| Threshold | Koopman handled | top-1   | KL mean   | Pearson |
|-----------|-----------------|---------|-----------|---------|
| 1.5       | 15500 (100%)    | 0.00%   | 1.07e+01  | +0.080  |
| 1.2       | 15500 (100%)    | 0.00%   | 1.22e+01  | -0.029  |
| 1.0 (off) | 15500 (100%)    | 0.00%   | 1.29e+01  | -0.075  |
| 0.8       | 400 (0.1%)      | 0.00%   | 8.28e+00  | +0.208  |
| 0.5       | 0 (0%)          | **99.95%** | 2.55e-06 | +1.000  |
| 0.3       | 0 (0%)          | **99.95%** | 2.55e-06 | +1.000  |
| 0.1       | 0 (0%)          | **99.95%** | 2.55e-06 | +1.000  |
| 0.05      | 0 (0%)          | **99.95%** | 2.55e-06 | +1.000  |
| 0.01      | 0 (0%)          | **99.95%** | 2.55e-06 | +1.000  |

**Curve has no knee with positive utility.** For any threshold where
Koopman fires non-trivially (≥ 1.0), output is wrong. For any threshold
where output is correct (< 0.5), Koopman doesn't fire at all.

Mechanism: ALL 356,500 measurement cuBLAS calls' inputs have
residual_ratio > 0.5 against the calibration V_x. The measurement
activation manifold is essentially orthogonal to the warmup activation
manifold even within the same prompt template distribution.

## 4. Architectural ceiling characterization

Three composing factors produce the ceiling:

**Factor 1: real LLM activations are NOT rank-64.** Day 1 H4 evidence:
Python ground-truth on real TinyLlama hidden states reports energy_X =
0.6016 at rank-64. 40% of input variance falls outside any rank-64
subspace. The Koopman premise (low-rank activation manifold) holds in
PRINCIPLE but not at the rank-64 budget v1 chose.

**Factor 2: prompt-trajectory diversity exceeds rank-64.** 22 short
prompts × 100 token decode = 2200 distinct activation positions across
22 different trajectories. The rank-64 SVD captures the dominant 64
directions of THIS calibration set; measurement uses 100 DIFFERENT
prompts producing 100 DIFFERENT trajectories. Their activation
manifolds intersect only partially.

**Factor 3: greedy decode is sensitive to logit perturbations.** Even
when residual is 30%, the LM head softmax argmax flips. Top-1 token
match is a strict gate; 30% residual ≈ 0% top-1 (Day 1 D1.3 baseline
confirms 90% top-1 requires residual ≈ 5%).

These three factors are individually known; their composition gives the
ceiling. The β detector measures Factor 1+2 directly via residual_ratio.

## 5. v1 ship options for Anil

**(1) Ship cipher_rt_phase4 at 352c996 with β at default threshold 0.05** —
Honest behavior: Koopman substrate present in libcipher_rt.so; in
production, β detector identifies real inference inputs as out-of-
distribution → cuBLAS runs unchanged → no behavioral difference from
vanilla → no quality regression. Goal 4 v1 quality bar NOT MET; substrate
primitive shipped + auto-passthrough preserves correctness. Pitch claim
revises to "Koopman compute substitution substrate + auto-passthrough
detector; production quality validation pending broader-distribution
calibration".

**(2) Re-scope goal 4 to substrate-only deliverable** — substrate primitive
+ OOD detector ship as the v1 narrow-domain Koopman deliverable. Honest
pitch: "Five substrate fixes including deterministic SVD + OOD detector
ship in v1; Koopman activation requires production-distribution calibration
not yet feasible in v1 scope; substrate is ready for v1.5 calibration
work." S2.G commits the substrate; v1.5 scope-locks the calibration work.

**(3) Continue rank-r research (not v1)** — increase rank, replace rank-r
with different approximation family, or pre-train calibration data from
production traces. ~2-4 weeks of research engineering. Pushes v1.2.3 W14
timeline out by that amount.

**(4) Defer goal 4 entirely to v1.5/v2; substrate fixes ship as bug fixes**
— five committed fixes are real substrate improvements regardless of
Koopman quality. Mark goal 4 as "v1 substrate prepared; v1.5 quality work
required". Cleanest honest framing if the 5-eng-day extension exhausted
the v1 budget without delivering the quality gate.

**Recommended (engineering perspective): (2) Re-scope to substrate + OOD
detector deliverable.** It's the option that honestly ships what was
built, names the v1.5 work explicitly, and avoids retracting fixes that
are real engineering wins. Pitch language adjusts to "Koopman compute
substitution substrate; activation gated by auto-passthrough OOD
detector; production-grade calibration is v1.5."

## 6. What goal 4 v1 ACTUALLY ships at 352c996

  - cipher_rt_koopman_engine: registers as second actuator on the
    cipher_rt_matmul_dispatch substrate (parallel to Marlin)
  - Five-fix substrate: H2 CGS2 + GPU upload + POWER_ITERS=4 +
    deterministic cuSOLVER SVD + OOD detector
  - cipher_edmd_live calibration pipeline: snapshot capture from real
    cublasGemmEx + deterministic SVD + per-shape registry
  - β OOD detector at default 0.05: substrate is correctness-preserving
    by construction (passthrough on out-of-distribution inputs)
  - Tagged + step-documented at every step
  - Memory anchor lists all five fixes for v1.5 reference

What it does NOT ship:
  - Top-1 ≥ 99% on real TinyLlama prompts at production calibration
  - KL ≤ 5.5e-5 (original scope-lock gate; retracted as ungrounded
    earlier; replacement top-1 gate also not met)
  - Frobenius residual < 0.10 (may13's design gate; also not met)
  - Workload-validated calibration distribution model

## 7. Anil surface

5-day box exhausted. Five real substrate fixes shipped. v1 quality gate
NOT met. Substrate is correct + auto-passthrough by default.

Picking v1 framing requires Anil. Recommendation (2). Awaiting
adjudication before S2.G commit (or its replacement architectural-finding
commit).

Substrate baseline at 352c996. cipher_kmod / cipher_vllm_kv.py UNCHANGED.
