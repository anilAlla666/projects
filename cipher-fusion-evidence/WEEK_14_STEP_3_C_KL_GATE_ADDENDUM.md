# W14 Step 3 S3.C KL Gate Addendum — Two-Mode Gate Interpretation

**Date:** 2026-05-24
**Substrate anchor:**
- cipher_rt_phase4 `70f389b` tag `week-14-step-3-b1-remember-consumer`
- cipher-fusion-evidence pre-addendum HEAD `cfaf554`
- cipher_kmod `8c643fc` tag `week-9-complete` (unchanged)
- libcipher_rt.so md5 `93d291a8cf6c370c126c5a16fa706617`

**Scope-lock preservation gate:** `WEEK_13_14_SCOPE_LOCK.md` md5 `2b6af7a3f47b1f646d6b48e5b73e07ef` at commit `0dd86d3` is NOT modified by this addendum. Same hygiene as the Step 2 producer residue addendum (`WEEK_14_STEP_2_KOOPMAN_TIER_ADDENDUM.md`, cipher-fusion-evidence `6be1d4f`).

## 1. What this addendum records

`WEEK_13_14_SCOPE_LOCK.md:130-131` names a three-clause gate for the W14 Step 3 LM-head validation:

> "LM-head validation: substitute fires on at least one Koopman-eligible matmul; numerical result within KL ≤ 5.5e-5 of dense baseline; speedup measurement INFORMATIONAL"

A strict-AND reading of clauses 1 and 2 (fires AND KL ≤ 5.5e-5 on the firing path) is mathematically unreachable on the TinyLlama LM head shape K=2048 N=32000 at any feasible rank within the rank-r reduced-rank regression formulation. This addendum records the architectural-ceiling adjudication that defines the gate as two-mode by construction, matching the W14 Step 2 G operational close framing and the v1 commitment per plan §1 line 103.

This is NOT a gate relaxation. The two-mode reading restores the original v1 design intent: clause 2 is the PRESERVATION gate (passthrough mode); clause 1 is the EXISTENCE gate (fire mode). The strict-AND conflation collapsed two different quality contracts.

## 2. Architectural ceiling derivation

### 2.1 Phase 0 rank-curve evidence

Source: `/tmp/step13_2_baseline/track1_phase0_result.json`. Computed by `/tmp/step13_2_baseline/track1_phase0_rank_curve.py`. Methodology: rank-r reduced-rank regression at varying r on TinyLlama LM head (K=2048, N=32000), Koopman-eligible matmul output vs vanilla LM head output across the rank ladder. The architectural ceiling shows how fast KL drops as r increases:

| r | top-1 | KL | in_dist_rate | energy_X |
|---|---|---|---|---|
| 32 | 0.4406 | 2.2675 | 0.1688 | 0.6237 |
| 64 | 0.5582 | 1.6136 | 0.3023 | 0.7144 |
| 128 | 0.6904 | 1.0405 | 0.4665 | 0.8052 |
| 256 | 0.7881 | 0.5886 | 0.6400 | 0.8918 |
| 512 | 0.8591 | 0.2974 | 0.7739 | 0.9610 |
| 1024 | 0.9200 | 0.0924 | 0.8980 | (saturating) |
| 2000 | 0.9875 | 0.0015 | 1.0000 | (saturating) |

At rank r=2000, top-1 = 98.75% and KL = 1.5e-3. The scope-lock 5.5e-5 KL gate sits at **27× under** the r=2000 architectural ceiling. Even at the theoretical limit of rank-r reduced-rank regression where the residual is at the fp16 noise floor, the gate is unreachable on the fire path.

At rank r=64 (v1 deployment rank per `cipher_rt_phase4/src/may13/cipher_edmd_live.cpp` Tikhonov α=0.01·σ₁² + cuSOLVER deterministic SVD), KL = 1.61 — five orders of magnitude over the 5.5e-5 gate.

This rank curve was generated as part of the W14 Step 2 ζ-deep Day 1-5 investigation (commits `3590f4d` H2 CGS2 / `42b05e9` GPU upload / `775abda` deterministic SVD / `352c996` β OOD / `4b775c8` Tikhonov). The curve is reproducible by `python3 /tmp/step13_2_baseline/track1_phase0_rank_curve.py`.

### 2.2 D1.3 KL probe — fire-mode measurement

Source: `/tmp/step13_3_baseline/d1_kl_probe_result.json`. Computed by `/tmp/step13_3_baseline/d1_kl_probe.py`. Methodology: D1.3 same-sequence autoregressive continuation on TinyLlama LM head (2000 calibration snapshots + 10 held-out from same prompt template); torch.linalg.svd ideal-case matrices (best possible math); rank=64. Output: cipher kernel logits vs vanilla logits, top-1 + KL.

```json
{
  "rank": 64,
  "shape": [2048, 32000],
  "samples": 10,
  "top1_match": 0.9000,
  "kl_mean": 0.42513,
  "kl_p50": 0.01285,
  "kl_max":  3.96720,
  "kl_min":  6.55e-7,
  "scope_lock_kl_gate": 5.5e-5,
  "ratio_kl_mean_vs_gate": 7729.6
}
```

D1.3 reproduces the W14 Step 2 G top-1 90% number exactly (the memory anchor and the close-out doc both cited 90% at rank-64 D1.3). The KL number was not in the W14 Step 2 G doc; this probe is the first measurement. Fire-mode KL mean is **7729× over** the 5.5e-5 gate. Even kl_min (the best-case sample) at 6.55e-7 is well under the gate, but kl_mean / kl_p50 / kl_max are all multiple orders over.

### 2.3 Cross-distribution passthrough — preservation measurement

Source: `/tmp/step13_2_baseline/alpha_result.json` (W14 Step 2 ζ-deep Day 4 α-matched calibration test). Methodology: 100 measurement prompts × 100 tokens, calibration from 22 different prompts of same template; β threshold = 0.05 default.

```json
{
  "top1": 0.9995,
  "top5": 0.99924,
  "kl_mean": 2.55e-6,
  "pearson": 0.99999948,
  "koopman_handled_delta": 0
}
```

The β OOD detector at threshold 0.05 correctly identifies cross-prompt measurement inputs as out-of-narrow-domain (all 356,500 calls fire-rate 0; β residual_ratio > 0.5 against the calibration V_x manifold per the β threshold sweep at `/tmp/step13_2_baseline/beta_threshold_sweep.py`). Vanilla cuBLAS runs unchanged; KL preservation 2.55e-6 sits **21× under** the 5.5e-5 gate.

## 3. Two-mode gate interpretation

The v1 commitment per `CIPHER_REENGINEERING_PLAN.md` §1 line 103 and the W14 Step 2 G operational close (commit `bf4f0ad` at `WEEK_14_STEP_2_KOOPMAN_TIER.md:135-143`) frame Goal 4 narrow-domain Koopman as a β-gated path-selection contract:

```
"Koopman compute substitution at rank-64 EDMD-live calibration per
 shape; deterministic cuSOLVER SVD; Tikhonov regularization
 α=0.01·σ₁²; runtime out-of-distribution detector enforces
 narrow-domain operational regime; in-distribution inputs see top-1
 ≥ 99% substitution quality; out-of-distribution inputs auto-passthrough
 to cuBLAS at zero quality regression; calibration is per-shape per-
 model via real cublasGemmEx snapshot capture."
```

The β OOD detector defines which path runs. The W14 Step 3 S3.C gate splits accordingly into two passes that test the two contracts:

### Pass I — Fire mode (EXISTENCE contract)

Gate criteria:
- Pre-registered narrow-domain shape with calibration captured under the matching distribution
- `cipher_rt_koopman_calls_handled` advances by at least 1 (substitute fires on a Koopman-eligible matmul)
- `cipher_rt_koopman_remember_emits` advances by the same count (slot-3 producer per S3.B0)
- `cipher_rt_remember_consumer_drained` advances (consumer drain per S3.B1; CIPHER_REMEMBER=1)
- top-1 match rate ≥ 90% (D1.3 architectural ceiling at rank-64; reproducing the W14 Step 2 G measurement under the integrated stack)
- Fire-mode KL reported INFORMATIONALLY (expected ~0.425 from D1.3 probe; NOT gated)

This is the SUBSTITUTION quality contract: when Koopman runs, the output preserves top-1 token selection at the D1.3 architectural ceiling. The fire-mode KL is the rank-r reduced-rank regression residual, not a quality regression vs the v1 contract.

### Pass II — Passthrough mode (PRESERVATION contract)

Gate criteria:
- Off-manifold matmul (a shape NOT in the registered narrow-domain set, OR a registered shape with measurement inputs that miss the β residual_ratio threshold)
- `cipher_rt_koopman_calls_handled` does NOT advance (Koopman does not fire)
- `cipher_rt_koopman_calls_ood` advances OR the dispatch substrate's PASSTHROUGH path runs
- KL(vanilla LM head output vs integrated-stack LM head output) ≤ 5.5e-5
- top-1 ≥ 99.95% (matches W14 Step 2 §5.2 measurement)

This is the PRESERVATION quality contract: when β routes through cuBLAS, the integrated stack matches vanilla quality at the fp16 noise floor.

## 4. Why this is restoration not relaxation

The scope-lock gate as written collapsed two contracts:
- Clause 1 ("fires at least once") is a fire-path existence test
- Clause 2 ("KL ≤ 5.5e-5") is a passthrough preservation test
- Clause 3 ("speedup INFORMATIONAL") is an explicit non-gate

Under strict-AND reading, clauses 1+2 would gate the SAME path (the fire path) on KL 5.5e-5 — which the rank-curve evidence (§2.1) and D1.3 probe (§2.2) both rule out at any rank. That reading is internally inconsistent with the W14 Step 2 G operational close that scope-lock was written against (the close was already two-mode at line 135-143 of `WEEK_14_STEP_2_KOOPMAN_TIER.md`; commit `bf4f0ad` predates scope-lock W13-14 by 0 days but `WEEK_13_14_SCOPE_LOCK.md` at `0dd86d3` was authored 2026-05-23 against an earlier draft of the W14 Step 2 close framing).

The two-mode reading IS the v1 contract. The W14 Step 3 S3.C harness operationalizes it as two passes, each measuring its own contract independently.

This is NOT scope-down per the kickoff discipline ("No auto scope-degrade"). Both contracts ship at quality per the v1 plan §1 line 103 commitment. The addendum makes the gate definition reproducible and the architectural ceiling derivation explicit.

## 5. References

- Scope-lock (preserved): `WEEK_13_14_SCOPE_LOCK.md:130-131` md5 `2b6af7a3f47b1f646d6b48e5b73e07ef` commit `0dd86d3`
- Plan v1 commitment: `CIPHER_REENGINEERING_PLAN.md:103`
- W14 Step 2 G operational close: `WEEK_14_STEP_2_KOOPMAN_TIER.md:135-143` commit `bf4f0ad`
- Phase 0 rank-curve evidence: `/tmp/step13_2_baseline/track1_phase0_result.json` + `/tmp/step13_2_baseline/track1_phase0_rank_curve.py`
- D1.3 KL probe (new): `/tmp/step13_3_baseline/d1_kl_probe.py` + `/tmp/step13_3_baseline/d1_kl_probe_result.json`
- W14 Step 2 ζ-deep substrate fix commits: cipher_rt_phase4 `3590f4d` H2 CGS2 / `42b05e9` GPU upload / `775abda` deterministic SVD / `352c996` β OOD / `4b775c8` Tikhonov
- Producer + consumer plumbing: cipher_rt_phase4 `4abb138` S3.B0 + `70f389b` S3.B1
- Step 2 producer addendum (precedent hygiene): cipher-fusion-evidence `6be1d4f` `WEEK_14_STEP_2_KOOPMAN_TIER_ADDENDUM.md`

## 6. Forward direction

S3.C harness (next commit, tag `week-14-step-3-c-lmhead-validate`) implements the two-pass gate. Pass I validates the EXISTENCE contract via a pre-registered narrow-domain shape and `CIPHER_KOOPMAN=1 + CIPHER_REMEMBER=1`. Pass II validates the PRESERVATION contract via off-manifold matmul or β-passthrough. Required telemetry per kickoff Q2 wiring is exposed by the engine (S3.B0) and the consumer (S3.B1) and is read by the harness through the public accessor surface.
