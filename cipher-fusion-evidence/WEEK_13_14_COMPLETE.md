# W13-14 COMPLETE — Koopman Tier Integration + G12 Model-Keying

**Date:** 2026-05-24
**Close tag:** `week-13-14-complete`
**Substrate anchor at close:**
- cipher_rt_phase4 `25970f3` aliases `week-14-complete` + `week-14-step-3-remember-validate` + `week-13-14-complete`
- cipher-fusion-evidence S3.E close commit aliases `week-14-complete` + `week-14-step-3-remember-validate` + `week-13-14-complete`
- cipher_kmod `8c643fc` aliases `week-9-complete` + `week-13-14-complete` (no ABI change across W13-14)
- libcipher_rt.so md5 `097cf8d907a7e866a3e3640eb0993003`

## 1. W13-14 substep chain

W13 Step 1 (G12 Koopman registry model-keying) + W14 Step 2 (narrow-domain Koopman substrate) + W14 Step 3 (REMEMBER consumer + LM-head validation + N=128 soak).

| Phase | Tag | Commit |
|---|---|---|
| W13 Step 1 — G12 model-keying | `week-13-step-1-g12-koopman-keying` | cipher_rt_phase4 `d93bf9d` + cipher-fusion-evidence prior |
| W14 Step 2 — Koopman tier (G CLOSURE) | `week-14-step-2-koopman-tier` (alias `week-14-step-2-tikhonov`) | cipher_rt_phase4 `4b775c8` + cipher-fusion-evidence `bf4f0ad` |
| W14 Step 2 addendum (slot-3 producer residue) | (no tag) | cipher-fusion-evidence `6be1d4f` |
| W14 Step 3 S3.B0 — slot-3 RING_WRITE producer | `week-14-step-3-b0-koopman-producer` | cipher_rt_phase4 `4abb138` + cipher-fusion-evidence `7154d5b` |
| W14 Step 3 S3.B1 — REMEMBER consumer drain | `week-14-step-3-b1-remember-consumer` | cipher_rt_phase4 `70f389b` + cipher-fusion-evidence `cfaf554` |
| W14 Step 3 S3.C KL gate addendum (two-mode α) | (no tag — addendum) | cipher-fusion-evidence `ef8b830` |
| W14 Step 3 S3.B1.1 — multi-slot drain housekeeping | `week-14-step-3-b1-1-multi-slot-drain` | cipher_rt_phase4 `68cc9c3` + cipher-fusion-evidence `dabafc3` |
| W14 Step 3 S3.C — LM-head validation harness | `week-14-step-3-c-lmhead-validate` | cipher_rt_phase4 `25970f3` + cipher-fusion-evidence `fc07e7a` |
| W14 Step 3 S3.D — regression + N=128 soak both modes | `week-14-step-3-d-soak` | cipher-fusion-evidence `cb4a5c3` |
| W14 Step 3 S3.E — close-out | `week-14-step-3-remember-validate` alias `week-14-complete` | cipher-fusion-evidence (this chain) |
| W13-14 close | `week-13-14-complete` | both trees (this commit) |

## 2. Plan §7 W13-14 row

`CIPHER_REENGINEERING_PLAN.md:1260`:

> | **13-14** | Koopman tier integration (EDMD pipeline real-input fix + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation) + **G12 Koopman registry model-keying** *(held from W11-12 to W13-14 per B.1)* | v1.2.3 |

→ Marked **DONE** in the same close commit chain.

## 3. What ships under W13-14

### G12 model-keying (W13 Step 1)

`cipher_rt_recipe_model_key(uint32_t shape_hash, uint64_t model_uuid_lo, uint64_t model_uuid_hi)` at `cipher_recipes.cpp:557-575`. XOR-mix pattern mirrors G3 KV-dedup keying at `cipher_rt_kv_alloc.c:705-707`. MODEL_UNKNOWN sentinel preserves pre-G12 single-tenant compat. Header surface at `cipher_recipes.h:202-204`.

### Koopman tier substrate (W14 Step 2)

Six substrate fixes shipped (per `WEEK_14_STEP_2_KOOPMAN_TIER.md`):

1. **H2 CGS2 fix** — `cipher_randsvd.h` twice-MGS at 3 sites resolves loss-of-orthogonality
2. **GPU pointer upload fix** — `cipher_edmd_live.cpp` cudaMalloc + cudaMemcpy replaces CPU buffer pass to GPU kernel
3. **POWER_ITERS=2** — randomized SVD power iteration count tuned
4. **Deterministic cuSOLVER SVD** — `det_svd_rank_r()` replaces `cipher_rs::rand_svd_rank_r` for reproducibility
5. **β OOD detector** — `cipher_koopman_fp16_ood_max_residual()` GPU kernel + runtime gate in `cipher_rt_koopman_engine.cpp:91-111`
6. **Tikhonov regularization** — α=0.01·σ₁² spectral filter per plan §1 line 103 formula

Koopman engine registered as second actuator on the T4.5.1 matmul-dispatch substrate, parallel to Marlin. Env-gated `CIPHER_KOOPMAN=1`.

### Producer + consumer + validation (W14 Step 3)

- **Slot-3 RING_WRITE producer** (S3.B0): weak-linked emission on each HANDLED Koopman substitution; payload `cipher_rt_koopman_event { M, K_dim, N_dim, op_class, ptr_B, ptr_C, reserved }` 40 B; p99 publish 82 ns
- **REMEMBER consumer drain** (S3.B1 + S3.B1.1): pthread thread iterates 128 tenants per cycle, drains slot 2 (REMEMBER) into `cipher_lnn_decide()` at `cipher_lnn.cpp:401-460`; CfC hidden state `h` updated on each drained event; multi-slot housekeeping drain of slots 0+1 to prevent producer overflow on idle CLASSIFY/ORACLE slots
- **LM-head validation harness** (S3.C): Python full-substrate harness on real TinyLlama LM head; two-pass gate (fire mode SUBSTITUTION top-1 ≥ 0.90; passthrough mode PRESERVATION KL ≤ 5.5e-5); both PASS at the integrated stack
- **KL gate addendum** (S3.C addendum): two-mode interpretation of scope-lock §4 Step 3 clauses 1+2; restores v1 contract per plan §1 line 103 + W14 Step 2 G operational close framing
- **N=128 30-min soak** (S3.D): both Mode A (consumer dormant) and Mode B (consumer active) ALL GATES PASS; consumer-active overhead 4.3% slower aggregate

## 4. v1 product contract delivered

### Goal 4 narrow-domain Koopman (plan §1 line 103)

- EDMD pipeline with snapshot-feeder fix: SHIPPED
- Per-layer rank-parameterized recovery (rank-64): SHIPPED
- Tikhonov regularization α=0.01·σ₁²: SHIPPED
- Spectral radius ≤ 1 enforcement: via Tikhonov + deterministic SVD
- SUBSTITUTE-Koopman lane validation: SHIPPED end-to-end on real TinyLlama LM head; substrate fires Koopman exactly as designed (β-gated path selection)

### Substrate quality contracts

| Contract | Path | Measurement | Result |
|---|---|---|---|
| EXISTENCE (fire mode) | β residual < threshold | top-1 vs vanilla cuBLAS | **0.9000** at rank-64 on TinyLlama LM head |
| PRESERVATION (passthrough mode) | β residual ≥ threshold | KL vs vanilla cuBLAS | **0.000000e+00** (byte-identical) |

## 5. Honest residue — what we did NOT measure

Per `WEEK_14_STEP_3_REMEMBER_VALIDATE.md` §3:

The W14 Step 3 deliverable is a substrate, not a measured product-feature uplift on a real customer workload. The harness fired Koopman exactly once on a single explicit gemm with a harness-raised β threshold. The substrate ships with β default 0.05 which keeps Koopman from firing on real LM head residuals (~0.6) cross-distribution. The aggregate tok/s impact of Koopman compute substitution on a real vLLM decode workload is NOT measured at W13-14 close.

The next campaign (memory `w14-step-3-followup-mistral-tok-s`) is queued to measure this: real-workload Mistral-7B (or TinyLlama N=4 first pass) + vLLM tok/s with Koopman threshold raised vs off. Anil adjudication 2026-05-24: "see once we close this we will work on option 2."

This residue is honest framing, not scope-down. The v1 contract per plan §1 line 103 is what ships at W13-14; whether the v1 contract translates to measured customer-workload uplift is a separate measurement campaign that follows.

## 6. Substrate state at close

| Tree | Commit | Tag aliases |
|---|---|---|
| cipher_rt_phase4 | `25970f3` | `week-14-step-3-c-lmhead-validate`, `week-14-step-3-remember-validate`, `week-14-complete`, `week-13-14-complete` |
| cipher-fusion-evidence | this close commit | `week-14-step-3-remember-validate`, `week-14-complete`, `week-13-14-complete` |
| cipher_kmod | `8c643fc` | `week-9-complete`, `week-9-step-5-n128-soak`, `week-13-14-complete` |
| libcipher_rt.so | md5 `097cf8d907a7e866a3e3640eb0993003` | — |
| cipher_kv_bridge.so | (W12 Step 3 anchor) | — |

## 7. Next phase

Per `CIPHER_REENGINEERING_PLAN.md:1246`:

> | **15-17** | hybrid workload-class + model-architecture heterogeneous benchmark on full unified runtime |

CP 5.5 follows the queued option-2 measurement campaign. Substrate at `week-13-14-complete` is the entry anchor for both.
