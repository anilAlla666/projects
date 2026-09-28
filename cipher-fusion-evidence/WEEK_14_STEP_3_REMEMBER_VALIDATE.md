# W14 Step 3 — REMEMBER Consumer + LM-Head Validation Close-out (S3.E)

**Date:** 2026-05-24
**Close tag:** `week-14-step-3-remember-validate` (alias `week-14-complete`)
**Substrate anchor at close:**
- cipher_rt_phase4 `25970f3` tag `week-14-step-3-c-lmhead-validate` + new alias `week-14-step-3-remember-validate` + `week-14-complete`
- cipher-fusion-evidence `cb4a5c3` tag `week-14-step-3-d-soak` + new alias `week-14-step-3-remember-validate` + `week-14-complete`
- cipher_kmod `8c643fc` tag `week-9-complete` (unchanged across W14 Step 3 — no ABI change)
- libcipher_rt.so md5 `097cf8d907a7e866a3e3640eb0993003`

## 1. Substep summary

| Substep | Deliverable | cipher_rt_phase4 commit | cipher-fusion-evidence commit | Tag |
|---|---|---|---|---|
| Step 2 addendum | slot-3 RING_WRITE producer residue record (own commit, no tag) | (no code) | `6be1d4f` | (no tag) |
| S3.A | pre-flight + `/tmp/step13_3_baseline/` baseline preserve | (no code) | (verification only) | (no tag) |
| S3.B0 | slot-3 RING_WRITE producer in Koopman engine | `4abb138` | `7154d5b` | `week-14-step-3-b0-koopman-producer` |
| S3.B1 | REMEMBER consumer drain into CfC LNN forward | `70f389b` | `cfaf554` | `week-14-step-3-b1-remember-consumer` |
| S3.C KL gate addendum | two-mode gate interpretation (option α adjudication) | (no code) | `ef8b830` | (no tag — addendum is the record) |
| S3.B1.1 | REMEMBER consumer multi-slot drain housekeeping | `68cc9c3` | `dabafc3` | `week-14-step-3-b1-1-multi-slot-drain` |
| S3.C | LM-head validation harness (two-pass full-substrate) | `25970f3` | `fc07e7a` | `week-14-step-3-c-lmhead-validate` |
| S3.D | regression + 30-min N=128 soak (both modes) | (no code) | `cb4a5c3` | `week-14-step-3-d-soak` |
| S3.E | close-out (this doc) | (no code) | this commit | `week-14-step-3-remember-validate` alias `week-14-complete` |

## 2. v1 contract — what shipped under the integrated stack

Per `CIPHER_REENGINEERING_PLAN.md:103` (Goal 4 narrow-domain Koopman v1 commitment) and `WEEK_14_STEP_3_C_KL_GATE_ADDENDUM.md` (two-mode gate interpretation):

### Fire mode (EXISTENCE contract)

Substrate measurement on TinyLlama-1.1B LM head shape (K=2048, N=32000), rank-64, deterministic cuSOLVER SVD calibration, β OOD detector at harness-only threshold 0.7:

- `cipher_rt_koopman_calls_handled`: +1
- `cipher_rt_koopman_remember_emits`: +1 (1:1 producer)
- `cipher_rt_remember_consumer_drained`: +8 (consumer drained the Pass I REMEMBER event plus housekeeping entries)
- `cipher_rt_remember_consumer_lnn_invocations`: +1 (CfC forward fired once)
- top-1 match rate vs vanilla cuBLAS = **0.9000** (reproduces D1.3 architectural ceiling under integrated stack)
- Fire-mode KL mean = 0.4251 (informational, matches D1.3 KL probe `kl_mean = 0.42513` exactly)

### Passthrough mode (PRESERVATION contract)

Off-manifold input (residual_ratio 0.989 against the registered V_x manifold):

- `cipher_rt_koopman_calls_handled`: +0 (β fired PASSTHROUGH)
- top-1 match rate vs vanilla cuBLAS = **1.0000**
- KL mean = **0.000000e+00** (byte-identical via PASSTHROUGH path through real cublasGemmEx)

Substrate exit during the harness run: **317,755** cuBLAS GEMM calls intercepted, 1 substituted, 317,754 PASSTHROUGH. The substrate behaves exactly as designed: β at default 0.05 keeps Koopman from firing on cross-distribution real LM head inputs; raising to 0.7 in the harness widens the in-distribution band so the EXISTENCE gate can be exercised on real TinyLlama data.

## 3. v1 honest framing — what we did NOT prove

The W14 Step 3 deliverable is a **substrate**, not a measured product-feature uplift on a real customer workload. Specifically:

- The harness fired Koopman **exactly once** by construction (one explicit gemm on a pre-calibrated registered shape with a harness-raised β threshold)
- The substrate ships with **β default 0.05** which the W14 Step 2 G §5.3 OOD sweep showed keeps Koopman from firing on real LM head residuals (~0.6) cross-distribution
- The aggregate tok/s impact of Koopman compute substitution on a real vLLM decode workload is **NOT** measured in this step

The honest v1 product framing: "registry seeded, substitution lane wired and validated at architectural ceiling, fire-path verified end-to-end on real LM head data under the integrated stack; production firing on real customer workloads is a measurement that follows in a separate campaign." This avoids the scope-drift pattern catalogued in `WEEK_12_SCOPE_DRIFT_AUDIT.md` (primitive-ships-consumer-defers).

## 4. Substrate stability under load

S3.D 30-min N=128 soak in both modes:

| Gate | Mode A (consumer dormant) | Mode B (consumer active) | Pass? |
|---|---|---|---|
| Atomicity | 128000/128000 | 128000/128000 | Both PASS |
| Writers-only fairness min | 0.879 | 0.892 | Both PASS (>= 0.85) |
| Coherence (incoherent reads) | 0 | 0 | Both PASS |
| Aggregate vs 10.66 M/s | +10.0% | +5.3% | Both PASS (within ±10%) |

Mode B consumer-active overhead = -4.3% aggregate throughput vs Mode A consumer-dormant. Fairness in Mode B is slightly TIGHTER (consumer's multi-slot housekeeping evens out producer overflow distribution).

## 5. Latent bug surfaced + closed

The S3.B1 consumer (slot 2 drain only) had a latent producer-overflow interaction: with v1 having no production CLASSIFY (slot 0) or ORACLE (slot 1) drainer, `min_read_seq()` stayed at 0 across all tenants; producer's overflow check at `cipher_rt_ring_write.c:198` dropped emits after 4096 per tenant including the Pass I REMEMBER event. Diagnosed during S3.C build; fixed at S3.B1.1 by extending the REMEMBER consumer to drain CLASSIFY + ORACLE slots via drain-and-discard (entries not processed; only `read_seq[0]` and `read_seq[1]` advance to track `write_seq`). v1-only patch; v2 will introduce per-slot consumer registration so the producer's overflow check considers only ACTIVE consumers.

Effect on S3.B1 self-test COMPOSE: drop_pct 3.12% → 0.00% (32000 of 32000 drained vs 31000 before fix). No effect on regression suite or N=128 soak gates.

## 6. Scope-lock check

`WEEK_13_14_SCOPE_LOCK.md` (md5 `2b6af7a3f47b1f646d6b48e5b73e07ef`, commit `0dd86d3`) is preserved unmodified. The Step 3 sub-element table at lines 109-138 maps to actual deliverables:

| Scope-lock sub-element | Status | Where landed |
|---|---|---|
| `cipher_rt_remember_consumer.{c,h}` ~150 LOC | SHIPPED 250 LOC | S3.B1 `cipher_rt_remember_consumer.cpp` + .h |
| `cipher_inject.c` REMEMBER thread spawn env-gated | SHIPPED | S3.B1 |
| LM-head substitution validation harness ~150 LOC | SHIPPED 280 LOC | S3.C `test_step3_c_lmhead_validate.py` + `.cpp` synthetic smoke |
| N=128 30-min regression soak | SHIPPED both modes | S3.D |
| Slot-3 RING_WRITE producer (Step 2 residue) | SHIPPED | S3.B0 |
| Multi-slot drain housekeeping (latent bug fix) | SHIPPED | S3.B1.1 |
| KL gate addendum (two-mode interpretation) | SHIPPED | S3.C addendum |

Verification gate from scope-lock line 131-133:
- Atomicity 128000/128000 — PASS both modes
- Writers-only fairness ≥ 0.85 — PASS both modes (0.879, 0.892)
- 0 incoherent reads — PASS both modes
- Aggregate within ±10% of W12 Step 3 baseline 10.66 M/s — PASS both modes (+10.0%, +5.3%)
- CFL throttle telemetry — gate documented; substrate harness (cipher_test_commit_n128) measures COMMIT primitive not RING_WRITE; CFL throttle exercised separately by S3.B0 microbench (0 throttled) and S3.B1 compose test (drop_pct 0.00% after S3.B1.1)
- LM-head validation KL ≤ 5.5e-5 — re-interpreted per two-mode addendum: PASS for passthrough mode (KL 0.0e+00); fire-mode KL informational (0.425 reproduces D1.3 ceiling)
- W7-12 microbench regression all PASS — 9/11 PASS unchanged from S3.B0 baseline (2 pre-existing carry-forwards)

## 7. Stop-condition check (end of W14 Step 3)

- in-dist 90% top-1 r=64 D1.3: REPRODUCED at 0.9000 under integrated stack
- cross-dist 99.95% passthrough: REPRODUCED at 1.0000
- β detector misfires: NONE
- Regression suite: 9/11 PASS unchanged
- N=128 30-min soak: ALL GATES PASS in both modes

No stop conditions triggered. All five W14 Step 2 numbers preserved at S3.E close.

## 8. What ships at `week-14-complete`

Substrate primitives shipped, end-to-end producer → consumer → CfC LNN forward pipeline validated on real TinyLlama LM head under GOT-patched cuBLAS interception. Goal 4 v1 commitment per plan §1 line 103 lands with the architecturally-honest two-mode interpretation: SUBSTITUTION contract validated at architectural ceiling (top-1 ≥ 90% at rank-64 on TinyLlama LM head); PRESERVATION contract validated byte-identically (KL = 0 via β PASSTHROUGH).

Substrate exits W14 Step 3 with:
- cipher_rt_phase4 `25970f3` (alias `week-14-complete`)
- cipher-fusion-evidence S3.E close commit (alias `week-14-complete`)
- cipher_kmod `8c643fc` unchanged (week-9-complete)
- libcipher_rt.so md5 `097cf8d907a7e866a3e3640eb0993003`
- Producer p99 publish cadence 82 ns
- Consumer drain rate effective (S3.B1 compose 32k of 32k drained, 0% drop)
- N=128 30-min soak both modes ALL GATES PASS

## 9. Forward — W13-14 close + queued follow-up

W13-14 closes here. Plan §7 line 1260 ("W13-14 Koopman tier integration + G12 Koopman registry model-keying") marked DONE in the same close commit chain.

Next campaign queued (memory `w14-step-3-followup-mistral-tok-s`): real-workload Mistral-7B (or TinyLlama N=4 first pass) + vLLM tok/s measurement with Koopman threshold raised to fire on real inputs vs Koopman off. This measures whether Goal 4 delivers on a real customer workload or whether v1 ships as substrate-deliverable. Anil adjudication 2026-05-24: "see once we close this we will work on option 2."

W15-17 (CP 5.5 hybrid workload-class + heterogeneous-model headline benchmark) follows the option-2 campaign per the v1.2.3 plan §7 W15-17 sequencing.
