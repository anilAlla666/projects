# W14 Step 2 B.1: EDMD real-input verification + conflation finding

**Date:** 2026-05-23
**v1.2.3 §7 W13-14:** Step 2 sub-step B.1 of 7 (revised post adjudication)
**Prior tag:** `week-14-step-2-port-landmark` (cipher_rt_phase4 ce4c1b8)
**This step:** verification only — no cipher_rt_phase4 source edit
**Successor:** Step 2 B.2 — cipher_rt_koopman_engine wire-up

## 1. Conflation finding (advisor 2026-05-23)

`WEEK_13_14_SCOPE_LOCK.md` Step 2 sub-element table line 95 names "EDMD pipeline
real-input snapshot-feeder fix" at ~200 LOC, citing plan §1 line 103 and
`include/may13/cipher_edmd.h`. Plan §4 line 401 is the load-bearing source
the scope-lock dereferences for "the bug."

Plan §4 line 401 reads (excerpted from `CIPHER_REENGINEERING_PLAN.md`):

> The registry is seeded with mixed 32+ entries (cipher_recipes.cpp:346)
> spanning gemm / Chebyshev / HyperFlux / A100 / shape-parametric categories;
> no real workload's M/N/K hash matches them; therefore apply_recipe() and
> the Koopman substitution branch never ran at the audit-time substrate.
> This is the central engineering reality the unified runtime must address.

**Line 401 explicitly names `cipher_recipes.cpp:346` as the bug location.** That
file/line is the W14 Step 2 C scope (narrow-domain recipe seeding). There is
no separate EDMD-pipeline code bug at the location the scope-lock implied.

Reading `cipher_edmd.cpp` end-to-end at the W14 Step 2 port-landmark substrate:
zero `FIXME`/`TODO`/`BUG`/`XXX`/`HACK` markers. The pipeline implementation is
mathematically correct as ported. Chebyshev observable continuity matches
may13's `tests/test_layer3_edmd.cpp` known values exactly (Case 1 below).

The W14 Step 2 B.0 port-landmark resolved the weak symbol
`cipher_edmd_live_collect` to its strong definition. The pre-existing
`edmd_live_post_relaunch_hook` at `cipher_rt_phase4/src/may13/cipher_dispatch.cpp:213`
fires real GEMM `ptr_A/ptr_B/ptr_C` + dtypes into the live snapshot path on
every passthrough (fallback-relaunch branch — taken because
`cipher_get_edmd_pipeline` stays weak/NULL, the Koopman-converged branch is
skipped, and the fallback ALSO calls the hook unconditionally). The "real-input
snapshot-feeder wire" is therefore complete with zero rt_phase4 source change.

**Conclusion:** S2.B.1 collapses into verification. No code edit. The "fix"
half of the scope-lock item duplicates S2.C registry-seeding scope; the
"snapshot-feeder wire" half is delivered by the port-landmark alone.

## 2. Verification test

`/tmp/step13_2_baseline/test_edmd_realinput_verify.c` (175 LOC). Builds against
the W14 Step 2 port-landmark substrate (`libcipher_rt.so` md5
`f6762245bff158a58bcbee1ff7e44193`). Four cases, 17 individual assertions.

### 2.1 Case 1 — Chebyshev observable continuity (regression sentinel)

Verifies the ported `cipher_cheb_eval` produces the same values as may13's
fixture in `cipher-may13-evidence/tests/test_layer3_edmd.cpp` lines 53-61.

```
PASS  T_0(0.5) = 1.0
PASS  T_1(0.5) = 0.5
PASS  T_2(0.5) = -0.5
PASS  T_3(0.5) = -1.0
PASS  T_10(1.0) = 1.0
```

5/5 PASS. The port did not perturb Chebyshev arithmetic.

### 2.2 Case 2 — EDMD on linear-map known-recoverable

`y = A·x` with `A` a 4×4 block-diagonal rotation; `K = N+1 = 5` observables
(constant + N linear terms) suffices for exact recovery. Drives 420 snapshots
through `cipher_edmd_collect`; expects `SOLVED` state and fit_error < 0.10
(linear in observable space → exactly recoverable).

```
PASS  linear-map pipeline SOLVED
PASS  Koopman operator marked valid
PASS  fit_error < 0.10 (linear exactly recoverable)
      fit_error = 0.001331  (K=5  m=400)
```

3/3 PASS. fit_error = 0.0013 (75× under the 0.10 gate). EDMD solver, Tikhonov
regularization, and back-substitution work correctly post-port.

### 2.3 Case 3 — LM-head-class smooth-nonlinear snapshots

Realistic distribution: 8-dim input vectors with skewed gaussian-like prior
(sum-of-three-uniforms scaled to `[-1.5, 1.5]`); target is RMSNorm followed by
elementwise `tanh` — both in the Barron class (integrable Fourier moments).
Drives 450 snapshots; expects `SOLVED` and fit_error < 0.50.

```
PASS  lm-head-class pipeline SOLVED
PASS  Koopman operator marked valid
PASS  fit_error < 0.50 (Koopman accepted)
      fit_error = 0.247602  (K=14  m=400)
PASS  cipher_edmd_predict returns true on SOLVED pipeline
      prediction error vs ground truth = 0.5005
PASS  prediction error finite + bounded
```

5/5 PASS. fit_error 0.2476 — also under `CIPHER_KR_FIT_THRESHOLD` (0.30) that
`cipher_koopman_runtime.cpp` uses to gate runtime Koopman registry acceptance,
so this pipeline would be accepted into the runtime Koopman tier under real
calibration conditions. Prediction round-trip is finite and bounded (no NaN/Inf).

### 2.4 Case 4 — post-port symbol resolution

Verifies the 5-source port resolved the expected weak/missing symbols.

```
PASS  cipher_edmd_live_collect strong (was weak NULL pre-port)
PASS  cipher_koopman_fp16_register_shape resolved (from .cu)
PASS  cipher_lnn_init resolved
PASS  cipher_kr_init resolved
```

4/4 PASS.

### 2.5 Aggregate

```
test_edmd_realinput_verify: 17 PASS / 0 FAIL
```

## 3. What S2.B.1 does NOT cover (handed forward)

- **Live GEMM round-trip with real `cudaMemcpyAsync` capture** — Case 3 uses
  CPU-side synthetic snapshots, not GPU pointers driven through
  `cipher_edmd_live_collect`. End-to-end verification requires either a real
  cuBLAS workload or the new `cipher_rt_koopman_engine` wire-up — both belong
  to W14 Step 2 B.2 / S2.E gates.
- **Koopman compute substitution at hot path** — Case 3 demonstrates the
  *pipeline* converges; it does not exercise `cipher_koopman_fp16_launch_shape`
  (the .cu kernel) as a cuBLAS substitute. That is the load-bearing S2.B.2 +
  S2.E work.
- **The 32 dead seed entries at `cipher_recipes.cpp:346`** — S2.C scope.
  Re-seeding with narrow-domain LM-head + small-attention shapes is what
  closes plan L401's "central engineering reality."

## 4. Substrate state

Unchanged from W14 Step 2 port-landmark:
- `cipher_rt_phase4` HEAD = `ce4c1b8` tag `week-14-step-2-port-landmark`
- `libcipher_rt.so` md5 = `f6762245bff158a58bcbee1ff7e44193`
- `cipher_kmod` at `week-9-complete` (0.6.5) UNCHANGED
- `cipher_vllm_kv.py` md5 = `2b6cedab89387c30becd49a27313ceb4` UNCHANGED

No commit on cipher_rt_phase4 for B.1 — verification only. The test artifact
lives at `/tmp/step13_2_baseline/test_edmd_realinput_verify.{c,exe}`.

## 5. Files

- `/tmp/step13_2_baseline/test_edmd_realinput_verify.c` (175 LOC)
- `/tmp/step13_2_baseline/test_edmd_realinput_verify` (compiled binary)
- `cipher-fusion-evidence/WEEK_14_STEP_2_B1_EDMD_VERIFY.md` (this doc)

## 6. Next step

S2.B.2 — `cipher_rt_koopman_engine` wire-up (NEW), pattern parallel to
`cipher_rt_marlin_engine`. Hooks the existing cuBLAS interception layer in
cipher_rt_phase4 at the same boundary Marlin uses. Three guards before
substitution: (1) registry-hit (model-keyed per W13 Step 1
`cipher_rt_recipe_model_key`), (2) shape in seeded narrow-domain set
(Barron-bounded), (3) model_uuid match. If all three pass + oracle PERMIT,
dispatch to `cipher_koopman_fp16_launch_shape()` instead of cuBLAS. Else
passthrough to cuBLAS. apply_recipe()'s Koopman-validator path left
untouched.
