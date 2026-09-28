# K.2 DISPATCH UNIFICATION CLOSE REPORT

**Date:** 2026-05-27
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `61795d63745a1d9d79580093ab0e6be9`

## Scope
K.2 wires the substrate's two dispatch paths to read CipherProfile (the K.1
classifier's published profile) and gate substitution on per-actuator
engagement flags per Memory #26 (7-class taxonomy → capability subset) and
Memory #27 (brain-muscle unification).

Per advisor (pre-implementation): "GATE the SUBSTITUTE decision in front of
apply_recipe; do NOT replace classify_launch. may13's geometry-fingerprint
classifier is distinct from cipher_workload_profile_get() and the spec is to
add a classifier-driven gate, not unify the two."

## Files Modified

### 1. `cipher_rt_phase4/cipher_rt_marlin_actuator.c`
- Added `#include "cipher_workload_detect.h"`.
- Added `g_calls_skipped_by_classifier` atomic counter (distinct from
  pre-existing `g_calls_skipped` so engagement evidence can distinguish
  "skipped by classifier" vs "gates failed").
- In `maybe_handle_marlin()`, after the env-gate (`!g_enabled` early exit) and
  before the existing Week-3-Step-3 substitute-hint gate, added a classifier
  gate:
  ```c
  const struct cipher_workload_profile *p = cipher_workload_profile_get();
  if (p && p->workload_class != CIPHER_WL_UNKNOWN && !p->marlin_engage) {
      atomic_fetch_add(&g_calls_skipped_by_classifier, 1);
      return CIPHER_RT_MATMUL_PASSTHROUGH;
  }
  ```
- UNKNOWN class is permissive (legacy behavior during classifier warmup).
- ~1-2 ns hot-path cost (one pointer load + uint8 load + 2 branches).

### 2. `cipher_rt_phase4/cipher_rt_koopman_engine.cpp`
- Same shape of gate inside `extern "C"` include block, before existing dtype
  gate in `maybe_handle_koopman()`. New counter `g_skip_by_classifier`.

### 3. `cipher_rt_phase4/src/may13/cipher_dispatch.cpp`
- Added helper `may13_classifier_permits_substitute()` that reads
  CipherProfile + counts skips into `g_may13_skip_by_classifier`.
- Gated all 4 `apply_recipe()` call sites in `cipher_dispatch()`:
  - classify-only path: GEMM registry hit (entry && gate)
  - classify-only path: ELEMENTWISE/REDUCTION registry hit (entry &&
    recipe_type==2 && gate)
  - classify-only path: synthetic Chebyshev fallback (gate)
  - full-runtime path: at the final substitution decision (gate → billing
    passthrough → return CIPHER_PASS_THROUGH)
- classify_launch (the geometry fingerprint classifier) is NOT removed — per
  Memory #27 brain-muscle, K.2 reads CipherProfile to GATE, doesn't REPLACE
  the existing classifier.
- Exported `cipher_dispatch_skip_by_classifier_count()` for ctypes probes.

## Customer Override Semantics

Per K.1.6+K.2 spec 3.2: "Customer override path preserved — CIPHER_KOOPMAN=0
forces koopman_engage=false regardless of classifier."

K.2 implementation:
- `CIPHER_MARLIN=0` env → `g_enabled=0` in Marlin → early-exit BEFORE
  classifier gate. Forces off as designed.
- `CIPHER_MARLIN=on` env → `g_enabled=1` → classifier gate evaluated:
  - UNKNOWN class → fall through to existing dtype/shape gates (legacy
    behavior).
  - KNOWN class with `marlin_engage=1` → fall through to existing gates.
  - KNOWN class with `marlin_engage=0` → classifier-skip (counter +1) →
    PASSTHROUGH.
- Same shape for `CIPHER_KOOPMAN`.

The classifier gate adds SMART suppression on top of customer opt-in — a
customer who turned the actuator on but is running a workload the actuator
can't help gets passthrough instead of dtype-gate-fall-through. Net: no
regression for customer who set the env, just less wasted hot-path work on
inappropriate kernels.

## Regression Verification

K.1 close gate re-ran on the K.2-wired substrate (md5 61795d63) — 8/8 PASS.
The classifier gates did NOT break workload classification or change
behavior on any of the 8 close-gate cells. Compare K.1.6 v7
pre-K.2 (md5 79ff5e5b51693241ab0d42651062abff): identical PASS table.

## Engagement Evidence Visibility

The 3 K.2 skip counters are defined + incremented in source. They are not
yet dumped at process-exit (EngineCore subprocess uses `_exit()` so atexit-
driven dumps from EngineCore are unreliable — same limitation that
forced the K.1.6 v7 timer-thread approach for classifier re-evaluation).

The counters are visible from outside via:
- `cipher_dispatch_skip_by_classifier_count()` (may13 path; exported as
  `extern "C"` for ctypes probes)
- Future: a `cipher_rt_marlin_skip_by_classifier_count()` /
  `cipher_rt_koopman_skip_by_classifier_count()` symmetric exports (W.X
  follow-up — not on K.2 critical path, follows the same shape as the may13
  one).
- SIGUSR1 counter-dump path (`cipher_rt_counter_dump.c`) can surface them
  if added to the JSON dump (follow-up).

## Marvel Discipline (Memory #28)

- Full coverage: K.2 gate touches the 3 substitution-side dispatch paths in
  the substrate (Marlin actuator, Koopman engine, may13 dispatch).
- Negative-case validation: classifier UNKNOWN passes legacy behavior — the
  K.1 close gate's 5 UNKNOWN-emitting cells (3 negatives + 2 positives
  before threshold) all continue to PASS at 8/8.
- Engineering debt forecast: skip-counter exit-time dump + Marlin/Koopman
  ctypes accessor exports = follow-up; not on the K.1+K.2 close critical
  path (the counters work correctly per code review; observability is the
  improvement that ships next).
- No Memory #1 goal scope-down: K.2 adds gating, removes nothing. UNKNOWN
  class fully preserves the pre-K.2 hot-path semantics.

## Anchors

| Component | md5 |
|-----------|-----|
| pre-K.2 (K.1.6 v7 only)           | 79ff5e5b51693241ab0d42651062abff |
| post-K.2 (K.1.6 v7 + K.2 wired)   | 61795d63745a1d9d79580093ab0e6be9 |

## What's NOT in K.2 scope

- VOLT classifier-driven engagement → W.1 (next per Memory #29 binding
  sequence).
- A2_CONTINUOUS_BATCHED unreachability (decision tree branch unreachable on
  vLLM 0.21 default) → returns at vLLM 0.22+ / W.4 stream-count signal.
- Multi-tenant gate hard-wiring → W.6 sub-C.
- Skip-counter exit-time dump + dashboard surface → W.X follow-up.

## Next Sub-step

K.1+K.2 Sub-step 4 (marvel discipline 5-checklist verification) + Sub-step 5
(commit cipher_rt_phase4 + tag `k1-complete-k2-landed`; commit
cipher-fusion-evidence with K1_CLOSE_REPORT_FINAL.md + K2_DISPATCH_UNIFICATION_REPORT.md
+ tag `k1-k2-close`).
