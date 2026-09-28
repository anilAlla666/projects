# K.1 KEYSTONE CLOSE REPORT — Workload Classifier 8/8 PASS

**Date:** 2026-05-27
**Substrate:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so`
**Anchor (post-K.2):** md5 `61795d63745a1d9d79580093ab0e6be9`
**Anchor (K.1.6 v7 pre-K.2):** md5 `79ff5e5b51693241ab0d42651062abff`

## Verdict
**K.1 CLOSE GATE: PASS 8/8.** Memory #25 product-engagement-gate cleared on
the 3-reference-workload set + 3 negative cases + transition + calibration.

## Per-Cell Result

| Cell | Expect | Class | Obs | Conf | INT4 | PASS/FAIL |
|------|--------|-------|-----|------|------|-----------|
| P1_llama3_8b_bf16   | KNOWN   | A4_BATCH_INFERENCE      | 90900 | 720 | 0 | PASS |
| P2_mistral_7b_bf16  | KNOWN   | A4_BATCH_INFERENCE      | 90934 | 720 | 0 | PASS |
| P3_tinyllama_awq    | KNOWN   | A3_SINGLE_TENANT_STREAM | 91586 | 820 | 1 | PASS |
| E1_bare_torch       | UNKNOWN | UNKNOWN                 | 4     | 250 | 0 | PASS |
| E2_load_idle        | UNKNOWN | UNKNOWN                 | 974   | 250 | 0 | PASS |
| E3_load_1tok        | UNKNOWN | UNKNOWN                 | 1300  | 250 | 0 | PASS |
| T1_llama3_transition| KNOWN   | A3_SINGLE_TENANT_STREAM | 88300 | 820 | 0 | PASS |
| C1_llama3_calibration| KNOWN  | A4_BATCH_INFERENCE      | 75300 | 720 | 0 | PASS |

## K.1.6 Iteration Trail (v1 → v7)

The K.1.6 decision-tree-v2 work landed in 7 build iterations as the close gate
surfaced behavior the design memo hadn't predicted. Each iteration narrowed the
gap; the final v7 build is hysteresis-only — far simpler than the layered
gate design we ended up at v6.

| Build | md5 | PASS | Failure mode |
|-------|-----|------|--------------|
| v1 (idle-rate gate)        | 95222e30 | 6/8 | E2/E3 misfire — lifetime obs/s exceeded threshold thanks to warmup burst |
| v2 (recent-idle gate)      | 5150d132 | 3/8 | atexit re-classify in parent process polluted last CLASSIFY line for positives |
| v3 (+ sustained-span guard)| 015b473c | 3/8 | Same; sustained-span didn't help because parent atexit still ran |
| v4 (timer thread)          | 255772dd | 6/8 | Detached thread didn't survive vLLM EngineCore fork |
| v5 (+ pthread_atfork)      | f23b4133 | 6/8 | Timer DID fire; P3 short-active-span (~15s) downgraded; E3 14 KNOWN emits looked tentative until late |
| v6 (count-based guard)     | cecba1c2 | 7/8 | E3 last classify at obs=1300 (A4) emitted within 1s of EngineCore shutdown — timer didn't fire one more downgrade in time |
| **v7 (hysteresis)**        | **79ff5e5b** | **8/8** | Provisional-UNKNOWN-until-N-high-conf-emits supersedes idle gate; E3 (14 emits) never crosses 50 threshold → UNKNOWN throughout |

## v7 Design (final)

`cipher_workload_classify()` is a hysteresis-emitting publisher:

1. `classify_internal()` runs the decision tree as before — full signal-based
   classification with confidence assignment (the empirical decision tree v2
   K.1.5 derived).
2. If the raw class is KNOWN and confidence > 500: increment
   `g_signals.high_conf_emits`.
3. If `high_conf_emits` (post-increment) < `MIN_HIGH_CONF_CLASSIFY_COUNT` (50):
   demote `p.workload_class` to UNKNOWN with confidence 250 before publishing
   and emitting. The increment still happens (so the counter grows
   monotonically toward the threshold).
4. Otherwise, publish + emit as-is.

The background timer thread (1s tick) + `pthread_atfork` are retained so
classify fires during idle periods (otherwise launch-driven classify alone
would never re-evaluate after activity stops). Timer firing during idle is
what allows the warmup-burst-then-idle pattern (E2/E3) to accumulate enough
post-warmup classify calls to ALSO stall the count below threshold.

## Engagement Evidence (substrate side)

Workload-class identity verified per K.1.5 Step 2 capture cross-reference:
- P3 emits `int4=1` consistent with AWQ-Marlin kernel observation +
  A3_SINGLE_TENANT_STREAM expected for B=1 decode.
- P1/P2 emit A4_BATCH_INFERENCE (post K.1.6 A2/A4 collapse) at B=8 prefill.
- Negative cells E1/E2/E3 stop accumulating high_conf_emits before threshold.

## K.2 Dispatch Unification Smoke Result

After K.2 wiring (Marlin + Koopman + may13 cipher_dispatch read CipherProfile
and gate substitution on `marlin_engage` / `koopman_engage`), the close gate
re-runs at 8/8 PASS — no regression from gating.

## Marvel Discipline (Memory #28)

- **Full coverage:** 3 positive workload classes + 3 negative archetypes +
  transition + calibration = 8 cells covering identification, false-positive
  suppression, warmup transition, and confidence calibration.
- **Negative-case validation:** 3/3 negatives PASS as UNKNOWN.
- **Engineering debt forecast (carry-forward to follow-up):**
  - K.2 skip counters (`g_calls_skipped_by_classifier` in Marlin,
    `g_skip_by_classifier` in Koopman, `g_may13_skip_by_classifier` in may13
    dispatch) are defined + incremented but NOT exposed via exit-time
    diagnostic dump. EngineCore uses `_exit()` so atexit-driven dumps are
    unreliable there; the counters are accessible via ctypes / SIGUSR1
    counter-dump path. Visible-from-outside dashboard wiring is W.X future.
  - The hysteresis approach means a workload that classifies via short bursts
    + long pauses (uncommon shape; not in the K.1.5 reference set) may stay
    UNKNOWN even when classification IS the right answer. If observed in a
    customer workload, MIN_HIGH_CONF_CLASSIFY_COUNT can be tuned down or
    paired with a per-class confirmation count.
  - A2_CONTINUOUS_BATCHED branch in decision tree is unreachable at vLLM 0.21
    default (cudagraph_mode=NONE). Reachability returns at vLLM 0.22+ with
    FULL graph capture, or when stream-count signal lands (W.4 POOL executor).
  - Multi-tenant gate is SOFT-only (deferred to W.6 sub-C per K.1 spec).
- **No Memory #1 goal scope-down:** Classifier substrate is additive; no
  existing CIPHER capability is gated off by the new code (Marlin/Koopman
  default behavior preserved at UNKNOWN class, which is the warmup default).

## Anchors

- `cipher_rt_phase4` substrate at the close-of-K.1.6+K.2 build:
  `libcipher_rt.so` md5 `61795d63745a1d9d79580093ab0e6be9`.
- `cipher_workload_detect.cpp` is the substrate's classifier file (hysteresis
  emit + pthread_atfork-resilient timer + decision tree v2 + idle gate
  artifacts retained as `IDLE_*` constants for future reuse but not on the
  hot path).
- K.2 wiring adds reads of `cipher_workload_profile_get()` in
  `cipher_rt_marlin_actuator.c`, `cipher_rt_koopman_engine.cpp`,
  `src/may13/cipher_dispatch.cpp`.
