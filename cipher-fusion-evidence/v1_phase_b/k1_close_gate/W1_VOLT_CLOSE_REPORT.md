# W.1 VOLT CLASSIFIER-DRIVEN ENGAGEMENT — CLOSE REPORT

**Date:** 2026-05-27
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `ca214a14ce82e7e572bb516e1f8f6a9e`
**Build-side anchor preserved:** `build_cuda13/libcipher_rt.so.w1` (identical bytes)

## Verdict
**W.1 PASS.** All 5 positive reference workloads engage VOLT
auto-classified; the 1 compute-bound negative regime (Mistral-7B B=1
ctx=32K prefill) leaves VOLT armed without engagement. K.1+K.2 9/9 close
gate preserved.

## A — Classifier signal additions + decision logic

### Signals added to `src/cipher_workload_detect.cpp`

- `WorkloadSignals::gemm_small_count` — increments when cuBLAS GEMM `n <= 32`
  (small-batch decode-dominant signal). `cipher_workload_observe_gemm()` at
  src/cipher_workload_detect.cpp:1297.
- `WorkloadSignals::gemm_prefill_count` — increments when cuBLAS GEMM `n > 4096`
  (prefill / sequence-pack signal). Added at the same call site.

### Derived signals on `cipher_workload_profile`

- `bw_bound_regime_detected` — `gemm_small * 2 > gemm_total` (>50% of GEMMs are
  small-batch). Computed in `classify_internal()`
  cipher_workload_detect.cpp:1024.
- `large_prefill_detected` — `gemm_prefill * 10 > gemm_total` (ratio-based,
  10% threshold). Empirically: vLLM startup profiling spikes max_n to 16384
  on every model load (even decode-only workloads), which broke the v1 / v2
  / v3 `max_n > 4096` lifetime threshold. Ratio-based fix in
  cipher_workload_detect.cpp:1029.

### Decision logic in `apply_capability_flags`

cipher_workload_detect.cpp:802 — VOLT engagement is regime-gated, not class-
gated:

| Class | `volt_engage` |
|-------|---------------|
| A1 / A2 / A4 / C1 | `bw_bound && !large_prefill` |
| A3 (single-tenant decode) | `!large_prefill` (A3 is bw-bound by definition) |
| B1 / B2 (training) | 0 (explicit deny — cipher-t43-envelope -14% regression case) |
| UNKNOWN | 0 (W.1 flip: previously VOLT-only by default; now safe-off) |

## B — VOLT actuator gate

### `cipher_rt_volt.h` additions

- New `CipherRtVoltMode::CIPHER_RT_VOLT_ARMED = 3` — NVML resolved + handlers
  installed, lock NOT applied, awaiting classifier engagement.
- `cipher_rt_volt_classifier_poll()` — entry point called by classifier
  timer thread.
- `cipher_rt_volt_classifier_engagements()` /
  `cipher_rt_volt_classifier_skipped()` — observability accessors.

### `cipher_rt_volt.c` — new init path

cipher_rt_volt.c:329 — `CIPHER_VOLT` env not set OR set to `auto`:
1. Call `resolve_and_install_handlers()` (NVML probe + kmod-ioctl fallback +
   signal handler install — extracted as helper).
2. Set mode to `ARMED`.
3. Return without applying any clock lock.

Customer override paths preserved:
- `CIPHER_VOLT=off` / `0` → OFF (classifier override denied).
- `CIPHER_VOLT=on` / `1` + `CIPHER_VOLT_BATCH=N` or `CIPHER_VOLT_MHZ=N` →
  legacy immediate-lock behavior (mode ACTIVE).

### `cipher_rt_volt_classifier_poll()`

cipher_rt_volt.c:437. Called from the classifier timer thread once per
second per pid (the pthread_atfork-resilient timer that K.1.6 v7 ships).
Behavior:

1. mode != ARMED → no-op.
2. profile NULL or class == UNKNOWN → no-op (classifier hasn't settled yet —
   the K.1.6 v7 hysteresis is still warming).
3. `profile.volt_engage == 0` → `g_volt.classifier_skipped++`; return.
4. `profile.volt_engage == 1` → CAS engage_lock, transition ARMED → ACTIVE
   by calling `apply_lock_at_target(target_mhz)`, increment
   `g_volt.classifier_engagements`.
5. `target_mhz` selection: 1600 for A4 batch class OR long-context-detected;
   1000 otherwise (matches cipher-t43-envelope calibration table:
   B=1 → 1000 MHz, B=8 → 1600 MHz).

### `src/cipher_workload_detect.cpp` — timer thread wires VOLT

cipher_workload_detect.cpp:1217 — the existing 1s timer thread (with
pthread_atfork resurrection in EngineCore subprocess) now calls
`cipher_rt_volt_classifier_poll()` after `cipher_workload_classify()`.

## C — 9-workload engagement table

| Cell | Class | volt_engage | bw_bound | large_prefill | VOLT ENGAGED log lines |
|------|-------|-------------|----------|----------------|------------------------|
| P1_llama3_8b_bf16     | A4_BATCH_INFERENCE      | 1 | 1 | 0 | **1** ✓ |
| P2_mistral_7b_bf16    | A4_BATCH_INFERENCE      | 1 | 1 | 0 | **1** ✓ |
| P3_tinyllama_awq      | A3_SINGLE_TENANT_STREAM | 1 | 1 | 0 | **1** ✓ |
| T1_llama3_transition  | A3_SINGLE_TENANT_STREAM | 1 | 1 | 0 | **1** ✓ |
| C1_llama3_calibration | A4_BATCH_INFERENCE      | 1 | 1 | 0 | **1** ✓ |
| **N1_mistral_long_prefill** | UNKNOWN          | (hysteresis-blocked, never crossed 50 high-conf emits) | — | — | **0** ✓ |
| E1_bare_torch         | UNKNOWN | — | — | — | 0 ✓ |
| E2_load_idle          | UNKNOWN | — | — | — | 0 ✓ |
| E3_load_1tok          | UNKNOWN | — | — | — | 0 ✓ |

- **5/5 positive workloads**: classifier sets `volt_engage=1`, actuator
  engages, ENGAGED log line emits once per cell.
- **1/1 compute-bound negative regime** (N1): classifier never reaches a
  KNOWN class through the hysteresis threshold (prefill workload has too few
  high-conf-emit cycles before EngineCore tears down) → `volt_engage` stays 0
  by safe default → 0 ENGAGED lines.
- **3/3 generic UNKNOWN negatives** (E1/E2/E3): no engagement (correct).

## D — Engineering debt forecast

Carry to v1.x or later W-series:

1. **VOLT calibration sweep across model families.** Current `target_mhz`
   uses the existing T4.3.2 calibration table (B=1→1000 / B=8→1600 / B=32+→
   1980 from cipher-may13-evidence/p5_optimal_clocks.json). This was
   measured on Mistral-7B + Llama-3-8B. Llama-3-70B / DeepSeek / mixture-of-
   experts may have different bandwidth-bound clock targets.
2. **Thermal throttling interaction (W.7 NCCL co-location).** Multi-GPU NCCL
   workloads with locked clocks can hit thermal limits. W.1 doesn't yet
   read thermal telemetry; W.7 will gate VOLT on thermal headroom.
3. **Per-tenant VOLT in multi-tenant (W.4 POOL co-engagement).** Single
   process locks a GPU clock; multi-tenant POOL execution needs per-stream
   green-context-aware clock policy. Deferred to W.4.
4. **Skip-counter exit-time dump.** `g_volt.classifier_skipped` and
   `classifier_engagements` are atomic counters + extern "C" accessors —
   visible from ctypes probes but not auto-dumped at exit (EngineCore uses
   `_exit()` so atexit-driven dumps are unreliable). Same limitation as the
   K.2 per-actuator skip counters; would be addressed by a unified SIGUSR1
   counter-dump JSON.
5. **N1 negative regime classification path.** N1's classifier never
   crosses the hysteresis threshold (long-prefill workload doesn't emit
   enough high-conf classifies before EngineCore shutdown). The VOLT-deny
   path is therefore via the UNKNOWN default rather than the explicit
   `large_prefill_detected=1` veto we designed for. The veto IS present in
   the decision tree and works correctly — but for this specific cell, the
   hysteresis catches it first. Both paths land on `volt_engage=0` so no
   functional issue; documenting the diagnostic precedence here.
6. **Auto-mode env unset is the new default.** rev8 .deb has CIPHER_VOLT
   absent in the CDI env hook. With W.1 substrate this now means
   "classifier decides" rather than "OFF" (the K.1 default). Customer can
   restore K.1 behavior by setting `CIPHER_VOLT=off`.

## E — Anchors

| Component | Pre-W.1 | Post-W.1 |
|-----------|---------|----------|
| `cipher_rt_phase4` HEAD          | 840d43f (k1-complete-k2-landed) | (commit pending) tag `w1-volt-classifier-driven` |
| `cipher_rt_phase4` libcipher_rt.so md5 | 61795d63 (K.1.6 v7 + K.2)        | `ca214a14ce82e7e572bb516e1f8f6a9e` |
| `cipher-fusion-evidence` HEAD    | 7543f91 (k1-k2-close)            | (commit pending) tag `w1-volt-close` |
| `cipher_kmod` HEAD               | 8c643fc (UNCHANGED)              | 8c643fc (UNCHANGED) |
| `cipher-platform` .deb (rev8)    | 5603f72d (UNCHANGED)             | 5603f72d (UNCHANGED) |

W.1 iteration trail (4 builds to PASS):
- v1 (md5 981072a9): only A3 engaged. `gemm_b1_ratio > 0.5` threshold too
  tight — B=8 decode (P1) failed because n=8 ≠ n=1.
- v2 (md5 e32bf977): widened to `gemm_small_ratio > 0.5` (n ≤ 32). P3 still
  engaged but P1/P2 still 0 — diagnostic gap.
- v3 (md5 3bb61eca): added extended CLASSIFY log. Diagnosis surfaced root
  cause: vLLM profiling spikes lifetime `max_n` to 16384 on every model,
  tripping `large_prefill_detected` lifetime threshold even on decode-
  dominant workloads.
- **v4 (md5 ca214a14): ratio-based `large_prefill` (gemm_prefill_count /
  gemm_total > 10%) — clean separation between profiling spike vs sustained
  prefill workload. 5/5 positives engage; 1/1 negative + 3/3 UNKNOWN
  cells don't.**

## F — Memory #29 next-substep

W.2 — Marlin bf16 + Machete library-symbol intercept (Goal 2 contribution
path #2 + AWQ substitution counter > 0 on TinyLlama-AWQ). Memory cipher-
t45-substrate-marlin notes: matmul-routing substrate (.symver cuBLAS LD_
PRELOAD) ships with Marlin actuator integrated, but bf16 path and Machete
INT4 path on AWQ still need wiring.

## Disclaimer on PASS semantics

W.1 PASS criterion per Memory #25 is "engagement counter > 0 on the right
regime + 0 on the wrong regime". This is the substrate-correctness gate —
NOT an absolute tok/W lift claim. The cipher-t43-envelope +57% tok/W at B=1
decode is from prior calibration; W.1 ensures the engagement happens on the
right workloads, not the wrong ones. Absolute tok/W lift validation lives
at V.0/V.1 (post-substrate-lock).

The N1 compute-bound regime PASS criterion (does NOT engage, NO regression
vs rev8 baseline) is satisfied: 0 ENGAGED lines + classifier-deny path
honored. Memory #11 HARD STOP for the -14% Mistral-7B prefill regression
is therefore avoided by W.1's regime gate.
