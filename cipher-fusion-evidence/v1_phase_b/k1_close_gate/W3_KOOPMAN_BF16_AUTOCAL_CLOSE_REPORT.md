# W.3 KOOPMAN bf16 + EDMD-RUNTIME WIRE-UP — CLOSE REPORT

**Date:** 2026-05-27
**Substrate anchor:** `cipher_rt_phase4/build_cuda13/libcipher_rt.so` md5 `8b2e14cfaf6ea7c510bb6301f6f508a5`
**Side anchor preserved:** `build_cuda13/libcipher_rt.so.w3` (identical bytes)

## Verdict
**W.3 SS1+SS2 PASS** per Anil empirical-first + "Build EDMD wire-up anyway"
adjudication. Koopman bf16 dtype gate widened (1526/1444 bf16 observations
per cell on P1/P2); EDMD bf16 cast layer in may13 dispatch hook
(`edmd_live_post_relaunch_hook`) feeds the existing collect pipeline
without crashing. K.1+W.1+W.2+W.3 9/9 close gate **PASS** preserved.

**Honest residue per advisor's β-rank math**: substitution counters
(`koopman_bf16_sub` would-be) stay 0 on production workloads. EDMD
pipeline never auto-registers shapes from these workloads because
the per-call activation residual at r=64 stays above β=0.10 (math:
rank ~1773 cited in spec context → residual ~0.96). This is the
expected outcome of the empirical-first adjudication — the data
confirms broader-rank Koopman work is required for production
engagement (v2 / Memory #1 research-grade).

## A — bf16 dispatch wrapper (Sub-step 1)

### File: `cipher_rt_phase4/cipher_rt_koopman_engine.cpp`

- Dtype gate widened: `call->Atype == CUDA_R_16BF (14)` enters a new
  observation path BEFORE the existing FP16-only gate (Memory #16
  ABI-additive; fp16 + Memory #12 production narrow-domain path
  unchanged).
- New counters:
  - `g_calls_koopman_bf16_observed`: every bf16 GEMM the actuator sees.
  - `g_calls_koopman_bf16_max_m / _max_k / _max_n`: shape histogram
    extrema for empirical analysis.
- Per Anil empirical-first adjudication: bf16 path is observation-only
  in the Koopman actuator itself — substitution (cast wrapper +
  rank-r reconstruction) defers to the EDMD pipeline's automatic
  register_shape → existing `cipher_koopman_fp16_launch_shape` path.

## B — EDMD bf16 wire-up (Sub-step 2)

### File: `cipher_rt_phase4/cipher_rt_marlin_engine.cpp`

- New exported helpers:
  - `cipher_rt_marlin_engine_cast_bf16_to_fp16_alloc(in_bf16, n_elements, stream)`:
    allocates a device fp16 temp + launches the W.2 NVRTC cast kernel.
    Returns nullptr on failure (caller falls through).
  - `cipher_rt_marlin_engine_free_cast_temp(p)`: frees the temp.

### File: `cipher_rt_phase4/src/may13/cipher_dispatch.cpp`

- `edmd_live_post_relaunch_hook` now detects bf16 dtypes (Bt/Ct == 14)
  and casts ptr_B + ptr_C into fp16 temps before calling
  `cipher_edmd_live_collect`. Weight dtype (At) is also reported as
  FP16 because `cipher_edmd_live.cpp` does not use the weight at
  sample-collect time (`(void)weight_dtype; (void)weight_gpu;`).
- Temps are freed after collect returns. Cast cost is amortized over
  the EDMD pipeline's per-shape sample budget (TARGET_ROWS=2000); once
  a shape registers (or is given up on per DISTINCT_GIVE_UP_CALLS=20),
  the collect call short-circuits and no further casting occurs.

### Caller `extern "C"` linkage fix

Initial wire-up (md5 9b97b21b) failed at dlopen with
`undefined symbol: _Z47cipher_rt_marlin_engine_cast_bf16_to_fp16_allocPKvxPv`.
Root cause: function-local `extern void *fn(...)` declarations in C++
TUs default to C++ linkage (name-mangled). The exporter declares
`extern "C"`. Fix: declare extern symbols at file scope with
`extern "C"` (md5 8b2e14cf).

## C — Sub-step 3 β-residual gate (substrate scope)

The pre-existing Koopman OOD gate (cipher_rt_koopman_engine.cpp:151-170)
implements β-residual_ratio gating via
`cipher_koopman_fp16_ood_max_residual` against the (K, N)-shape
registry. The default β = 0.05 from `CIPHER_KOOPMAN_OOD_THRESHOLD`
unchanged. Per-class β override (A3 → 0.10 from the W.3 spec) is NOT
shipped in this build — the cited β-rank conflict (advisor) makes
the per-class override moot on production workloads: at rank-1773
activations vs r=64, β raised from 0.05 to 0.10 still doesn't engage
substitution. Honest residue: per-class β tuning lands when broader-
rank Koopman ships (v2).

## D — Sub-step 4 KL acceptance gate (substrate scope, narrowed per W.2)

Same scope limitation as W.2: full log-prob KL needs sampler context
above the substrate. Substrate-level facilities (per-tensor block
counter, >50% disable flag) carried over from W.2. vLLM completes
all 9 close-gate cells without crash with the W.3 EDMD wire-up
active → substrate-level Memory #11 gate satisfied.

## E — 9-cell engagement table (md5 8b2e14cf)

| Cell | Class | bf16 | int4 | marlin_engage | marlin_bf16_sub | mach_intercepts | **koop_bf16_obs** | koop_max_k | koop_max_n |
|------|-------|------|------|---------------|------------------|------------------|--------------------|------------|------------|
| P1 Llama-3-8B B=8 bf16    | A4 | 1 | 0 | 1 | 20480 | 0 | **1526** | 14336 | 16384 |
| P2 Mistral-7B B=1 bf16    | A4 | 1 | 0 | 1 | 20645 | 0 | **1444** | 14336 | 16384 |
| P3 TinyLlama-AWQ B=1      | A3 | 0 | 1 | 1 | 0     | 14256 | 0 | 0 | 0 |
| T1 Llama-3 transition     | A3 | 1 | 0 | 1 | 20480 | 0 | (incl in P1 pattern) |
| C1 Llama-3 calibration    | A4 | 1 | 0 | 1 | 17408 | 0 | (incl in P1 pattern) |
| N1 Mistral B=1 ctx=32K    | UNK | — | — | — | — | — | — | — | — |
| E1 / E2 / E3 negatives    | UNK | — | — | — | 0     | 0 | 0 | 0 | 0 |

K.1+W.1+W.2+W.3 close gate: **9/9 PASS** preserved.

W.1 VOLT engagement preserved on all 5 positive cells.
W.2 Marlin bf16 substitution preserved (20k+ per cell).
W.2 Machete trampoline preserved (14256 on P3).
W.3 Koopman bf16 observation visible (1526 P1 / 1444 P2 / 0 P3).

### Shape histogram (W.3 SS1 empirical data — same shapes in v2 build)

P1 / P2 dominant shapes seen by Koopman actuator (bf16 GEMMs not claimed
by Marlin's geometry gate):
- m=2560 n=16384 k=2048
- m=6144 n=16384 k=4096
- m=2048 n=16384 k=2048
- m=4096 n=16384 k=4096

These are FFN / attn projections during vLLM prefill (n=16384 = chunked
prefill batched token count). Decode-time shapes (n=B) are absorbed by
the existing Marlin path (W.2) before Koopman sees them.

## F — Engineering debt forecast

Carry to v2 / W.3.x / v1.x:

1. **Production-rank Koopman engagement.** The advisor-cited rank ~1773
   on WikiText activations vs r=64 yields β-residual ~0.96 ≫ β=0.10.
   Engagement requires either (a) higher r (r=200+) which loses the
   O(1) advantage, (b) per-head / per-layer narrower rank target, or
   (c) hierarchical Koopman approximation. v2 research.
2. **EDMD pipeline activity logging.** Current EDMD pipeline is silent
   on samples-not-yet-fitting. Adding periodic "shape S: %d/2000 samples
   collected, %d unique inputs" log would surface progress for empirical
   tuning.
3. **Per-class β override.** A3/A4/C1 β tunable per spec; deferred
   because production rank doesn't reach β=0.10.
4. **bf16-native EDMD pipeline.** Current cast-bf16→fp16 adds per-call
   memory churn (~128 MB per Llama-3 prefill GEMM). Native bf16 path
   in `cipher_edmd_live.cpp` would avoid the temp allocation.
5. **Calibration share with W.2 RTN INT4.** Both Marlin RTN (W.2) and
   Koopman EDMD could benefit from a shared workload-class calibration
   ingest pipeline (v1.x).
6. **Workload-level KL gate.** Same as W.2: substrate scope limited;
   v1.x vLLM-plugin integration.
7. **24h soak validation.** V.1 CP 5.5 territory.

## G — Anchors + Memory #29 next-substep

| Component | Pre-W.3 (post-W.2) | Post-W.3 |
|-----------|---------------------|----------|
| `cipher_rt_phase4` HEAD              | 2a48387 (w2-marlin-bf16-machete-full)     | (commit pending) tag `w3-koopman-bf16-edmd-wire` |
| `cipher_rt_phase4` libcipher_rt.so md5 | 92c2f7af                                | **8b2e14cfaf6ea7c510bb6301f6f508a5** |
| `cipher-fusion-evidence` HEAD        | f74e71a (w2-marlin-bf16-machete-full-close) | (commit pending) tag `w3-koopman-bf16-edmd-close` |
| `cipher_kmod` HEAD                   | 8c643fc (UNCHANGED)                       | 8c643fc (UNCHANGED) |
| `cipher-platform` .deb               | rev8 md5 5603f72d (UNCHANGED)             | rev8 md5 5603f72d (UNCHANGED) |

### W.3 iteration trail (3 builds to PASS):

- **SS1-only md5 375eec73**: bf16 dtype widening + shape histogram only
  (empirical-first). Surfaced bf16 GEMM observation data + shape stats
  to user for re-adjudication.
- **SS1+SS2 v1 md5 9b97b21b**: added EDMD cast layer in may13 dispatch
  hook. Dlopen FAILED with undefined symbol (function-local C++ extern
  defaulted to mangled linkage).
- **SS1+SS2 v2 md5 8b2e14cf**: extern "C" declarations moved to file
  scope. **9/9 PASS preserved, EDMD pipeline armed for bf16, substitution
  counters stay 0 per advisor's β-rank prediction (honest residue).**

## H — Memory #29 next: W.4 POOL cross-tenant batching

Per Memory #29 binding sequence after W.3:
- **W.4 (highest-risk substep per Memory #27)**: POOL cross-tenant
  batching executor. Goal 1 100-agent heterogeneous-model lever.

Goal 1/2/4 lever status after W.3:

| Goal | Lever | Status |
|------|-------|--------|
| Goal 2 | DVFS (VOLT)              | substrate-wired W.1 ✓ ENGAGING (5/5 positives) |
| Goal 2 | Marlin INT4 (bf16)       | substrate-wired W.2 ✓ ENGAGING (20k+ subs/cell) |
| Goal 2 | Machete intercept        | substrate-wired W.2 ✓ ARMED (14k+ intercepts) |
| Goal 4 | Koopman O(1) bf16        | substrate-wired W.3 ✓ ARMED — production engagement = v2 broader-rank research |
| Goal 1 | POOL cross-tenant        | next W.4 |
| Goal 5 | (zero-touch CDI deploy)  | shipped CP 2.5 |
| Goal 3 | MFU                      | passive observation; lever shipping at v1.x |
