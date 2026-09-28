# CP 5.4 — Step 1.6B-3 — RESULTS (naive Arm-A baseline sweep)

**Date:** 2026-05-19. **Verdict: SWEEP COMPLETE — 15/15 reps PASS.** The
naive Arm-A baseline for the Step 1.6 2-arm methodology. This is the
comparison point Arm-B (CIPHER mixed deployment, 1.6B-4) will be measured
against — **not yet a comparison; this document is the baseline only.**

Tools: `cp54_s16_naive_orchestrator.py`, `cp54_s16_naive_sweep.py`. Summary
JSON: `cp54_naive_{op2,op5,opasym}_result.json`, `cp54_naive_sweep_summary.json`.

---

## Method

N **naive** TinyLlama-1.1B tenants per OP — plain PyTorch, full GPU, ordinary
CUDA time-slicing; **no CIPHER kmod, no `CUDA_INJECTION64_PATH`, no
`LD_PRELOAD`, no MPS/MIG**. Per-round lockstep barrier keeps contention
windows co-extensive. Identical model / WL01 prompts / decode length / B=1 /
per-decode-step timing path as the Arm-B tenant. 10 rounds/rep; the first
timed round is discarded as warmup (the tenant also runs an untimed warmup
round before that). 5 reps per OP. CV = std/mean of per-decode-step latency.

**Tenant counts — the adjudicated design-memo §2/§4 Arm-A totals**
(PARTITION-class + 5 POOL-class logical tenants; confirmed 2026-05-19 after
the 1.6B-3 build-log flagged the Phase-2 prose undercount):

| OP | tenants | = |
|---|---|---|
| OP-2 | **7** | 2 partition-class + 5 pool-class |
| OP-5 | **10** | 5 + 5 |
| OP-asym | **9** | 4 + 5 |

In Arm A every tenant is the identical naive workload — there is no SM slice,
so OP-asym carries no per-tenant "size"; its asymmetry is purely an Arm-B
property.

**Environment.** kmod **unloaded for the entire sweep** (`rmmod` before,
verified `kmod_loaded=False` in all 15 rep JSONs, `insmod 008b3c66` after).
Anchors `md5`-verified **unchanged** start and end: kmod `008b3c66`,
libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`, libcipher_v2 `cc0479b8`.

## Results — per OP, 5 reps each

| OP | tenants | aggregate tok/s | per-tenant CV (mean) | per-tenant CV (max) | KL gates |
|---|---|---|---|---|---|
| OP-2 | 7 | **267.4 ± 2.0** | **0.160 ± 0.004** | 0.167 – 0.187 | 35/35 PASS |
| OP-5 | 10 | **351.0 ± 9.5** | **0.183 ± 0.018** | 0.176 – 0.263 | 50/50 PASS |
| OP-asym | 9 | **318.6 ± 17.6** | **0.210 ± 0.010** | 0.228 – 0.254 | 45/45 PASS |

(± is the standard deviation across the 5 reps. KL count = tenants × reps;
every per-tenant teacher-forced logit-KL gate passed, ≤ 0.1, campaign-norm
~1e-4.)

**Per-tenant decode latency (warm, pooled across all reps):**

| OP | tok/s/tenant (range, mean) | decode-step mean (ms) | decode-step p99 (ms) |
|---|---|---|---|
| OP-2 | 35.8 – 41.4, mean 38.2 | 24.2 – 28.0 | 55.0 – 60.3 |
| OP-5 | 31.9 – 38.5, mean 35.1 | 26.0 – 31.4 | 48.6 – 67.4 |
| OP-asym | 32.0 – 39.5, mean 35.4 | 25.3 – 31.2 | 47.5 – 66.6 |

**Framebuffer (3-point, MiB) — every rep:**

| OP | idle | loaded | exited |
|---|---|---|---|
| OP-2 | 0 | 21619 | 0 |
| OP-5 | 0 | 30882 | 0 |
| OP-asym | 0 | 27794 | 0 |

`loaded` is **bit-identical across all 5 reps** of each OP — deterministic;
`exited` returns to `idle` (0) every rep — **clean teardown, no leak**.
`loaded / tenants` ≈ **3088 MiB/tenant**, constant across OPs — each naive
tenant holds a full private model copy (≈ 2.1 GiB weights + ≈ 1.0 GiB
context). This is the no-sharing counterfactual that Track 2 weight-sharing
addresses; recorded here as the Arm-A memory baseline.

## Observations (baseline characterization — not a comparison)

- **CV grows with contention — but NOT monotonically in tenant count.**
  OP-2 (7 t) 0.160 → OP-5 (10 t) 0.183 → **OP-asym (9 t) 0.210**. OP-asym has
  *fewer* tenants than OP-5 yet a *higher* CV — the cv_max ranges do not even
  overlap (OP-asym 0.228–0.254 vs OP-5 0.176–0.263 — and OP-asym's *floor*
  0.228 sits above OP-5's mean). **This non-monotonicity is flagged, not
  smoothed over.** In naive Arm-A all 9 OP-asym tenants are identical, so the
  break is not a workload-size effect; it is either cross-rep variance or a
  contention-mode shift at N=9. **To revisit once Arm-B OP-asym data lands**
  (1.6B-4): if Arm-B OP-asym shows the same break, it is a real finding; if
  Arm-B is clean, the Arm-A OP-asym spike is naive-only and worth root-causing.
  Do not fit it under a "CV rises with tenant count" claim — that claim is
  false as stated.
- **Aggregate tok/s is non-monotonic** (267 → 351 → 319) — OP-5's 10 tenants
  extract the most aggregate throughput from time-slicing; OP-asym's 9 sit
  between. Descriptive only — *not* a substrate claim (no substrate here).
- **OP-5 has the widest cross-rep spread** (agg ±9.5, CV ±0.018, cv_max up to
  0.263) — the heaviest-contention point is the noisiest, as expected.
- All 15 reps clean: KL gates pass, kmod confirmed unloaded, teardown leak-free.

## Measurement caveats — recorded honestly

- **Framebuffer determinism is observed, not audited.** `loaded` was
  byte-identical across all 5 reps of each OP (21619 / 30882 / 27794 MiB).
  This is plausible — identical model, identical load order, the CUDA caching
  allocator rounds reservations — but the reading is a single-shot
  `nvidia-smi memory.used`. It is recorded as *observed exact reproducibility*,
  not as audited determinism; if a memory claim is built on it in 1.6B-4 it
  should be re-sampled at varied points first.
- **CIs are deferred to 1.6B-4 by design.** Per design memo §11.2 the headline
  is a bootstrap-95%-CI comparison of Arm-B vs Arm-A per-tenant CV. The Arm-A
  CVs above are the population for that CI; 1.6B-4 computes the bootstrap CIs
  on **both** arms so the comparison has CI separability as its PASS/FAIL test
  — this is not a "we got CV X" claim, and the CI must be computed when Arm-B
  lands, not deferred further.

## Verdict

**1.6B-3 COMPLETE — the naive Arm-A baseline is established and healthy.**
Per-tenant throughput is reasonable for time-sliced TinyLlama B=1; per-tenant
CV is elevated under contention (0.16–0.21); teardown is leak-free;
correctness holds on every tenant; no anomaly indicating a baseline-harness
defect. The baseline is sound to measure Arm-B against. One datum to carry
forward: the **OP-asym CV non-monotonicity** (above) — open, to be resolved
against Arm-B OP-asym in 1.6B-4.

**STOP for adjudication before Phase 4** (the 1.6B-4 Arm-B CIPHER harness
build). The headline 1.6 finding — does CIPHER tighten per-tenant CV vs this
naive baseline — is computed in 1.6B-4, not here.
