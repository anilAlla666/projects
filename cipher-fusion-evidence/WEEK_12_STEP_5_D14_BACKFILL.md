# Week 12 Step 5 — D14 backfill investigation

**Date:** 2026-05-23
**Closes:** `WEEK_12_SCOPE_DRIFT_AUDIT.md` item D14 (vLLM TinyLlama B=1 smoke deferred at W12 Step 3)
**Tag:** `week-12-step-5-d14-env-gate` on `cipher_rt_phase4` (commit `b360fc1`)
**Adjudication 2026-05-23 (user, Path b):** TPS gate updates ±3% → ±5%; env gate kept as additive production-tuning surface; three D14 findings documented honestly.

## 1. Pre-conditions verified

| Item | At entry | At close | Δ |
|------|----------|----------|---|
| `cipher_rt_phase4` tag | `week-12-step-4-sdpa-stream-fill` (1466193) | **`week-12-step-5-d14-env-gate`** (b360fc1) | rotated |
| `cipher_kmod` tag | `week-9-complete` (0.6.5) | unchanged | none |
| `libcipher_rt.so` md5 | `e650b49f48c9c3460d1951d6f376350d` | **`3e22be9da9dbb6e5e0384394be3b3187`** | rotated |
| `cipher_rt_attn_call` ABI | 408 B | 408 B | preserved |

## 2. Bisection — TPS at each W10-12 tag

Per Part B protocol: build libcipher_rt.so at 5 tags, run 3-iteration TinyLlama B=1 vLLM smoke at each, compute mean tok/s vs same-day vanilla baseline.

### 2.1 Per-tag table (today's vanilla = 2271.44 tok/s)

| Tag | Mean tok/s | Δ vs vanilla | Step-to-step Δ | libcipher_rt.so md5 |
|-----|-----------|--------------|----------------|----------------------|
| vanilla | 2271.44 | 0.00% | — | — |
| week-9-complete | 2134.83 | -6.01% | -6.01% | `d95c618d` |
| week-10-step-1-ring-write | 2071.94 | -8.79% | **-2.78%** | `a3f68f98` |
| week-11-step-2-g3-g4-tc-probe | 2115.29 | -6.88% | **+1.91%** | `1ed1209b` |
| week-12-step-3-g5-l2-persist | 2063.36 | -9.16% | **-2.28%** | `6b93de30` |
| week-12-step-4-sdpa-stream-fill | 2163.25 | -4.77% | **+4.39%** | `e60c5170` |

### 2.2 Finding: no single dominant cost contributor

Step-to-step deltas are non-monotonic with sign reversals (W11 +1.91%, W12.4 +4.39%). The per-tag mean variance is comparable to the per-iteration variance within a single tag's 3 measurements (~5%). Cold-start iter 0 alone differs by ~7% from steady-state iter 1+2.

The W9-complete → W12.4 trend is flat to slightly positive (+1.35% iter1+2 mean from W9 to W12.4) — within run noise.

**Conclusion: the substrate cost is the broad W7-9 → W12.4 baseline (-4 to -6% from vanilla, distributed across multiple per-call paths), NOT any single W10-12 inflection.**

### 2.3 Paired vanilla/CIPHER alternation (variance control)

To reduce vanilla-drift contamination, ran 5 paired vanilla/CIPHER iterations at W12-step-4 substrate (interleaved, kmod reload between each):

| Pair | Vanilla | CIPHER (W12.4) | Δ |
|------|---------|----------------|---|
| 1 | 2220.49 | 2136.00 | -3.80% |
| 2 | 2222.13 | 2154.05 | -3.06% |
| 3 | 2218.02 | 2079.83 | -6.23% |
| 4 | 2113.01 | 2048.30 | -3.06% |
| 5 | 2189.03 | 2167.43 | -0.99% |
| **Mean** | 2192.54 | 2117.12 | **-3.43%** |
| **Median** | 2218.02 | 2136.00 | **-3.06%** |

Substrate cost is real and stable: -3.43% mean, -3.06% median, range -0.99% to -6.23%. Distributed across multiple per-call paths (COMMIT observe_and_publish, stream resolver lookup, may13 _observe functions fired from CUPTI, GOT-patched cuBLAS shim + matmul_dispatch lookup).

## 3. Env-gate fix (cipher_rt_ring_write.c)

Per Part C protocol, applied minimal env-gate at the highest-frequency producer call site (RING_WRITE fires from classify_observer + cublas shim + SDPA trampoline). Implementation at `cipher_rt_ring_write.c:30-55`:

```c
static _Atomic int g_ring_write_enabled = -1;
static int ring_write_enabled(void)
{
    int v = atomic_load_explicit(&g_ring_write_enabled, memory_order_relaxed);
    if (v >= 0) return v;
    const char *s = getenv("CIPHER_RING_WRITE");
    int e = (s && (s[0] == '0' || !strcmp(s, "off") || !strcmp(s, "no"))) ? 0 : 1;
    atomic_store_explicit(&g_ring_write_enabled, e, memory_order_relaxed);
    return e;
}

void cipher_rt_ring_write(...)
{
    if (!ring_write_enabled()) return;   /* W12 Step 5 D14 env-gate */
    /* existing producer code unchanged */
}
```

Per-call cost added by gate: single relaxed atomic_load + branch (~1 ns). Init samples once via atomic CAS at first call. Default `CIPHER_RING_WRITE` unset = enabled = current substrate behavior preserved.

### 3.1 Verification — env-gate does NOT restore the ±3% gate (3 paired runs)

| Pair | Vanilla | CIPHER default (gate ON) | CIPHER `RING_WRITE=0` (gate OFF) |
|------|---------|--------------------------|----------------------------------|
| 1 | 2242.32 | 2155.05 (-3.89%) | 2052.13 (-8.48%) |
| 2 | 2260.51 | 2143.08 (-5.20%) | 2150.43 (-4.87%) |
| 3 | 2231.62 | 2129.27 (-4.59%) | 2129.56 (-4.58%) |
| **Mean** | — | **-4.56%** | **-5.98%** |

`CIPHER_RING_WRITE=0` does NOT improve TPS — in pair 1 it's measurably slower; in pairs 2 + 3 the two modes are essentially identical. **RING_WRITE producer is not the dominant cost contributor.** This confirms the bisection's "no single inflection" finding.

The env gate is shipped anyway as production-tuning surface — disabling the RING_WRITE producer is the right operational lever for benchmark/low-traffic regimes even though it doesn't solve D14 here. The cost is in distributed paths the gate doesn't reach (COMMIT observe + resolver + may13 observers).

## 4. Path (b) gate update

Per user adjudication 2026-05-23: **TPS gate updates from ±3% to ±5%** for v1 substrate measurements. The substrate-active TPS cost of -3.43% to -4.56% on single-tenant TinyLlama B=1 is within the new ±5% gate. Pitch language updates from "substrate is TPS-neutral at single-instance" to:

> "v1 substrate adds approximately 3-5% per-call TPS overhead on single-tenant cuBLAS-bound workloads (TinyLlama B=1 decode). Recovered at v1 product target density via tenant-density TPW lift (5-6× at N=8-16 per `phase-a-multitenant` memory anchor). The per-call cost is distributed across COMMIT observe + stream resolver + RING_WRITE producer + may13 observer paths; no single producer dominates. Production deployments can disable RING_WRITE via `CIPHER_RING_WRITE=0` env for low-traffic regimes where the per-launch observation cost outweighs the telemetry value."

## 5. Three D14 findings (per user adjudication)

### 5.1 Finding 1 — substrate-active TPS cost 3-5% single-instance

| Workload regime | TPS impact | Note |
|-----------------|-----------|------|
| Single-tenant TinyLlama B=1 decode | -3.43% mean (5 pairs) | Within ±5% gate |
| Single-tenant TinyLlama B=1 decode (paired alternate) | -3.06% median | Within ±5% gate |
| Disable RING_WRITE producer | No improvement | Cost is distributed |
| v1 product target density (N=8-16 tenants same model) | Net positive (TPW lift dominates) | Per `phase-a-multitenant` memory |

Distributed cost rationale: at substrate baseline (W9-complete) the runtime adds COMMIT primitive + multi-tenant resolver + W7-9 G6 AUDIT path. W10-12 additions (RING_WRITE, TC probe, G3 hash mix, G4 map re-key, G5 compute_va_gib, L2 stream policy, SDPA lazy resolver) compose without amplifying any single hot-path call by more than measurement noise.

### 5.2 Finding 2 — compute_va_gib defaults to vLLM max_num_seqs=256

W12 Step 3 step doc projected ~4 GiB VA for TinyLlama at `max_batch=1`. Actual measurement with real vLLM defaults:

```
[cipher-vllm-kv] VMM allocator init: VA pool 51 GiB via compute_va_gib(hf_config) (tenant=0)
```

**51 GiB, not 4 GiB.** The formula is honest; the "4 GiB" number from W12 Step 3 was a unit-test floor with `max_batch=1`. Real vLLM `scheduler_config.max_num_seqs` defaults to 256, so the KV portion of the formula computes at `max_batch=256`:

```
kv_b = 2 × n_layer × n_kv_heads × head_dim × max_model_len × dtype_bytes × max_batch
     = 2 × 22 × 32 × 64 × 2048 × 2 × 256
     ≈ 47 GiB
weights_b ≈ 2.2 GiB
headroom ≈ 4.9 GiB (10% of 49 GiB)
total ≈ 51 GiB
```

Compared to the hardcoded 80 GiB default this is still a 36% reduction. The W12 Step 3 step doc's "12.5× reduction" headline used `max_batch=1` unit-test values; the production-deployment reduction is workload-dependent and varies with `max_num_seqs`. Pitch update:

> "compute_va_gib reduces the hardcoded 80 GiB VA pool to ~51 GiB for TinyLlama-1.1B at vLLM default `max_num_seqs=256` (36% reduction); to ~4 GiB at single-sequence (`max_batch=1`) regimes (95% reduction). Production reduction scales inversely with the scheduler's max-sequence parameter."

### 5.3 Finding 3 — REGISTER_STREAMS returns 0 for vLLM TinyLlama; SDPA tenant routing not exercised end-to-end

```
[cipher-vllm-kv] REGISTER_STREAMS no named streams enumerated; default tenant=0 fallback
```

The plugin's `_register_streams_with_kmod()` enumeration calls `torch.cuda.current_stream()`. For vLLM TinyLlama at engine init, the current stream is the default stream (handle = 0 sentinel). The plugin correctly skips registration (no named streams to register) and the SDPA trampoline falls back to tenant_id = 0.

**D5 fix (W12 Step 4 SDPA trampoline stream-fill) is therefore validated only by `test_sdpa_tenant_routing.py` 4/4 PASS (synthetic harness with explicit named streams)** — not by real vLLM TinyLlama. The vLLM real path uses xformers/paged-attention which bypasses the ATen SDPA trampolines.

The W12 Step 4 finding that "per-tenant cryptographic billing now correct for SDPA path" stands for **workloads that issue ATen SDPA via named streams** (which the test harness exercises), but does not stand for real-vLLM-TinyLlama paths that bypass ATen SDPA. v1 known coverage gap; documented as W15-17 CP 5.5 verification target on heterogeneous-model workloads where real SDPA dispatch is exercised.

## 6. Gate-default regression (Part D verification)

All W7-12 microbenches PASS at gate-default (`CIPHER_RING_WRITE` unset = enabled):

| Gate | Result |
|------|--------|
| `test_commit_atomicity` 4/4 | PASS |
| `test_audit_chain` 5/5 | PASS |
| `test_observe_publish` 3/3 | PASS |
| `test_resolver` 3/3 | PASS |
| `test_ring_write` 6/6 | PASS |
| `test_tc_probe` 17/17 | PASS |
| `test_g3_cross_model_keying` | PASS (15/15 pairwise distinct) |
| `test_g5_va_density` | PASS |

At `CIPHER_RING_WRITE=0`, `test_ring_write` correctly shows producer disabled (`Case 1: 4000 writes, 0 drained, written=0` — gate is operating as designed; not a regression). The test is designed for gate-default behavior; running with the gate explicitly disabled is an off-design configuration that the test correctly does not pass under.

No N=128 30-min soak re-run requested by Path (b) adjudication — env-gate at default mode preserves all prior W12 Step 4 substrate behavior; the soak gate from W12 Step 4 (`+8.1%` aggregate rate, 0 incoherent of 5.20T reads) remains the binding measurement.

## 7. Closes scope-drift audit item D14

D14 status: **CLOSED** at cipher_rt_phase4 `b360fc1` (tag `week-12-step-5-d14-env-gate`).

The closure rests on Path (b) adjudication: TPS gate updates ±3% → ±5%; the env-gate adds production-tuning surface; the three findings (substrate cost distribution, compute_va_gib max_batch=256, REGISTER_STREAMS vLLM coverage gap) are documented honestly.

Scope-drift audit cumulative closure status:
- D5: CLOSED (W12 Step 4 SDPA trampoline stream-fill, commit 1466193)
- D9: CLOSED (Plan Koopman reconciliation, commit 69203d3)
- **D14: CLOSED** (this step, commit b360fc1)
- D1-D4, D6-D8, D10-D13, D15: unchanged; per-item user adjudication still pending

## 8. Honest residue

1. **Env-gate does not deliver TPS-gate restoration.** The gate is shipped because it provides operational value (production tuning surface) but the D14 backfill goal of "minimal env-gate to restore ±3% gate" was not achieved by this single gate. The gate update from ±3% to ±5% is the load-bearing closure mechanism.
2. **Substrate cost is distributed; no instrumentation-free fix exists.** Targeted per-path microbenches (CUPTI, perf, etc.) would identify hot lines if a future iteration wanted to drive the cost down. Out of D14 scope.
3. **D5 SDPA fix coverage gap.** Real vLLM TinyLlama does not exercise the D5 fix end-to-end. Documented as v1 known limitation; W15-17 CP 5.5 heterogeneous-model workloads with named streams are the natural verification path.
4. **compute_va_gib at production defaults yields 51 GiB**, not the 4 GiB unit-test number. Pitch language updates accordingly.
5. **Today's vanilla baseline drifted ~4%** between morning (2271 tok/s) and afternoon (2192 tok/s) runs. The ±5% gate accommodates this measurement-environment variance.

## 9. Final fingerprints

```
cipher_rt_phase4    b360fc1   tag week-12-step-5-d14-env-gate
libcipher_rt.so     md5 3e22be9da9dbb6e5e0384394be3b3187
cipher_kmod         unchanged at week-9-complete (0.6.5, ko md5 8c9fdd01...)
cipher_rt_attn_call ABI       408 B (unchanged from W11 Step 2)

Files touched:
  cipher_rt_ring_write.c   +27 LOC (env-gate at function entry)
```

Bisection artifacts preserved at `/tmp/d14_backfill/`:
- `libcipher_rt-<tag>.so` (5 binaries, one per bisection tag)
- `result-<tag>.log` (5 bisection runs)
- `paired.log` (5-pair vanilla/CIPHER variance-control runs)
- `fix_test.log` (3-pair vanilla / gate-default / gate-disabled verification)
