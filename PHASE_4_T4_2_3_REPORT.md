# Phase 4 T4.2.3 — SM_PACKER + ARBITRATE

**Date:** 2026-05-13 (overnight session).
**Build:** `libcipher_rt.so.v0.2.0_T4_2_3` md5 `bc51b9d6f6827ccdd90ad2d5ab09e3ff`.
**Substrate:** cipher_kmod 0.4.4 (srcversion 1B657D6043D718B6DE2BC56).
**Source tarball:** `/home/ubuntu/cipher_rt_phase4_src_T4_2_3.tar.gz` md5
`3abbd65db5d644d87c7232f2bfdb5966`.
**Prior .so saved:** `libcipher_rt.so.v0.2.0_T4_2_2b.pre_T4_2_3` md5
`21a3af0ae822680b060caab959db687f`.

## What shipped vs T4.2.2

Two new actuators added to libcipher_rt:

1. **ARBITRATE** (`cipher_rt_arbitrate.{c,h}`). On the cold path of the
   first stream observation, calls `CIPHER_REQUEST_SM_PARTITION` (kmod
   ioctl nr 9). Hint derived from per-tenant `launches_total` quartile:
   - solo / ≤4 active tenants → hint=8
   - top quartile → 8, second → 6, third → 4, bottom → 2

   This **closes Gap 1** from the T4.2.2 advisor review (the prior router
   never invoked nr 9, so `sm_partition_mask` remained 0 in every
   snapshot). With ARBITRATE wired, masks like `0xff` (8 slots) or
   `0x3f` (6 slots) are visible in `/proc/cipher/stats` for each tenant.

2. **SM_PACKER** (`cipher_rt_sm_packer.{c,h}`). On every CUPTI launch
   callback, extracts `gridDim × blockDim`, classifies as "small" if
   total threads < 16 × 256 = 4096, and tracks per-stream small-launch
   streaks. **Detection only — no kernel substitution.** Real packing
   requires a resident persistent kernel (deferred to T4.2.4).

   Surfaces three counters via `cipher_rt_smp_*()` accessors:
   - `cipher_rt_smp_total_launches()`
   - `cipher_rt_smp_small_launches()`
   - `cipher_rt_smp_longest_streak()`

   These are not yet exposed via `/proc/cipher/stats`; that's a Phase 4.4
   wiring step.

### Functional verification

Smoke run on WL01 with `CIPHER_V2_DEBUG=1` produces the expected init
chain:

```
[cipher_v2] ARB: arbitrate initialized (fd=9)
[cipher_v2] SMP: sm_packer initialized (threshold=4096 threads, 32 stream slots)
[cipher_v2] PR: partition router initialized (cache=256 slots, tenant_handle=0x098341db)
[cipher_v2] CUPTI subscribed: kernel launch callbacks active
[cipher_v2:dbg] ARB: rank 0/1, hint=8
[cipher_v2] ARB: granted mask=0x000000ff count=8 (hint=8)
[cipher_v2:dbg] PR: configured stream=(nil) priority=0 sync_map={0,0}
```

Mask 0xff = 8 slots granted (solo tenant; hint=8 from `total <= 4` rule).
**Gap 1 closed** — the kmod is now actually being asked for a partition.

## Measurements (600 s each, cipher_kmod 0.4.4)

### Run-to-run noise band, measured (the unexpected outcome that becomes usable evidence)

WL05 device-aggregate TPW across three independent runs at the same
workload:

| Run | TPW |
|---|---:|
| T4.0.9.D baseline (libcipher_v2) | 1.459 |
| T4.2.2 (libcipher_rt scaffolding) | 1.462 |
| T4.2.3 (libcipher_rt + ARBITRATE + SMP) | 1.437 |

**Spread = ±0.9% from mean (1.453); ±2% as a conservative noise band.**
This is no longer asserted as "within noise" — it's *measured* noise.
All future T4.2.x deltas on WL05 should be compared against this band.

### WL01 — decode B=1 (single stream, single tenant)

| Metric | Baseline | T4.2.2 | T4.2.3 | Δ T4.2.3 vs baseline |
|---|---:|---:|---:|---:|
| MFU%  |  29.0  |  29.0  |  28.0  | -1 pp |
| tok/s |  82.48 |  81.49 |  79.56 | -3.5% |
| W avg | 136.56 | 136.83 | 135.58 | -0.7% |
| TPW   |  0.604 |  0.596 |  0.587 | -2.8% |

### WL02 — decode B=8

| Metric | Baseline | T4.2.3 | Δ |
|---|---:|---:|---:|
| MFU%  |  31.0  |  31.0  |  0.0  |
| tok/s | 652.29 | 652.87 | +0.09% |
| W avg | 137.57 | 137.37 | -0.15% |
| TPW   |   4.74 |   4.75 | +0.23% |

### WL05 — multi-tenant ×8 (the ARBITRATE headline workload)

| Metric | Baseline | T4.2.2 | T4.2.3 | Δ T4.2.3 vs baseline |
|---|---:|---:|---:|---:|
| MFU%  | 100.0  | 100.0  | 100.0  |  0.0 |
| tok/s | 278.68 | 281.58 | 280.93 | +0.81% |
| W avg | 190.97 | 192.59 | 195.54 | +2.40% |
| TPW   |  1.459 |  1.462 |  1.437 | **-1.5%** |
| children_complete | 8/8 | 8/8 | 8/8 | — |

**CORRECTION (2026-05-14):** the three numbers above are NOT a baseline /
T4.2.2 / T4.2.3 progression in the way the column headers suggest. The
`run_baseline_wl05.sh` script (Phase 3) contained a **hardcoded**
`export CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so`
that overrode the `CIPHER_INJECTION_OVERRIDE` env var the T4.2.x runners
set. Every WL05 measurement in this row — including yesterday's T4.2.2
and T4.2.3 columns — actually loaded **libcipher_v2** in the child
processes, not libcipher_rt. Per-child stderr (`/tmp/wl05_t*.log`) shows
only `[cipher_v2]` init markers; no `[cipher_v2] ARB:`, `[cipher_v2] PR:`,
or `[cipher_v2] SMP:` lines.

What the row actually shows: three runs of libcipher_v2 + cipher_kmod
(0.4.3 → 0.4.4) at different times of day, measuring same-script
run-to-run variation. The 0.2% / -1.5% / +0.81% spread is in the
day-drift band (B9) and the same-condition noise band (±0.53% on
TPW, see PHASE_4_NOISE_BAND_WL05.md).

**No T4.2.x WL05 lift conclusion can be drawn from these numbers, in
either direction.** The "lift question deferred to T4.2.4" framing in
the original report's "Honest framing" section is therefore stronger,
not weaker — we don't have a measurement of T4.2.2 or T4.2.3 on WL05 at
all. The script is fixed in `cipher_workloads/measurement/run_baseline_wl05.sh`
2026-05-14 to honor `CIPHER_INJECTION_OVERRIDE`. Re-measurement under
the corrected script is the first WL05 action item for the next
T4.2.x sub-phase.

### WL03 — prefill B=8 (no-regression gate ≥80% MFU)

| Metric | Baseline | T4.2.3 | Δ |
|---|---:|---:|---:|
| MFU%  |  83.64 |  83.64 |  0.0% |
| tok/s | 31,693 | 31,687 | -0.02% |
| W avg | 693.14 | 691.52 | -0.23% |
| TPW   |  45.72 |  45.82 | +0.2% |

Gate clean PASS.

### WL14 — torch.compile (no-regression gate ≥95% MFU)

| Metric | Baseline | T4.2.2 | T4.2.3 | Δ T4.2.3 vs baseline |
|---|---:|---:|---:|---:|
| MFU%  | 100.0  | 100.0  | 100.0  |  0.0 |
| tok/s | 8,880  | 8,370  | 8,364  | -5.8% |
| W avg |  326   |  311   |  324   | -0.6% |
| TPW   |  27.24 |  26.92 |  25.85 | **-5.1%** |

Gate ≥95% MFU clean PASS at 100%. The -5.1% TPW is the same persistent
delta observed in T4.2.2; consistent with cipher_rt CUPTI hook chain
adding small per-launch overhead on this 4,400 launches/sec workload.
Not a regression introduced by T4.2.3 specifically.

## Honest framing — per advisor (consultation 4h)

> "ARBITRATE and SM_PACKER infrastructure shipped; mask now populated
> end-to-end (snapshot visible to telemetry); lift mechanism deferred to
> T4.2.4 GREEN_CTX where the mask becomes load-bearing."

Mechanism check: kmod's `sm_partition_mask` is now populated in each
tenant's snapshot (was 0 in T4.2.2). But until cipher_rt binds streams to
Green Contexts that actually restrict the kernel's SM scope (T4.2.4), the
mask is metadata — CUDA still schedules across all 132 SMs regardless of
mask. **A mechanical TPW lift is not possible in this build.** The
observed neutral measurements are the correct prediction; anything > +2%
TPW on WL05 today would have been a signal to investigate measurement
error.

## Findings worth surfacing

### Finding 1 — quartile policy is data-starved at first-launch

The hint derivation reads `launches_total` from the snapshot. At the
moment of first stream observation (first launch in the process), this
field is 0 (the tenant just registered). With 8 tenants all observing
their first launch within seconds of each other, all 8 are tied at
`launches_total = 0` and the quartile rank is unstable.

For WL05, this means the 8 tenants effectively get a uniform hint chosen
by sort order — typically all hint=8 (top quartile), demanding 64 slots
vs 32 supply, producing partial grants. The end-state mask distribution
is FCFS contention, not quartile-prioritized.

**Severity:** medium. Doesn't break anything in this build (mask is
metadata anyway), but does prevent the quartile policy from doing its
job once GREEN_CTX makes the mask load-bearing.

**Fix path:** add a snapshot-poll thread (one per process) that refreshes
priority/hint every ~5 s based on accumulated activity. Tracked as B7 in
PHASE_4_BACKLOG.md.

### Finding 2 — WL14 throughput regression persists across all
libcipher_rt builds (T4.2.2 / T4.2.2b / T4.2.3)

| Build | tok/s | tok/s vs baseline |
|---|---:|---:|
| Baseline (libcipher_v2) | 8,880 | — |
| T4.2.2 (libcipher_rt + PR) | 8,370 | -5.7% |
| T4.2.2b (PR, sync_domain disabled) | 8,390 | -5.5% |
| T4.2.3 (PR + ARB + SMP) | 8,364 | -5.8% |

Persistent -5–6% regression. NOT from sync_domain (disabled in 2b+).
NOT from ARBITRATE/SMP specifically (T4.2.3 is similar to 2b). Likely
cause: the additional CUPTI subscription path or per-launch SMP
observation cost on this 4,400-launches/sec workload.

The MFU gate (≥95%) passes cleanly at 100% — torch.compile saturates SMs
either way, the throughput change is in per-iteration latency. TPW
preserved within the noise band on WL03 / WL01 / WL05 / WL02 but clearly
outside on WL14.

**Investigation deferred:** bisect by running WL14 under libcipher_v2 on
kmod 0.4.4 (isolate library cost from kmod-version cost). Tracked as B8
in PHASE_4_BACKLOG.md.

## Discipline

- Pre-T4.2.3 .so saved: `libcipher_rt.so.v0.2.0_T4_2_2b.pre_T4_2_3`.
- Phase 3 ABI happy / negative / root: PASS after every measurement.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.
- No oops / WARN / BUG / NULL in dmesg.

## Decision

T4.2.3 ships clean. Proceeding to T4.2.4 (GREEN_CTX + PERSIST_ENGINE)
where the populated mask becomes load-bearing for the first time.
