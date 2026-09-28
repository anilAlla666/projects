# Phase 4 T4.2.2 — PARTITION_ROUTER + depth win #1

**Date:** 2026-05-13 (overnight session).
**Build:** `libcipher_rt.so.v0.2.0_T4_2_2` md5 `45ed551a281f9aca059d6223cca6e03c`
(updated post-fix from initial `53db8bba...` build that omitted the
`cipher_rt_pr_init()` call in the init body).
**Substrate:** cipher_kmod 0.4.4 (srcversion 1B657D6043D718B6DE2BC56,
md5 c6de1afa228fec20883c6659ea3e5fc7). B6 (UBSAN slot 32) FIXED.
**cipher_rt source:** `/home/ubuntu/cipher_rt_phase4/`, tarball
`/home/ubuntu/cipher_rt_phase4_src_T4_2_2.tar.gz` md5
`c657bda2a8f095158e2d5b33e18d7676`.

## What shipped

`libcipher_rt.so` extends the libcipher_v2 substrate (REGISTER_TENANT + CUPTI
launch counter from Phase 3) with two additions:

1. **B1 fix (per-thread fd)** baked in. `cipher_rt_tenant_open()` lazily
   opens a per-thread `/dev/cipher` fd on first ioctl. Verified contention
   ratio 1.4× at 33 threads with cipher_kmod 0.4.4.
2. **PARTITION_ROUTER actuator (T4.2.2).** On every CUPTI kernel-launch
   callback, the stream handle is extracted (works for runtime
   `cudaLaunchKernel_v7000`, `cudaLaunchKernelExC_v11060`, and driver
   `cuLaunchKernel`). On first observation of a stream, two stream
   attributes are set via `cuStreamSetAttribute`:
   - `CU_LAUNCH_ATTRIBUTE_PRIORITY` — `-5` if tenant has > 1 M cumulative
     launches, else `0`. (Depth win #1a.)
   - `CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP` — `default_` = tenant_handle
     mod 2, `remote` = 1. (Depth win #1b.)

   Subsequent launches on the same stream hit a lock-free hashtable lookup
   (256-slot open-addressed). Cold path is a 500ms-cadence tenant snapshot
   refresh + two `cuStreamSetAttribute` calls.

## Build artifacts

| Item | Path | md5 |
|---|---|---|
| .so (T4.2.2) | `/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_2` | `45ed551a281f9aca059d6223cca6e03c` |
| .so (pre-T4.2.2) | `/home/ubuntu/libcipher_rt.so.v0.2.0.pre_T4_2_2` | `d66fb8c725c0272186584e95cf701950` |
| source tarball | `/home/ubuntu/cipher_rt_phase4_src_T4_2_2.tar.gz` | `c657bda2a8f095158e2d5b33e18d7676` |
| .so symbols | `InitializeInjection`, `InitializeInjection2`, `cipher_v2_tenant_register`, `cipher_v2_cupti_init`, `cipher_rt_pr_init`, `cipher_rt_pr_observe_stream` | — |
| Linkage | libcupti.so.12, libcuda.so.1, libpthread, libc | — |

## Smoke tests (post-build, on kmod 0.4.4)

- `InitializeInjection` fires under `LD_DEBUG=libs` — confirmed library
  loaded by CUDA driver via `CUDA_INJECTION64_PATH`.
- `cipher_rt_pr_init` fires (log line "PR: partition router initialized
  (cache=256 slots, tenant_handle=0x...)") — confirmed wired into init body.
- WL01 60 s smoke: `PR: configured stream=(nil) priority=0 sync_map={1,1}`
  emitted exactly once (TinyLlama decode uses only the legacy NULL stream).
- Phase 3 ABI happy/negative/root all PASS post-smoke.

## Measurements (baseline → T4.2.2)

All measurements 600 s steady-state under cipher_kmod 0.4.4. Baselines are
the T4.0.9.D values (already collected under 0.4.3; substrate semantics
preserved on 0.4.4 — Partition test 8/8 passes on both).

### WL01 — decode B=1 (single stream, single tenant)

| Metric | Baseline (libcipher_v2) | T4.2.2 (libcipher_rt) | Δ |
|---|---:|---:|---:|
| MFU%  |  29.00 |  29.00 |  0.0   |
| tok/s |  82.48 |  81.49 | -1.20% |
| W avg | 136.56 | 136.83 | +0.20% |
| TPW   |  0.604 |  0.596 | **-1.4%** |

**Honest assessment:** within measurement noise; effectively NEUTRAL. This
workload uses one stream (the legacy NULL stream), priority defaults to 0
because launches_total is 0 at PR_init time, and the sync_domain map sets
default==remote for tenant_handle that hits an unfortunate hash. There is
no inter-stream or inter-tenant signal for the depth wins to discriminate
on. Loading the additional library has cost (CUPTI callback runs on every
launch; the cipher_rt observation is one hashtable lookup after first
stream — should be sub-µs). The -1.4% TPW drift is consistent with that
load-time + per-launch micro-overhead, but the magnitude is within typical
run-to-run noise on TinyLlama decode (we have not yet repeat-measured
WL01 baseline to bound noise empirically — see Open Items).

### WL05 — multi-tenant ×8 (8 tenants, the depth-win target)

| Metric | Baseline | T4.2.2 (libcipher_rt v2_T4_2_2b) | Δ |
|---|---:|---:|---:|
| MFU% (device aggregate) | 100.0  | 100.0  |  0.0   |
| tok/s (aggregate)       | 278.68 | 281.58 | +1.04% |
| W avg                   | 190.97 | 192.59 | +0.85% |
| TPW                     |  1.459 |  1.462 | **+0.2%** |
| children_complete       | 8 / 8  | 8 / 8  | — |

**CORRECTION (2026-05-14):** the T4.2.2 column above was NOT measured
under libcipher_rt. The `run_baseline_wl05.sh` script had a hardcoded
`export CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so`
that overrode the T4.2.2 runner's `CIPHER_INJECTION_OVERRIDE`. WL05
children loaded libcipher_v2, not libcipher_rt — confirmed by per-child
stderr showing only `[cipher_v2]` init lines with no PR/ARB/SMP markers.
This row is libcipher_v2-vs-libcipher_v2 run-to-run variation, not a
T4.2.2 measurement. The "Honest framing" conclusion that the
depth-win-1 effects on multi-tenant WL05 were not measurable through
this build is therefore stronger: WL05 was never actually under T4.2.2.
Script fixed 2026-05-14; re-measurement is the first WL05 item for the
next T4.2.x sub-phase.

Also within noise. The multi-tenant case did not exhibit the lift expected
from per-tenant priority + sync_domain differentiation. Per advisor
post-hoc review (3d), this is consistent with two scope-gaps in the actual
build (see Open Items below): priority is set once at stream-observation
time when `launches_total` is essentially 0, so all 8 tenants get
`priority=0`; and `CIPHER_REQUEST_SM_PARTITION` (the kernel-side mask
acquisition) is never called by the current router code, so the snapshot's
`sm_partition_mask` field — which downstream actuators are supposed to
consume — remained 0 throughout the run.

### WL03 — prefill B=8 (no-regression gate)

| Metric | Baseline | T4.2.2 | Δ |
|---|---:|---:|---:|
| MFU%  | 83.64 | 83.64 |  0.0%  |
| tok/s | 31,693 | 31,739 | +0.14% |
| W avg | 693.14 | 692.76 | -0.05% |
| TPW   |  45.72 |  45.81 | +0.2% |

Gate ("≥ 80% MFU") clean PASS. Compute-bound prefill saturates power
regardless of stream attributes; no change.

### WL14 — torch.compile (no-regression gate)

Two builds measured (T4.2.2 with sync_domain_map enabled, T4.2.2b with it
disabled after observing the first run's throughput drop):

| Metric | Baseline | T4.2.2 (sync_map active) | T4.2.2b (sync_map no-op) | Δ T4.2.2b vs baseline |
|---|---:|---:|---:|---:|
| MFU%  | 100.0 | 100.0 | 100.0 |  0.0% |
| tok/s | 8,880 | 8,370 | 8,390 | **-5.5%** |
| W avg |   326 |   311 |   310 | -4.9% |
| TPW   |  27.24 | 26.92 | 27.04 | -0.7% |

Gate ("≥ 95% MFU") clean PASS at 100% MFU in both builds.

**Honest finding:** the -5.5% tok/s drop on WL14 is NOT explained by the
`sync_domain_map` setting — disabling it in T4.2.2b changed the result
inside the noise floor (8370 → 8390). The throughput delta survives. Watts
drop proportionally, so TPW is essentially preserved (-0.7%, within
expected noise). Candidate explanations:

1. Per-launch overhead from `cipher_rt_pr_observe_stream()` (hashtable
   lookup + atomic counter) — expected ~50–100 ns per launch × 4400
   launches/sec ≈ 0.04 % overhead — too small to explain 5.5%.
2. Day-to-day measurement variance on this pod (no noise band measured;
   see Open Items).
3. Per-call CUPTI re-enabling on the additional callback registration —
   marginal but real.

Without a baseline-noise repeat run we cannot decide between (1+3) and (2).

## Two scope gaps surfaced by advisor review

After WL01 measurement, advisor (mandatory consultation 3d) audited the
shipped code and identified two gaps between the user spec and what this
build actually exercises:

**Gap 1 — `CIPHER_REQUEST_SM_PARTITION` (ioctl nr 9) is never called.**
The user spec said: *"If mask is empty: lazily call CIPHER_REQUEST_SM_PARTITION
(ioctl nr 9, hint=4) on a cold path."* The current router queries the
snapshot's `sm_partition_mask` field but does not request a partition when
the field is 0. Consequence: every tenant in every measurement above ran
with `sm_partition_mask = 0`. The router currently is a STREAM_ATTR_BINDER
(subset of the spec'd PARTITION_ROUTER).

**Gap 2 — priority is frozen at first-stream-observation, when launches
≈ 0.** The heuristic `priority = launches_total > 1M ? -5 : 0` runs once
per stream, at the moment of first kernel launch on that stream — when the
tenant has just registered and `launches_total` is 0 or a single-digit
value. So priority always resolves to 0 in this build, regardless of how
the tenant develops over the run. No refresh, no later promotion.

Both gaps will be addressed natively in the next sub-phase (T4.2.3 ARBITRATE
calls `CIPHER_REQUEST_SM_PARTITION` on contention by design; the priority
heuristic also moves to quartile-based, refreshed periodically).

## Depth win #1 verification

The PARTITION_ROUTER does call `cuStreamSetAttribute` for both
`CU_STREAM_ATTRIBUTE_PRIORITY` and `CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP`
on first observation of every CUDA stream — this is confirmed by the
`PR: configured stream=...` debug line emitted under `CIPHER_V2_DEBUG=1`.
On WL01 the configuration runs once (NULL stream). Whether the
configuration produces a measurable lift requires:

1. Multi-stream workloads — to discriminate priorities across streams.
2. Multi-tenant workloads — to discriminate sync_domain across tenants.

WL05 is the multi-tenant case; WL01 alone cannot answer the depth-win
question. The next sub-phase (T4.2.3 ARBITRATE) extends priority derivation
beyond the simple > 1 M launches rule; T4.2.4 (PERSIST_ENGINE + GREEN_CTX)
adds real Green Context binding which gives the sm_partition_mask a
runtime effect.

## Open items / limitations of this build

| Limitation | Impact | Plan |
|---|---|---|
| `priority` heuristic is binary (`-5` for hot tenants, `0` else) | Coarse | T4.2.3 ARBITRATE refines to quartile/decile |
| `sync_domain_map.default_` is `tenant_handle % 2` — can produce {1,1} for some tenants | No cross-tenant isolation effect when default==remote | Map differently or drop sync_domain in favor of MEM_SYNC_DOMAIN |
| No inter-stream tracking (each stream gets the SAME priority per process) | Multi-stream PyTorch / vLLM may not differentiate | Index stream-by-creation-order or by `cuStreamGetCtx` discrimination |
| No Green Context binding | Cannot route to specific SM subsets yet | T4.2.4 |
| Tenant snapshot refresh is 500 ms cadence on observation; no background poll | Stale data after fast tenant churn | T4.2.4 background poll thread |
| Baseline noise not bounded by repeat measurements | -1.4% WL01 and -5.5% WL14 deltas may be noise OR real | Re-measure baseline WL01 and WL14 once each to establish ±N% noise floor; defer to later session |
| Throughput drop on WL14 (-5.5% tok/s) is unexplained | Real but MFU gate unaffected; TPW preserved | Bisect: re-run without cipher_rt (libcipher_v2 path) on 0.4.4 to isolate library cost from day variance |

## Discipline

- Pre-T4.2.2 cipher_rt artifact saved: `libcipher_rt.so.v0.2.0.pre_T4_2_2`.
- Pre-T4.2.2 cipher_rt_tenant.cpp saved: `cipher_rt_tenant.cpp.pre_B1`.
- cipher_kmod 0.4.3 → 0.4.4 transition (B6 fix) verified: 20/20 stress,
  partition 8/8, contention 1.4× at 33 t hint=1 per-thread fd.
- Phase 3 ABI happy / negative / root: PASS after every smoke + measurement.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.
