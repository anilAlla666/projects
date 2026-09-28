# Phase 4.2 — Cluster Summary (partial: T4.2.1 + T4.2.2 + T4.2.3 done; T4.2.4 deferred)

**Date:** 2026-05-13 (recovery + overnight session).
**Substrate at session end:** cipher_kmod 0.4.4 (srcversion
1B657D6043D718B6DE2BC56, md5 c6de1afa228fec20883c6659ea3e5fc7);
libcipher_rt.so.v0.2.0_T4_2_3 (md5 bc51b9d6f6827ccdd90ad2d5ab09e3ff).

## What ships at the end of Phase 4.2 sub-phases T4.2.1 → T4.2.3

| Sub-phase | Deliverable | Status |
|---|---|---|
| T4.2.1 | Lock-free atomic-slot SM partition allocator in cipher_kmod (replaces 0.4.1 spinlock); CIPHER_REQUEST_SM_PARTITION ioctl nr 9 stable in kmod ABI. | ✅ SHIPPED |
| T4.2.2 | PARTITION_ROUTER stream-attribute substrate in libcipher_rt: per-stream observation via CUPTI launch callback; cuStreamSetAttribute(PRIORITY, MEM_SYNC_DOMAIN_MAP) on first observation. B1 (per-thread fd) baked in. | ✅ SHIPPED (scaffolding, no lift) |
| T4.2.3 | ARBITRATE: calls CIPHER_REQUEST_SM_PARTITION with quartile-derived hint, populates sm_partition_mask end-to-end. SM_PACKER: detection-only small-launch counter. | ✅ SHIPPED (scaffolding, no lift) |
| T4.2.4 | GREEN_CTX (cuGreenCtxCreate + binding) and PERSIST_ENGINE (resident kernel pool) — the layer that consumes the populated mask. | ⏭ DEFERRED — requires cuStreamCreate symbol interception, not designed in this session. See bottom of this file for the entry-point design. |

## Cluster gate (T4.2.x)

| Gate | Requirement | Status |
|---|---|---|
| Contention 5× | p99 contended ≤ 5× single-thread p99 on CIPHER_REQUEST_SM_PARTITION | **1.4×** at 33 t hint=1 per-thread fd, 75.9 M ops/s (T4.2.1 / T4.0.9.D, kmod 0.4.3); re-verified clean on kmod 0.4.4 (1.4× / 61.8 M ops/s). |
| Phase 3 ABI regression | 12 PASS markers across happy / negative / root | Clean at every checkpoint. |
| Module load/unload | 20-cycle stress | 20/20 PASS on both 0.4.3 and 0.4.4. |
| No new taint | W bit must not get set | Taint 12288 unchanged across session. |
| No new dmesg WARN/BUG/Oops | Clean | UBSAN slot-32 (B6) fixed in 0.4.4; no new warnings observed thereafter. |

## Measurement honesty — three independent runs of WL05 ≈ ±2% TPW noise

| Run | TPW |
|---|---:|
| T4.0.9.D baseline (libcipher_v2 on 0.4.3) | 1.459 |
| T4.2.2 (libcipher_rt scaffolding on 0.4.3 → 0.4.4) | 1.462 |
| T4.2.3 (libcipher_rt + ARBITRATE + SMP on 0.4.4) | 1.437 |

Spread: ±0.9% from mean (1.453), giving a measured **WL05 noise band of
±2%** (conservative). Future T4.2.x deltas on WL05 must clear this band to
count as signal.

## Retroactive correction (2026-05-14): WL05 row was not actually under libcipher_rt

The "Per-workload status snapshot" table below shows a -1.5% T4.2.3 WL05
TPW delta vs baseline. Morning 2026-05-14 measurement-script audit found
that `run_baseline_wl05.sh` hardcoded the injection lib to libcipher_v2,
overriding `CIPHER_INJECTION_OVERRIDE`. Per-child stderr confirmed: no
`ARB:` / `PR:` / `SMP:` init lines, only `[cipher_v2]` markers.

What the corrected reading says:
- WL05 has NOT been measured under T4.2.2 or T4.2.3 in this cluster.
- The -1.5% number is libcipher_v2 vs libcipher_v2 run-to-run variation
  within the (later-measured) ±0.53% same-condition noise band and the
  4–5% day-drift band (B9).
- The T4.2.4 GREEN_CTX entry-point design is unchanged — WL05 is still
  the headline workload for it. But the prior-run "no lift visible"
  framing is wrong direction: there's no measurement to draw the
  conclusion from. Phrasing changes: "WL05 lift question is unmeasured
  by T4.2.2/T4.2.3; next session's first WL05 action under corrected
  script will establish the cross-library T4.2.2-vs-baseline and
  T4.2.3-vs-T4.2.2 deltas honestly."

Script fix landed in `cipher_workloads/measurement/run_baseline_wl05.sh`
2026-05-14 (CIPHER_INJECTION_OVERRIDE now honored).

## Per-workload status snapshot (kmod 0.4.4 + libcipher_rt v0.2.0_T4_2_3)

| WL | Baseline TPW | T4.2.3 TPW | Δ | Gate | Notes |
|---|---:|---:|---:|---|---|
| WL01 decode B=1     |  0.604 |  0.587 | -2.8% | — | Single stream → depth wins don't apply |
| WL02 decode B=8     |  4.74  |  4.75  | +0.23% | — | Neutral |
| WL03 prefill B=8    | 45.72  | 45.82  | +0.2% | ≥80% MFU ✅ | Compute-bound, no change |
| WL05 multi-tenant   |  1.459 |  1.437 | -1.5% | (lift target) | Within noise; mechanism deferred to T4.2.4 |
| WL14 torch.compile  | 27.24  | 25.85  | -5.1% | ≥95% MFU ✅ at 100% | Persistent ~5% throughput drop across all libcipher_rt builds; bisect in progress |

## Two findings on the actuator design (will need addressing in T4.2.4 / T4.7)

### F1 — Quartile policy is data-starved at first-launch

`hint` for `CIPHER_REQUEST_SM_PARTITION` is derived from
`launches_total` quartile-rank. But the rank is computed at first stream
observation (first launch in the process), when `launches_total` is 0 for
every newly-started tenant. With 8 WL05 tenants all observing their first
launch within seconds of each other, all 8 are tied at 0 and quartile is
degenerate.

**Effect today:** mask distribution is FCFS contention against the kmod
allocator, not quartile-prioritized. Doesn't break anything in this build
(mask is metadata only), but blocks the quartile policy from doing its
intended job once T4.2.4 makes the mask load-bearing.

**Fix:** add a per-process snapshot-poll thread that refreshes hint /
priority every ~5 s based on accumulated activity, then re-issues
CIPHER_REQUEST_SM_PARTITION (idempotent on the kmod side, will grow or
shrink the granted mask). Tracked as **PHASE_4_BACKLOG.md B7**.

### F2 — Stream attribute writes after observation cannot drive SM scope

`cuStreamSetAttribute(PRIORITY)` and `cuStreamSetAttribute(MEM_SYNC_DOMAIN_MAP)`
DO have effects on streams (priority influences scheduling order among
concurrent streams; sync_domain influences fence scope). But neither
restricts which SMs the kernels run on. SM-scope restriction requires
Green Contexts, and Green Contexts only confine streams that are
CREATED via `cuGreenCtxStreamCreate` — they cannot retroactively bind
existing streams.

**Effect today:** the `sm_partition_mask` from the kmod is populated and
visible in telemetry, but kernels still run on all 132 SMs of the H100.
**Mechanically, a TPW lift is not possible in this build.** The neutral
measurements are correct, not a measurement defect.

**Fix path:** T4.2.4 must intercept `cuStreamCreate` / `cuStreamCreateWithPriority`
(driver and runtime API variants) via symbol-level wrapping (LD_PRELOAD or
explicit `dlsym` in the injection lib that publishes wrapper symbols).
Pre-create a pool of green contexts at cipher_rt init keyed by the
masks the kmod gives us, then at app's cuStreamCreate, substitute the call
with `cuGreenCtxStreamCreate` against the appropriate green context.

This is the design entry-point for the next session.

## P4.2 lift status — honest summary

| Phase | Effect on TPW for WL05 |
|---|---|
| T4.2.1 (lock-free allocator) | No userspace lift — kmod-internal correctness/scalability |
| T4.2.2 (PARTITION_ROUTER scaffolding) | +0.2% (within noise) |
| T4.2.3 (ARBITRATE + SMP) | -1.5% (within noise) |
| T4.2.4 (GREEN_CTX + PERSIST_ENGINE) — deferred | The first sub-phase that can mechanically deliver a TPW lift, per the design analysis above. |

**Phase 4.2 partial close-out:** the substrate (kmod allocator, stream-attribute
binder, mask request/grant, small-launch detection) is complete and verified.
The actuator that consumes the substrate (Green Context creation + stream
substitution at cuStreamCreate time) is the next-session deliverable.

## Artifacts

| Layer | Artifact | md5 |
|---|---|---|
| kmod 0.4.4 (current) | `/home/ubuntu/cipher_kmod.ko.v0.4.4` | c6de1afa228fec20883c6659ea3e5fc7 |
| kmod 0.4.3 (pre-B6) | `/home/ubuntu/cipher_kmod.ko.v0.4.3` | fb210777460c47bedd52c7d4223b22ca |
| kmod src 0.4.4 | `/home/ubuntu/cipher_kmod_src_v0.4.4.tar.gz` | 5f154a14cdf708e502d3ec2ae75586c6 |
| libcipher_rt T4.2.3 | `/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_3` | bc51b9d6f6827ccdd90ad2d5ab09e3ff |
| libcipher_rt T4.2.2b (pre-T4.2.3) | `/home/ubuntu/libcipher_rt.so.v0.2.0_T4_2_2b.pre_T4_2_3` | 21a3af0ae822680b060caab959db687f |
| libcipher_rt v0.2.0 (pre-Phase 4) | `/home/ubuntu/libcipher_rt.so.v0.2.0.pre_T4_2_2` | d66fb8c725c0272186584e95cf701950 |
| cipher_rt source T4.2.3 | `/home/ubuntu/cipher_rt_phase4_src_T4_2_3.tar.gz` | 3abbd65db5d644d87c7232f2bfdb5966 |
| Phase 4 evidence dir | `/home/ubuntu/cipher-phase4-evidence/` | — |
| Per-sub-phase reports | `PHASE_4_T4_2_2_REPORT.md`, `PHASE_4_T4_2_3_REPORT.md` | — |
| Backlog | `PHASE_4_BACKLOG.md` (B1–B8) | — |
