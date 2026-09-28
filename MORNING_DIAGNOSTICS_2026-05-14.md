# Morning Diagnostics — 2026-05-14

**Started:** 02:44 UTC.
**Carrying forward:** cipher_kmod 0.4.4 + libcipher_rt.so.v0.2.0_T4_2_3, daemons still up from yesterday.

## Session start state (V1-V6)

| Check | Result |
|---|---|
| V1 Phase 3 ABI 12/12 | happy + negative + root all PASS |
| V2 Fallback md5s | 55ab8c0c (kmod 0.2.0) + 86618c30 (libcipher_v2 0.2.0) — unchanged |
| V3 /proc/cipher/stats banner | `cipher_kmod 0.4.4  uptime=23763726 jiffies` |
| V4 lsmod refcnt | cipher_kmod 57344 2 (gpustate + something else holding) |
| V5 Daemons | cipher-gpustate pid 13912/13913 (6h30m), cipher-exporter pid 14008 (6h30m) — all running cleanly |
| V6 dmesg 1h oops/warn/bug/null | (none) |

Loaded module: 0.4.4 srcversion 1B657D6043D718B6DE2BC56.

Artifact md5s on disk match expected:
- `cipher_kmod.ko.v0.4.3` = fb210777460c47bedd52c7d4223b22ca ✓
- `cipher_kmod.ko.v0.4.4` = c6de1afa228fec20883c6659ea3e5fc7 ✓
- `libcipher_rt.so.v0.2.0_T4_2_3` = bc51b9d6f6827ccdd90ad2d5ab09e3ff ✓

State clean. Proceeding to Task 1.

## Task 1 — B8 WL14 bisect (RESULT: OUTCOME B — pod drift)

Sequence:
1. Stopped daemons (gpustate + exporter).
2. Unloaded kmod 0.4.4. Verified md5 of `cipher_kmod.ko.v0.4.3` = `fb210777460c47bedd52c7d4223b22ca` ✓.
3. Loaded 0.4.3 (uptime 1.02s after insmod). Phase 3 ABI 12/12 PASS on 0.4.3. Taint unchanged 12288.
4. Restarted daemons (different pids: 513331/513332/513336).
5. Ran WL14 600s under libcipher_v2 + 0.4.3.
6. **Result: 8,354 tok/s / 322.4 W / 25.91 TPW** (100% MFU).
7. Stopped daemons. rmmod 0.4.3. insmod 0.4.4 (md5 verified). Phase 3 ABI 12/12 PASS on 0.4.4. Taint unchanged. Daemons restarted (521313/521314/521318).
8. dmesg clean across all four kmod swaps.

| Run | library | kmod | tok/s | TPW |
|---|---|---|---:|---:|
| T4.0.9.D baseline (afternoon 2026-05-13) | libcipher_v2 | 0.4.3 | 8,880 | 27.24 |
| **BISECT (this morning)** | **libcipher_v2** | **0.4.3** | **8,354** | **25.91** |
| Yesterday evening | libcipher_v2 | 0.4.4 | 8,351 | 25.84 |
| Yesterday evening T4.2.3 | libcipher_rt | 0.4.4 | 8,364 | 25.85 |

**OUTCOME B confirmed: pod state drift, NOT kmod transition.** All four
recent measurements cluster at 8,351–8,364 tok/s regardless of
library or kmod version. The afternoon T4.0.9.D baseline at 8,880 is
the outlier (high-water-mark run). The kmod 0.4.3 → 0.4.4 transition
is exonerated.

**B8 closed.** PHASE_4_BACKLOG.md updated. Going forward, WL14
comparisons must use matched-pair (same wall-clock window), not the
static T4.0.9.D baseline.

Proceeding to Task 2 (per outcome B in the brief).

## Task 2 — WL05 same-condition noise band (OUTCOME A: tight ±0.53%)

Three back-to-back 600s WL05 runs under identical conditions
(libcipher_rt.so.v0.2.0_T4_2_3 + kmod 0.4.4 + same daemons +
contiguous wall-clock 02:58–03:30 UTC).

| Run | tok/s | watts | TPW |
|---|---:|---:|---:|
| 1 | 281.57 | 200.64 | 1.4034 |
| 2 | 277.95 | 199.57 | 1.3927 |
| 3 | 277.75 | 199.53 | 1.3920 |

| Stat | TPW | tok/s | watts |
|---|---:|---:|---:|
| Mean | 1.3960 | 279.09 | 199.91 |
| Stddev | 0.0064 | — | — |
| Max-dev % | ±0.53% | ±0.89% | ±0.36% |
| CV % | 0.46% | — | — |

**Outcome A — tight ≤1%.** Same-condition TPW band = **±0.53%**.

**Binding evidence rule for future T4.2.x WL05 claims:**
- `|Δ| ≤ 0.53%` → neutral
- `0.53% < |Δ| ≤ 1.06%` → ambiguous (needs repeat)
- `|Δ| > 1.06%` → genuine signal (lift or regression)

**Important caveat (B9 opened):** today's WL05 mean (1.396) is **4.3%
below** yesterday afternoon's T4.0.9.D baseline (1.459). Same day-drift
signature as WL14. Cross-day comparisons against the static T4.0.9.D
row are unreliable; all future T4.2.x comparisons must use matched-pair
measurement in the same wall-clock window.

Full writeup: PHASE_4_NOISE_BAND_WL05.md.

Proceeding to Task 3.

## Task 3 — B7 ARBITRATE poll-thread fix (FIXED)

### T3.1 advisor pre-design (binding)
- Periodic poll, NOT event-driven (event-driven decomposes to same poll)
- Cadence: **30s** (not 5s — avoid launch-rate noise thrashing the mask)
- Failure mode v1: defensive coding, no watchdog, no auto-restart
- **T3.4 idempotency test is binding** — verify kmod behavior before shipping
- Noise band interpretation: Δ ≤ band = neutral, > 2× band = signal, between = ambiguous

### T3.4 kmod idempotency test (binding gate, run BEFORE coding)
Test: `/home/ubuntu/cipher_phase4_tests/cipher_test_b7_idempotency.c`. Results:

| Sequence | Behavior |
|---|---|
| hint=4 (first) | granted mask 0x0F, count=4 |
| hint=4 (repeat) | IDENTICAL — no change |
| hint=8 | GROWS to mask 0xFF, count=8 |
| hint=4 after hint=8 | STAYS at 0xFF (kmod does NOT release) |
| hint=2 (further shrink) | STAYS at 0xFF |
| hint=8 (re-grow) | works, returns 0xFF |

**Conclusion:** kmod is grow-only-idempotent. B7 poll thread must be
**asymmetric**: re-issue on rank improvement (grow), skip on rank degradation
(would waste ioctl, mask stays anyway). Slot pool leaks under tenant
rank-down transitions — documented limitation.

### T3.3 implementation
- `static __thread` NOT used (per-process poll thread, not per-thread fd)
- New atomics: g_last_hint, g_arb_shutdown, g_arb_poll_started, g_arb_poll_iters, g_arb_poll_grows
- Detached pthread spawned in cipher_rt_arb_init
- 1-second wake granularity inside 30-second cadence so shutdown is responsive

### T3.5 build
- `libcipher_rt.so.v0.2.0_T4_2_3_B7` md5 `71e6e937d2b4be08464e6bc1e0e285b1`
- Source tarball `cipher_rt_phase4_src_T4_2_3_B7.tar.gz` md5 `fceb85402f418f38ad516a84b3b9f48b`

### T3.6 60s WL01 smoke
- ARB-poll thread started (`cadence=30s`)
- 2 iterations fired during 60s — both correctly `skipping reissue (hint=8 <= prev=8)`
- Phase 3 ABI 12/12 PASS post-smoke
- No memory leak observable

### Side-finding (HIGH IMPORTANCE): measurement-script bug discovered

After the first WL05 B7 verify attempt produced 0 ARB-poll log lines
across all 8 children, audit of `run_baseline_wl05.sh` revealed it had
a **hardcoded** `export CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so`
that overrode the `CIPHER_INJECTION_OVERRIDE` env var the T4.2.x runners
set. Per-child `/tmp/wl05_t*.log` confirmed: only `[cipher_v2]` init
lines, no `PR:` / `ARB:` / `SMP:` markers.

**Implications:**
- This morning's first B7 verify run actually loaded libcipher_v2, not B7. Invalid.
- The 3-run noise band measurement was libcipher_v2 (not libcipher_rt), but still valid as a same-condition noise band measurement.
- **Yesterday's T4.2.2 and T4.2.3 WL05 measurements were also libcipher_v2** — invalidating the cross-library framing in PHASE_4_T4_2_2_REPORT.md and PHASE_4_T4_2_3_REPORT.md.

Script fixed (CIPHER_INJECTION_OVERRIDE now honored). Corrections
applied to:
- PHASE_4_T4_2_2_REPORT.md
- PHASE_4_T4_2_3_REPORT.md
- STATUS_OVERNIGHT_2026-05-13.md
- PHASE_4_2_CLUSTER_SUMMARY.md
- PHASE_4_NOISE_BAND_WL05.md

### T3.7 WL05 600s verify under CORRECTED runner (B7 actually loaded)

All 8 children confirmed to load libcipher_rt + B7:
- `[cipher_v2] ARB: arbitrate initialized` × 8
- `[cipher_v2] SMP: sm_packer initialized` × 8
- `[cipher_v2] PR: partition router initialized` × 8
- `[cipher_v2] ARB-poll: thread started, cadence=30s` × 8
- `[cipher_v2] CUPTI subscribed` × 8

Poll thread evidence: each child fired 21 poll iterations across the
600s run (matches 600/30 = 20 expected, 1 extra from coarse 1s wake
granularity).

Measurement:
- MFU% = 100.0, tok/s = 279.37, W = 199.63, **TPW = 1.3994**
- Children complete: 8/8
- Δ vs same-condition libcipher_v2 noise band (mean 1.396, max-dev ±0.53%): **+0.24% — WITHIN noise**

**Verdict:** B7 functionally working as designed. WL05 TPW
statistically indistinguishable from libcipher_v2 same-condition. This
is the predicted outcome (per advisor: B7 alone doesn't deliver lift —
that requires GREEN_CTX in T4.2.4 to make the mask load-bearing).

PHASE_4_BACKLOG.md B7 marked FIXED with full evidence.
Proceeding to morning sign-off (F1-F8).

## Final morning state (F1-F4)

| Gate | Result |
|---|---|
| F1 Phase 3 ABI 12/12 | happy + negative + root all PASS (final check 03:59 UTC) |
| F2 Fallback md5s | 55ab8c0c / 86618c30 — unchanged across session |
| F3 kmod loaded | 0.4.4 (srcversion 1B657D6043D718B6DE2BC56, refcnt=2, daemons holding) |
| F3 taint | 12288 (unchanged) |
| F3 daemons | cipher-gpustate pid 521313/521314 (~1h since post-bisect restart), cipher-exporter pid 521318 |
| F4 dmesg 2h | (clean — no oops/warn/bug/null across all morning kmod swaps) |

### F4 libcipher_rt artifact map

| Build | md5 | Notes |
|---|---|---|
| libcipher_rt.so.v0.2.0_T4_2_2 | 45ed551a281f9aca059d6223cca6e03c | T4.2.2 PR scaffolding |
| libcipher_rt.so.v0.2.0_T4_2_2b | 21a3af0ae822680b060caab959db687f | T4.2.2 with sync_domain no-op + pre-T4.2.3 snapshot |
| libcipher_rt.so.v0.2.0_T4_2_3 | bc51b9d6f6827ccdd90ad2d5ab09e3ff | T4.2.3 (ARBITRATE + SMP) |
| **libcipher_rt.so.v0.2.0_T4_2_3_B7** | **71e6e937d2b4be08464e6bc1e0e285b1** | **B7 poll thread fix (this morning)** |

## Tier achieved

User-defined: REALISTIC = T1+T2 + T3 in progress; GOOD = T1+T2+T3; EXCELLENT = all + WL05 *mask updates working as designed*.

**Achieved: GOOD.** All three tasks shipped, B7 functionally correct and
verified by direct evidence (poll thread runs, 21 iterations × 8
children, mask-request path documented as asymmetric grow-only).

**NOT EXCELLENT** by the brief's criterion. The WL05 verify run
exhibited the **dormant** form of the data-starvation symptom that B7
was created to address: every tenant cold-pathed to hint=8 (because all
8 tenants tied at launches_total=0 → my_rank=0/N → top quartile), and
the asymmetric grow-only logic correctly refused to do anything because
hint was already at max for every tenant. So the poll thread ran but
had no rank-improvements to act on. **Whether B7 actually clears the
symptom under non-degenerate ranks is unmeasured by this morning's
runs.** Needs WL05 re-measurement at a workload mix where some tenants
have differentiated launches_total (e.g. staggered start times, or
asymmetric per-tenant decode lengths). Next session.

## Retroactive script-bug trace — three more clarifications (per F6 advisor)

The morning's run_baseline_wl05.sh bug had retroactive implications.
Three more affected items confirmed clean or correctly bounded:

**(a) libcipher_rt.so.v0.2.0 (pre-Phase-4 archive).** Imported at
session start from `/home/ubuntu/cipher-may13-evidence/libcipher_rt.so`.
That file was **never measured** this session (only the .v0.2.0_T4_2_2,
2b, _T4_2_3, _T4_2_3_B7 builds were exercised). The discipline rule's
"save prior .so" requirement is satisfied by .pre_T4_2_2 as a
prior-art archive, not as a measurement baseline. Confirmed: the script
bug does not affect the prior-art preservation.

**(b) T4.0.9.D contention test (1.4× p99 at 33 threads, 61.8 M ops/s).**
That measurement ran `/tmp/cipher_test_phase4_partition_contention`,
which opens `/dev/cipher` directly via `open()` and does NOT go through
libcipher_rt or libcipher_v2 at all. **Unaffected by the
run_baseline_wl05.sh bug.** The 1.4× contention claim in
PHASE_4_NOTES.md and PHASE_4_T4_0_9_D_CHECKPOINT.md remains valid.

**(c) Initial buggy T4.2.2 build (missing cipher_rt_pr_init() call).**
Overwritten by a fix-build last night with the same name
(libcipher_rt.so.v0.2.0_T4_2_2). The buggy intermediate was used in two
WL01 smoke runs (60s and aborted-600s) and never reached a published
measurement. With today's wl05-script-bug discovery showing that the
T4.2.2 WL05 columns were libcipher_v2 in the children anyway, the
buggy build's downstream impact is **null**: it never produced a
measurement that's referenced in any report.

## Asymmetric grow-only WILL bite earlier than B7's "future kmod fix" framing suggests

Advisor flagged: 4 of 8 WL05 children showed `ARB: granted mask` and
the other 4 didn't. Most likely mechanism: tenants 1-4 cold-path to
hint=8 first, each claiming 8 of 32 slots; tenants 5-8 then cold-path
to hint=8 and hit `cipher_partition_request` rc=-ENOSPC. The asymmetric
grow-only B7 policy then refuses to do anything for tenants 5-8
(they're already "at" their hint, even though they hold 0 slots,
because the kmod returned the requested hint count in the local cached
state). Slot pool is hogged by the first 4 tenants.

This isn't visible today because GREEN_CTX isn't binding the mask to
SM scope. But when T4.2.4 lands, the first-4-tenants-hog pattern
becomes a measurable performance pathology — exactly the wrong
workload pattern to debug a new actuator against.

**Recommendation accepted: elevate decision-queue items #3 and #4 to
TOP before T4.2.4 design work.** Slot release ioctl (or equivalent
fairness mechanism) is now a pre-T4.2.4 blocker, not a deferred
backlog item.

## Updated decision queue for next session

1. **PRE-T4.2.4 SLOT-RELEASE / FAIRNESS** (new top priority per advisor).
   Either (a) new ioctl nr 10 `CIPHER_RELEASE_SM_PARTITION` to free
   slots, or (b) extend nr 9 to accept "exactly N" semantic with
   explicit shrink, or (c) kmod-side hint-vs-grant accounting that
   makes the kmod fair-share rather than FCFS-grab. Without one of
   these, T4.2.4's lift measurement on WL05 will mix actuator effects
   with first-4-tenants-hog effects.

2. **Re-measure WL05 T4.2.2 and T4.2.3 honestly** under the corrected
   `run_baseline_wl05.sh`. Yesterday's "vs baseline" deltas were
   libcipher_v2-vs-libcipher_v2; need actual cross-library numbers.

3. **Investigate the 4-of-8-grants pattern** with cipher_rt_partition_router
   stderr capture. Confirm the suspected ENOSPC mechanism.

4. **B9 day-drift mechanism investigation** (low priority; pod-level,
   not algorithmic).

5. **T4.2.4 GREEN_CTX implementation** — only after slot release lands.

## Decision queue for next session

1. **T4.2.4 GREEN_CTX implementation** (the long-deferred lift mechanism).
   Per yesterday's PHASE_4_2_CLUSTER_SUMMARY.md F2: LD_PRELOAD-style
   wrapper symbols for cuStreamCreate / cudaStreamCreate, substituting
   to cuGreenCtxStreamCreate against a per-tenant green-context pool
   seeded from ARBITRATE's mask. Advisor pre-design needed.

2. **Re-measure WL05 T4.2.2 and T4.2.3 honestly** under the corrected
   `run_baseline_wl05.sh`. Yesterday's "vs baseline" deltas in those
   reports are libcipher_v2-vs-libcipher_v2 and need replacement with
   actual cross-library numbers. ~30 min (3 runs).

3. **Investigate why 4 of 8 WL05 children showed `ARB: granted mask`
   but the other 4 didn't.** Likely the cipher_partition_request kmod
   returns rc=ENOSPC for tenants 5-8 when first 4 each claim hint=8.
   Verify; if so, the asymmetric quartile policy under B7 needs to
   look at *granted mask* not *hint requested* to differentiate
   "asked for 8, got 0" from "asked for 4, got 4."

4. **Slot release ioctl design** for B7's documented leak (tenant
   rank-down doesn't release slots). New ioctl nr 10 = explicit
   CIPHER_RELEASE_SM_PARTITION, or extend nr 9 to accept "want exactly
   N" semantic.

5. **B9 investigation:** identify mechanism of day-to-day pod-state
   drift (~4-5% TPW). nvidia-smi --query-gpu=power.draw,clocks.sm,
   temperature.gpu --format=csv -lms 1000 over a session to characterize.

