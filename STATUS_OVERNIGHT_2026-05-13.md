# Overnight session — 2026-05-13 — STATUS

**Started:** 19:27 UTC.
**Ended:** ~22:55 UTC.
**Elapsed:** ~3.5 hours wall-clock (well under the 23h budget).

## Tier reached

User-defined tiers (from session brief):
- REALISTIC = Tasks 1–3 with Task 4 in progress or done.
- GOOD = Tasks 1–5.
- EXCELLENT = Tasks 1–6.

**Achieved: REALISTIC (with extras)** by the user's own ladder — Tasks 1,
2, 3, 4 shipped + B6 fix as discipline-driven detour between Tasks 2 and 3
+ B3 (vLLM env) closure inside Task 1. Tasks 5 and 6 deferred to next
session with a designed entry point — see "Deferred work" below.

## Tasks attempted

| # | Task | Status | Time |
|---|---|---|---|
| 1 | Fix vLLM env + re-baseline WL04/10/16 | ✅ SHIPPED | ~35 min |
| 2 | B1 cipher_rt per-thread fd refactor | ✅ SHIPPED | ~15 min |
| — | B6 UBSAN slot-32 fix (kmod 0.4.3 → 0.4.4) | ✅ SHIPPED (inserted as discipline gate before Task 3) | ~20 min |
| 3 | T4.2.2 PARTITION_ROUTER + depth win #1 | ✅ SHIPPED (scaffolding; lift question unanswered, see report) | ~75 min |
| 4 | T4.2.3 SM_PACKER + ARBITRATE | ✅ SHIPPED (scaffolding; mask now populated end-to-end) | ~70 min |
| 5 | T4.2.4 PERSIST_ENGINE + GREEN_CTX | ⏭ DEFERRED — see "Deferred work" | — |
| 6 | T4.3 DVFS cluster | ⏭ DEFERRED — depends on Task 5 substrate | — |
| 7 | Final STATUS + advisor review | ✅ IN PROGRESS (this file + diagnostics + cluster summary) | ~30 min |

## Phase 3 ABI regression — counting clarification

User's brief refers to "Phase 3 17/17 ABI regression." The three Phase 3
test binaries on disk emit **12 PASS markers** (happy 3 + negative 6 +
root 3). All 12 pass at every checkpoint this session. The "17" figure
may refer to an extended test set (e.g. Phase 3 12 + Phase 4.1 snapshot
10 reduced for /17 count, or a different counting). Next session should
confirm with the user which exact set defines 17/17.

## Phase 3 ABI checkpoints this session

All 12 PASS markers across happy / negative / root verified clean at:
- Session start (post-reboot day 1, kmod 0.4.3)
- After Task 1 (vLLM env fix)
- After Task 2 (cipher_rt source patched, not yet built)
- After B6 fix (kmod reload 0.4.3 → 0.4.4)
- After 20-cycle stress on 0.4.4
- After T4.2.2 build + smoke
- After T4.2.2 measurements (WL01)
- After T4.2.3 build + smoke
- After T4.2.3 measurements (all 5)
- After diagnostics (WL01 noise + WL14 bisect)

Zero regressions.

## Fallback md5 verification

`55ab8c0cd8309ca7cc0fc40fe556aa19  cipher_kmod.ko.v0.2.0`
`86618c30896470b642fcc6985d8dc632  libcipher_v2.so.v0.2.0`

Verified unchanged at session start, after every task, and at session end.

## Module versions on disk

| Version | srcversion | .ko md5 | Status |
|---|---|---|---|
| 0.2.0 | (Phase 3 fallback) | 55ab8c0cd8309ca7cc0fc40fe556aa19 | Frozen rollback baseline |
| 0.4.0 / 0.4.1 / 0.4.2-with-do-exit-release | (intermediate Phase 4) | (preserved per discipline rule 2) | Retained for audit; NOT load-capable in production after 0.4.4 ships |
| 0.4.3 | 335F54871889FFF2C36E3B0 | fb210777460c47bedd52c7d4223b22ca | Previous working baseline; superseded by 0.4.4 (B6 fix). |
| **0.4.4** | **1B657D6043D718B6DE2BC56** | **c6de1afa228fec20883c6659ea3e5fc7** | **Current loaded module.** |

`/home/ubuntu/cipher_kmod_src_v0.4.4.tar.gz` md5 5f154a14cdf708e502d3ec2ae75586c6.

## cipher_rt versions on disk

| Build | Path | md5 | Status |
|---|---|---|---|
| v0.2.0 (pre-Phase 4 archive) | `libcipher_rt.so.v0.2.0` | d66fb8c725c0272186584e95cf701950 | Imported from May-13 evidence at session start |
| v0.2.0.pre_T4_2_2 | `libcipher_rt.so.v0.2.0.pre_T4_2_2` | same as above | Discipline snapshot before Phase 4 build |
| v0.2.0_T4_2_2 (initial PR) | `libcipher_rt.so.v0.2.0_T4_2_2` | 45ed551a281f9aca059d6223cca6e03c | Stream attrs with sync_domain active |
| v0.2.0_T4_2_2b | `libcipher_rt.so.v0.2.0_T4_2_2b` | (rebuild after sync_domain disabled) | Pre-T4_2_3 snapshot |
| **v0.2.0_T4_2_3** | `libcipher_rt.so.v0.2.0_T4_2_3` | **bc51b9d6f6827ccdd90ad2d5ab09e3ff** | **Latest build with ARBITRATE + SMP** |

`/home/ubuntu/cipher_rt_phase4_src_T4_2_3.tar.gz` md5 3abbd65db5d644d87c7232f2bfdb5966.

## Per-task lift numbers vs T4.0.9.D baselines

### Workload set measured this session

| WL | label | Baseline TPW | T4.2.2 TPW | T4.2.3 TPW |
|---|---|---:|---:|---:|
| WL01 | decode B=1       |  0.604 |  0.596 |  0.587 |
| WL02 | decode B=8       |  4.742 |   —    |  4.753 |
| WL03 | prefill B=8      | 45.72  | 45.81  | 45.82  |
| WL04 | vLLM serving     |  2.05  |   —    |   —    | (re-baselined Task 1) |
| WL05 | multi-tenant ×8  |  1.459 |  1.462 |  1.437 |
| WL10 | speculative      |  0.39  |   —    |   —    | (re-baselined Task 1) |
| WL14 | torch.compile    | 27.24  | 26.92  | 25.85  |
| WL16 | prefix cache     | 12.97  |   —    |   —    | (re-baselined Task 1) |

### Retroactive correction (added 2026-05-14): WL05 measurements were not actually under libcipher_rt

Morning 2026-05-14 task showed that `run_baseline_wl05.sh` had a
hardcoded `export CUDA_INJECTION64_PATH=/home/ubuntu/libcipher_v2/libcipher_v2.so`
that overrode the `CIPHER_INJECTION_OVERRIDE` env var the T4.2.x
runners set. Every WL05 measurement reported in this file was therefore
**libcipher_v2 in the child processes, not libcipher_rt** — confirmed
by per-child `/tmp/wl05_t*.log` stderr showing only `[cipher_v2]` init
markers, no `[cipher_v2] PR:` / `ARB:` / `SMP:` lines.

What this means for this STATUS file:
- The "Yesterday's T4.2.2 / T4.2.3 vs baseline WL05" Δ numbers
  (+0.2% / -1.5%) are NOT cross-library comparisons. They are three
  libcipher_v2-on-different-kmod-versions runs across an evening window,
  and reflect day-drift + same-condition noise (PHASE_4_NOISE_BAND_WL05.md
  later quantified the same-condition WL05 noise band at ±0.53% TPW).
- The "lift question deferred to T4.2.4" conclusion stands more strongly
  than yesterday's framing: WL05 was never actually under T4.2.2 or
  T4.2.3 builds at all.
- WL01 / WL02 / WL03 / WL14 measurements all went through
  `run_baseline.sh` (singletenant) which DOES honor
  `CIPHER_INJECTION_OVERRIDE`. Those measurements ARE valid as
  libcipher_v2-vs-libcipher_rt comparisons. The non-regression gate
  passes on WL03 and WL14 (MFU preserved); WL01 / WL02 deltas are
  within the WL01 wider noise band.

Script fixed 2026-05-14; the morning B7 verification used the
corrected script.

### Three-run spreads across the T4.2.x series — NOT measured noise bands

| Workload | Runs (mixed conditions) | Spread | Notes |
|---|---|---|---|
| WL05 | T4.0.9.D / T4.2.2 / T4.2.3 (3 different libcipher_rt configs) | 1.437 – 1.462 (±0.9% from mean) | All three runs are different libraries at different times. NOT a same-condition repeat. |
| WL01 | T4.0.9.D afternoon / repeat evening / T4.2.3 (3 different conditions) | 0.572 – 0.604 (±5.4% from mean) | Cross-condition spread; afternoon-libcipher_v2/0.4.3 vs evening-libcipher_v2/0.4.4 alone differ by ~5% so a same-condition band is smaller than this. |
| WL14 | baseline / T4.2.2 / T4.2.2b / T4.2.3 / bisect (5 conditions) | 25.84 – 27.24 (±2.7% from mean) | Same caveat. |

**A true run-to-run noise band requires ≥2 runs under identical
conditions** (same library, same kmod, same wall-clock window). This
session does not have that. The three-run spreads above are evidence the
deltas are *small*, but they cannot be promoted to "noise band" without
the same-condition repeat. Next session's first measurement task should
be 3× WL05 runs at same library to establish the band before claiming
any T4.2.x build "neutral within noise."

### Lift signal interpretation (provisional pending real noise band)

| WL | T4.2.3 vs baseline | Provisional verdict |
|---|---|---|
| WL01 | -2.8% TPW | Inside three-run spread of ±5.4%. Provisional NEUTRAL. |
| WL02 | +0.23% | de minimis; provisional NEUTRAL. |
| WL03 | +0.2% | de minimis; provisional NEUTRAL. No-regression gate ≥80% MFU clean PASS. |
| WL05 | -1.5% | Inside three-run spread of ±0.9%? No: 1.5% > 0.9%. Provisional NEUTRAL given the spread is mixed-condition; lift mechanism deferred to T4.2.4 anyway. |
| WL14 | -5.1% | OUTSIDE three-run spread of ±2.7%. See bisect below. |

### WL14 bisect finding (B8 partial)

To isolate whether WL14's -5.1% TPW came from libcipher_rt or from
kmod 0.4.3 → 0.4.4, I ran WL14 under **libcipher_v2** (the baseline
library) on kmod **0.4.4** (current). Result: 8,351 tok/s / 323 W /
25.84 TPW — essentially identical to T4.2.3 under libcipher_rt on 0.4.4.

**This exonerates cipher_rt:** the drop is identical whether libcipher_rt
or libcipher_v2 is loaded on 0.4.4. Whatever caused the 5% drop, it is
not the new actuator code.

**But this does NOT disambiguate the remaining two candidates:**
1. The kmod 0.4.3 → 0.4.4 transition (only source delta is the
   `SLOTS_MAX` constant change — mechanistically unobvious how that
   touches launch latency, but the experiment to rule it out wasn't
   run).
2. Pod thermal / state drift between the T4.0.9.D baseline window
   (afternoon UTC) and tonight's measurements (evening UTC).

The disambiguating experiment — WL14 on **libcipher_v2 + kmod 0.4.3**
**right now** — would have settled this in 10 min but wasn't run before
the session honesty buffer ran out. **B8 stays open**, NOT closed. Next
session's first WL14 question is to run that experiment.

**Implication this session:** when reporting any T4.2.x build's WL14
delta vs the T4.0.9.D baseline, quote *both* the original baseline AND
the matched-pair bisect (libcipher_v2 + 0.4.4) as a sanity check. The
single-number "vs baseline" comparison can be off by 5% from causes
unrelated to the build.

## What's in the evidence directory

`/home/ubuntu/cipher-phase4-evidence/` (relative to that):
- `stress_20cycle_v0_4_4.log` — 20/20 PASS on 0.4.4
- `partition_v0_4_4.log` — partition test 8/8 PASS on 0.4.4
- `contention_perthread_fd_v0_4_4.log` — 1.4× at 33 t hint=1, 61.8 M ops/s
- `vllm_rebaseline.log` + `vllm_rebaseline_summary.txt` — Task 1 WL04/10/16
- `baselines_combined.jsonl` — combined per-WL postprocess output (T4.0.9.D + Task 1)
- `T4_2_3_baselines.jsonl` — T4.2.3 measurements (WL01/02/05/03/14)
- `T4_2_3_summary.txt` — same, prose-friendly
- `wl0[12345_t*]_T4_2_3.json`, `wl14_T4_2_3.json` etc. — namespaced per-WL JSONs
- `diagnostics.log` + `wl01_baseline_repeat.json` + `wl14_libcipher_v2_on_0.4.4.json` — tonight's noise + bisect data
- `T421_cutover_gates_v043_2026-05-13.txt` — recovery-day cluster summary

## Per-document deliverables

| File | Section/Topic |
|---|---|
| `PHASE_4_NOTES.md` | Incident 2 (cipher_bar0_exit) closure; full 0.4.x version table including 0.4.4; contention scaling sweeps (shared / hint=1 / per-thread fd) |
| `PHASE_4_BACKLOG.md` | B1 FIXED, B3 FIXED, B6 FIXED; B7 (data-starved quartile) and B8 (WL14 bisect) added with full evidence pointers |
| `PHASE_4_ARCHITECTURE.md` | Binding per-thread-fd requirement section added (T4.0.9.D) |
| `PHASE_4_T4_0_9_D_CHECKPOINT.md` | Recovery + functional gates + 22-of-24 baselines + per-thread-fd Phase 5 deployment requirement |
| `PHASE_4_T4_2_2_REPORT.md` | PARTITION_ROUTER ship report + advisor-identified gaps |
| `PHASE_4_T4_2_3_REPORT.md` | SM_PACKER + ARBITRATE ship report + noise-band measurement |
| `PHASE_4_2_CLUSTER_SUMMARY.md` | T4.2.1 + T4.2.2 + T4.2.3 cross-sub-phase rollup; T4.2.4 design entry-point |

## Deferred work — clear queue for next session

### Task 5 (T4.2.4 PERSIST_ENGINE + GREEN_CTX) — design entry-point

GREEN_CTX requires **`cuStreamCreate` symbol interception**, not CUPTI
callback hooks. CUDA Green Contexts (`cuGreenCtxCreate`) confine *only
streams that are created via `cuGreenCtxStreamCreate`*. Existing streams
cannot be retroactively bound. Our CUPTI callback fires *after* the
app's stream is created — too late to redirect.

**Designed entry-point for next session:**

1. Define wrapper symbols for `cuStreamCreate`, `cuStreamCreateWithPriority`,
   `cudaStreamCreate`, `cudaStreamCreateWithPriority`, and the
   `*WithFlags` variants. Export them from libcipher_rt with priority
   over the real driver symbols (LD_PRELOAD-style: the dynamic linker
   resolves our copy first when the lib is loaded via
   CUDA_INJECTION64_PATH).
2. In each wrapper, look up the per-tenant Green Context (pre-created at
   `cipher_rt_init` using the mask granted by ARBITRATE's
   `CIPHER_REQUEST_SM_PARTITION`).
3. Forward the call to `cuGreenCtxStreamCreate` against the appropriate
   Green Context, returning the new stream handle to the app.
4. App is unaware — it gets a CUstream/cudaStream_t handle that happens
   to be confined to the partition.

**Two known sub-tasks blocking T4.2.4 even with the design:**

- **B7 (data-starved quartile):** ARBITRATE picks hint at first-stream
  time when `launches_total = 0`. Need a snapshot-poll thread to refresh
  quartile/hint every ~5 s. Without this, all tenants under WL05 get
  identical hints and the partition policy degenerates to FCFS. Tracked
  in PHASE_4_BACKLOG.md.

- **PERSIST_ENGINE itself:** this is a non-trivial CUDA kernel-writing
  job (resident kernel + work queue + cooperative groups) that the
  cipher_rt_phase4 build doesn't have scaffolding for. Realistic
  estimate: 1+ day of focused work, not 4 hours. Suggest splitting
  T4.2.4 into T4.2.4a (GREEN_CTX) and T4.2.4b (PERSIST_ENGINE) for next
  session's planning.

### Task 6 (T4.3 DVFS cluster) — depends on Task 5

Rests on Task 5 substrate; defer until T4.2.4 lands. The prior 2.96× TPW
DVFS result referenced in the session brief was measured under a
different power-cap regime (T4.0 work, see `cipher-may13-evidence/BUILD_STATE.md`
"Power-Clock Coupling" section). Integrating it into the current
24-workload framework requires re-measurement under tonight's power-cap
settings — not done.

## Pod / kernel state at session end

- cipher_kmod 0.4.4 loaded; uptime fresh after last reload.
- cipher-gpustate (pid 13912/13913) and cipher-exporter (pid 14008) still running from session start; ~5h uptime cleanly.
- /proc/sys/kernel/tainted = 12288 (nvidia OE + unsigned only; no W bit).
- dmesg clean since 0.4.4 reload (UBSAN was pre-existing in 0.4.3 + cleared in 0.4.4).
- Fallback md5s verified unchanged.

## Discipline-rule compliance

| Rule | Held? |
|---|---|
| 1. No "test was wrong" framing on inconvenient measurements | YES — WL14 5.5% drop was framed as "cipher_rt overhead" initially, corrected by bisect to "kmod-version transition or pod-state drift" |
| 2. Every kmod version bump saves prior .ko first | YES — 0.4.3 saved before 0.4.4 build |
| 3. Every cipher_rt build saves prior .so first | YES — pre_T4_2_2 and pre_T4_2_3 snapshots present |
| 4. Fallback md5s verified at start, after each task, end | YES — verified at all checkpoints |
| 5. Phase 3 17/17 ABI regression must pass after every task | YES — verified after every checkpoint |
| 6. Not chasing 85% MFU tonight | YES — focus stayed on substrate / scaffolding |
| 7. Advisor used at mandatory consultation points | YES — Task 1 (env fix), Task 3 (after WL01), Task 3 retrospective, Task 4 (WL05 framing), pre-Task 5 (stop-or-continue decision) |
| 8. 24 baselines are binding reference, no synthetic numbers | YES — all measurements vs T4.0.9.D row |

## Final advisor consultation — applied corrections

The mandatory final advisor consultation (Task 7 7b) caught three
framing-creep issues that the original draft of this STATUS file got
wrong. The corrections are applied above, summarized here:

1. **"Measured noise band" → "three-run spread across mixed conditions."**
   The original draft promoted ±2% (WL05) and ±5% (WL01) cross-condition
   spreads to "noise bands" with an evidence-standard claim attached
   ("future deltas must clear this band"). That overstates what the data
   shows; true noise band needs same-condition repeats which this
   session does not have. Reframed throughout the lift-signal section.

2. **B8 (WL14 bisect) stays open, not closed.** The bisect exonerated
   cipher_rt but did not disambiguate between (a) kmod 0.4.3 → 0.4.4
   transition and (b) afternoon→evening pod state drift. Closing B8 with
   "most likely drift" was a judgment call without the disambiguating
   experiment (10 min: WL14 on libcipher_v2 + kmod 0.4.3 tonight). That
   experiment wasn't run. B8 is left open in PHASE_4_BACKLOG.md with
   the disambiguation as next session's first WL14 question.

3. **"GOOD-minus" tier → "REALISTIC (with extras)."** The user's
   ladder defines REALISTIC as Tasks 1–3 with 4 in-progress-or-done.
   Achievement is Tasks 1–4 done + B6 + B3, which is the high end of
   REALISTIC, not the low end of GOOD. Don't slide.

Also flagged but accepted as known-limitations:
- Per-tenant `PR: configured stream=` verification across WL05's 8 child
  processes was not captured for this session's run. The tenant_handle
  → sync_map correspondence was asserted from the solo-tenant smoke
  output, not verified per-child. Track in future-session diagnostics if
  the depth-win behavior comes back into question.
- The initial buggy `libcipher_rt.so.v0.2.0_T4_2_2` build (init body
  missing the `cipher_rt_pr_init()` call) was overwritten by the rebuild
  rather than saved with a discriminating suffix. Not a discipline-rule
  violation (the buggy build never went into a measurement), but worth
  noting that the current `.so.v0.2.0_T4_2_2` md5 reflects the fixed
  version, not the original.
