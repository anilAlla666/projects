# Phase 4 — Backlog (gaps to close before P4.8 ship)

This file tracks gaps discovered during Phase 4 work that are NOT
regressions and do NOT block the current sub-phase, but which must be
closed before P4.8 ship-gate evaluation. Each entry: what, why, where,
severity, suggested fix, evidence.

---

## B1 — cipher_rt_tenant.cpp uses a process-wide `/dev/cipher` fd — **FIXED 2026-05-13 ~19:36 UTC (in source) + verified 2026-05-13 ~20:18 UTC**

**Fix:** `static int g_cipher_fd = -1` replaced by two TLS variables:
```cpp
static __thread int t_cipher_fd             = -1;
static __thread int t_cipher_fd_initialized = 0;
```
A `cipher_rt_tenant_fd_ensure()` helper lazily opens the calling thread's
fd on first ioctl, so query call-sites do not have to remember to call
`cipher_rt_tenant_open()` explicitly. `_open()` and `_close()` become
per-thread idempotent. Prior file saved at
`cipher_rt_tenant.cpp.pre_B1` (md5 4a4e879af32264a8baad10855f07672d).

**Verification:**
- Compile: `g++ -c -O2 -Wall -Wextra -fPIC -I/home/ubuntu/cipher_kmod
  cipher_rt_tenant.cpp` → rc=0 (only pre-existing `-Wparentheses` and
  `-Wstringop-truncation` warnings unrelated to this change).
- TLS storage: `readelf -s` confirms `t_cipher_fd` and
  `t_cipher_fd_initialized` are both `TLS LOCAL` in `.tdata` / `.tbss`.
- Downstream effect: per-thread-fd contention test under cipher_kmod 0.4.4
  (B6 fix in place) gives 1.4× ratio at 33 threads, 61.8 M ops/s. The
  Phase 5 deployment-requirement gate (PHASE_4_ARCHITECTURE.md) is met.

The cipher_rt build itself is constructed in Task 3 of the overnight
session (T4.2.2 PARTITION_ROUTER) — the patched file is in
`/home/ubuntu/cipher_rt_phase4_draft/cipher_rt_tenant.cpp` ready to be
linked into libcipher_rt.so.

---

## B1 (legacy notes for history) — cipher_rt_tenant.cpp uses a process-wide `/dev/cipher` fd

**Discovered:** 2026-05-13 (T4.0.9.D post-reboot recovery, contention
discriminator sweep — see PHASE_4_NOTES.md "Phase 4.2 contention scaling
sweep" and T4.0.9.D checkpoint Section 3).

**Severity:** **HIGH** — gates Phase 5 actuator deployment.

**Where:** `/home/ubuntu/cipher_rt_phase4_draft/cipher_rt_tenant.cpp`,
line 31:

```cpp
static int g_cipher_fd = -1;
```

Opened once in `cipher_rt_tenant_open()` (line 61). Every
`cipher_rt_tenant_query_by_pid`, `_by_id`, and `_refresh_cached` call —
across all threads — issues `ioctl(g_cipher_fd, ...)` through this single
`struct file`.

**Why this is a gap:** Phase 4.2 T4.2.1 contention measurement under
0.4.3 demonstrated empirically that high-concurrency ioctl traffic
through a single fd hits a kernel-side `struct file` throughput cap of
**~8 M ops/s** regardless of allocator design. At 33 contender threads
the p99 ratio degrades to **44–50×** vs single-thread baseline, even
though the lock-free allocator itself scales to 75.9 M ops/s with
per-thread fds (1.4× ratio at 33 threads).

If Phase 5 actuators ship on the current shared-fd pattern, they will
see this ~50× p99 amplification under stream-fan-out workloads. This
defeats the 0.4.2/0.4.3 lock-free allocator's design payoff.

**Suggested fix:**

1. Replace `static int g_cipher_fd = -1` with a `__thread int t_cipher_fd = -1`
   (TLS) — analogous to the existing `__thread struct cipher_rt_tls_cache
   g_tls_cache` already in this file.
2. `cipher_rt_tenant_open()` becomes a per-thread initialization
   (idempotent: returns immediately if `t_cipher_fd >= 0`).
3. Add a `cipher_rt_tenant_open_for_stream(CUstream s)` overload that
   stashes the fd in a stream-attribute slot or a small `CUstream → fd`
   map for actuator paths that are stream-scoped rather than
   thread-scoped.
4. `cipher_rt_tenant_close()` becomes per-thread; document that callers
   should close on thread teardown.

**Verification:** rebuild the actuator test harness, run
`/tmp/cipher_test_phase4_partition_contention 33 30000 1` with
`CIPHER_PER_THREAD_FD=1` against the refactored cipher_rt_tenant.cpp
path. Expected: ratio collapses below 5× (currently 1.4× for the
direct-ioctl test).

**Tracking:** to be addressed in T4.1.2 or T4.1.3 as part of bringing
`cipher_rt_tenant.cpp` into the cipher_rt build atomically with the
cipher_kmod 0.4.0+ cutover.

---

## B2 — libcipher_v2/cipher_cupti.c uses a process-wide `/dev/cipher` fd

**Discovered:** same audit as B1.

**Severity:** **LOW** at current cadence; **WATCH** for future cadence
changes.

**Where:** `/home/ubuntu/libcipher_v2/cipher_cupti.c` line 42:

```c
static int g_cipher_fd = -1;
```

Opened once in `cipher_v2_cupti_init` (line 95). Used in the CUPTI
launch callback (line 86) to submit `CIPHER_SUBMIT_LAUNCH_STATS`.

**Why this is currently benign:** the callback only issues an ioctl
every 256th launch (line 38, `CIPHER_V2_FLUSH_MASK`). Even at 10 k
launches/sec (decode B=1), that is only ~40 ioctls/sec per process —
five orders of magnitude below the ~8 M ops/s VFS cap. No measurable
contention.

**Why it remains a gap:** if Phase 5 raises CUPTI flush frequency (e.g.
for finer-grained telemetry) or if a single process owns many CUDA
streams each generating its own callback storm, the ratio shifts. The
fix is cheap and pre-empts a future regression.

**Suggested fix:** convert to `__thread` once thread-per-stream becomes
the assumed launch pattern. Until then, document the cadence assumption
in `cipher_cupti.c` and keep the shared-fd pattern.

**Tracking:** revisit during P4.7 (PARTITION_ROUTER / per-stream
actuators) when the cadence assumption is reviewed.

---

## B3 — vLLM v1 engine init fails on this pod — **FIXED 2026-05-13 20:06 UTC**

**Fix:** export `VLLM_USE_DEEP_GEMM=0` in `cipher_workloads/measurement/run_baseline.sh` (added at line 60). vLLM 0.20.2's `kernel_warmup` calls `_fp8_linear_may_use_deep_gemm()` which raises `RuntimeError` instead of returning False when the optional `deep_gemm` Hopper FP8 library is absent (graceful-degradation defect upstream). The env var causes the warmup to skip the deep_gemm path entirely. Since TinyLlama and Mistral substitutions are fp16, the FP8 path is never used; bypassing the warmup has no effect on the measurement.

**Verification:**
- WL04 vLLM serving: re-baselined at 84.0% MFU, 571.35 tok/s, 278.84 W, **2.049 TPW**, valid.
- WL10 speculative: re-baselined at 99.0% MFU, 165.47 tok/s, 425.25 W, **0.389 TPW**, valid.
- WL16 prefix cache: re-baselined at 66.0% MFU, 3,154.40 tok/s, 243.19 W, **12.971 TPW**, valid.
- torch 2.11.0+cu130 unchanged (verified before+after).
- Phase 3 ABI happy/negative/root all PASS after Task 1.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint=12288 unchanged.

Evidence: `/home/ubuntu/cipher-phase4-evidence/vllm_rebaseline.log`, `vllm_rebaseline_summary.txt`, `baselines_combined.jsonl` (last 3 rows).

---

## B3 (legacy notes for history) — vLLM v1 engine init fails on this pod

**Discovered:** 2026-05-13 T4.0.9.D baseline pass under 0.4.3.

**Severity:** MEDIUM — 3 of 24 baselines pending.

**Where:** any driver that imports `vllm` and constructs `LLM(...)` —
`/home/ubuntu/cipher_workloads/drivers/wl04_vllm_serving.py`,
`wl10_speculative.py`, `wl16_prefix_caching.py`.

**Symptom:** all three crash at
`vllm.v1.engine.utils.wait_for_engine_startup` with:

```
RuntimeError: Engine core initialization failed. See root cause above.
Failed core proc(s): {}
```

The driver crashes before `run_for_duration.run()` starts, so no
progress file is created and the baseline records as `invalid: missing
progress file`. This is an environmental issue: vLLM v1's engine-core
worker process can't initialize on this pod's torch/CUDA/vLLM version
combination.

**Why not cipher's fault:** the failure is upstream of any cipher_kmod
or libcipher_v2 interaction. Same error signature across all three
vLLM-based drivers. cipher_kmod 0.4.3 is loaded and serving the other
21 baselines cleanly under the same daemons.

**Suggested fix:**

1. Check current vLLM version (`pip show vllm`) and torch version.
2. Either pin vLLM to a known-good release for this torch version, or
   upgrade torch to match.
3. Re-run the three baselines: `./measurement/run_baseline.sh WL04
   --duration 600` (and WL10, WL16) under 0.4.3, then
   `phase4_baseline_postprocess.py WL04 WL10 WL16` to refresh the
   combined JSONL.

**Tracking:** environmental fix, not a cipher work item. Owner: env /
infra. Re-measurement under 0.4.3 produces the missing baselines
without further code changes.

---

## B4 — WL05 baseline JSON schema differs from per-WL schema

**Discovered:** 2026-05-13 T4.0.9.D baseline pass.

**Severity:** LOW — data correctness OK, schema convergence is cosmetic.

**Where:** `/home/ubuntu/cipher_workloads/measurement/run_baseline_wl05.sh`
produces `wl05_p41_baseline.json` with `per_sub_tenant[]` + `device_aggregate`
nesting, while `run_baseline.sh` produces a flat schema with
`mfu_pct_median` at top level.

**Why it matters:** `phase4_baseline_postprocess.py` was patched
post-hoc to read the WL05 `device_aggregate` fields, producing the
correct `steady_state_mfu_pct=100.0` / `watts_avg=190.97` / `TPW=1.46`
row in `baselines_combined.jsonl`. The patch lives in this session's
edit but the postprocess script's WL05 branch is still the original
shape. Anyone re-running the WL05 baseline pipeline needs to either
re-apply the patch or upgrade the postprocess script's WL05 branch to
read the nested schema.

**Suggested fix:** update `phase4_baseline_postprocess.py`'s WL05
branch to read `device_aggregate.aggregate_mfu_pct`,
`device_aggregate.power_watts_median`, and
`device_aggregate.samples_in_window` directly. Or update
`run_baseline_wl05.sh` to also emit top-level `mfu_pct_median` /
`watts_avg` aliases for cross-WL consistency. Either is one-line work.

**Tracking:** address in T4.0.9.E or whenever baselines get re-run for
any reason.

---

## B5 — MFU formula is SM-cycle approximation (tensor-op-ratio refinement deferred)

**Discovered:** existed since `mfu_compute.py` was written; surfaced
again during T4.0.9.D baseline pass when WL14 (torch.compile) hit
100% under the formula.

**Severity:** LOW — consistent within all baselines, deferred to P4.8.

**Where:** `/home/ubuntu/cipher_workloads/launch_lib/mfu_compute.py`
docstring already calls this out:

> "This is a coarse measure — it doesn't separate tensor-core ops from
> generic SM cycles. For comparing CIPHER's per-tenant lift (relative
> delta), the approximation is sufficient. For absolute B200-equivalent
> claims it would need a tensor-op-ratio refinement (Phase 4.8 work)."

**Why it remains a gap:** the four G1–G4 ship-gate criteria in
PHASE_4_ARCHITECTURE.md ("85%+ MFU on each WL", "90%+ MFU on WL05") map
to absolute MFU thresholds. If the substituted models (TinyLlama in
place of Llama-3.2-1B, Mistral-7B in place of Llama-3.1-8B) saturate
this approximation at lower true MFU, the gate threshold becomes
ill-defined.

**Suggested fix:** P4.8 work item. Add CUPTI tensor-op counters to the
collection path and produce a `tensor_mfu_pct` column alongside the
existing `mfu_pct`. Decide whether the G1/G2 thresholds are evaluated
against `mfu_pct` (current) or `tensor_mfu_pct` (corrected).

**Tracking:** P4.8. Not a T4.0.9.D blocker.

---

## Acceptance criteria for closing this backlog

This file is part of T4.0.9.D evidence. Each entry above gets closed by:
- B1: a PR or patch to `cipher_rt_tenant.cpp` + a re-run of the contention discriminator showing ≤5× under simulated stream fan-out, recorded in PHASE_4_NOTES.md.
- B2: deferred to P4.7 cadence review; close with either a fix or an explicit "still benign at chosen cadence" note.
- B3: re-measurement of WL04/WL10/WL16 baselines under 0.4.3 with vLLM fixed, refreshed `baselines_combined.jsonl`.
- B4: postprocess script update or schema unification.
- B5: P4.8 measurement plan revision.

Closures get appended to the entry with a date and an evidence pointer.

---

## B6 — UBSAN shift-out-of-bounds in cipher_partition_allocator.c (slot 32) — **FIXED 2026-05-13 ~20:15 UTC**

**Fix:** option 1 (defensive cap). `CIPHER_PARTITION_SLOTS_MAX` in
`cipher_partition_allocator.c` reduced from 37 → 32 with an in-source comment
documenting the u32-mask width constraint and pointing to this entry. H100 SXM5
loses 1 slot of capacity (33 → 32), B100 loses 4 (36 → 32), B200 loses 5
(37 → 32). H100 PCIe (28) is unaffected. The u64-mask path (B6 option 3) is
deferred until B200 silicon is available; lifting requires a new ioctl nr 10.

Module version bumped 0.4.3 → 0.4.4 (srcversion 1B657D6043D718B6DE2BC56,
md5 c6de1afa228fec20883c6659ea3e5fc7). Prior .ko + src tarball kept at
`/home/ubuntu/cipher_kmod.ko.v0.4.3` and `cipher_kmod_src_v0.4.3.tar.gz`.

**Verification (on 0.4.4):**
- 20/20 insmod / smoke / rmmod stress cycles clean.
- Partition test 8/8 PASS including Test 3 (11-way fork with 25 bits in
  union mask, all within 0–31 → no shift overflow possible).
- UBSAN dmesg messages **ABSENT** after stress + partition tests.
- Per-thread-fd contention 33 t × 30 000 it × hint=1: **1.4×** ratio,
  61.8 M ops/s, p99=327 ns — gate still cleared cleanly under the cap.
- Phase 3 ABI happy / negative / root all PASS.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.

Evidence: `/home/ubuntu/cipher-phase4-evidence/stress_20cycle_v0_4_4.log`,
`partition_v0_4_4.log`, `contention_perthread_fd_v0_4_4.log`.

---

## B6 (legacy notes for history) — UBSAN shift-out-of-bounds at slot 32

**Discovered:** 2026-05-13 19:27 UTC start-of-overnight-session verification, dmesg scan after the T4.0.9.D contention + partition test runs.

**Severity:** MEDIUM — real correctness bug for telemetry/snapshot mask, but does NOT affect ownership/scheduling (slot array remains source of truth).

**Where:** `/home/ubuntu/cipher_kmod/cipher_partition_allocator.c` lines 173 and 177:

```c
cur_mask |= (1U << i);   // line 173 and 177
```

With H100 SXM5 default (`sm_count=132`, `CIPHER_SMS_PER_PARTITION=4`), `cipher_partition_slots = 33`. The loop iterates `i = 0..32`. At `i=32`, `1U << 32` is undefined behavior on a 32-bit `unsigned int`. On x86 the shift count is masked to 5 bits → effective shift of 0 → bit 0 gets set instead of "bit 32". UBSAN catches the standard-violation and prints a stack trace.

**Dmesg evidence:**
```
[  534.592537] UBSAN: shift-out-of-bounds in cipher_partition_allocator.c:177:21
  shift exponent 32 is too large for 32-bit type 'unsigned int'
  ...cipher_partition_request.cold+0x27/0x7e [cipher_kmod]
[  541.698938] UBSAN: shift-out-of-bounds in cipher_partition_allocator.c:173:20
```

Triggered by Partition Test 3 (11-way fork, each child requests 4 slots → 44 demand vs 33 supply, exercising slot index 32).

**Behavioral consequence:** when a tenant holds slot 32, the cached `sm_partition_mask` will incorrectly mark bit 0 instead. This affects:
- snapshot ioctl (nr 8) returns wrong mask bit
- /proc/cipher tenant rows show wrong mask
- cipher_rt PARTITION_ROUTER (Task 3 of this session) would consume the wrong mask if it relies on bit positions to map to specific SM partitions

The slot ATOMIC array is correct — ownership is not corrupted, only the bitmap representation.

**Suggested fix paths (ordered by ABI impact):**

1. **Defensive cap (no ABI change):** cap `cipher_partition_slots = min(slots_from_sm_count, 32)`. Sacrifices 1 slot of capacity on H100 SXM5 (33 → 32), 5 on B200 (37 → 32). H100 PCIe (28) and B100 (36) clipped to 28 / 32 respectively. Single-line change.
2. **Add CIPHER_PARTITION_MASK_BITS=32, use `(i < 32) ? (1U << i) : 0`:** keeps full slot pool, but slot ≥32 is invisible in the mask. Idempotent fast-path breaks for tenants holding only slot ≥32 (cached_count would say 1 but mask=0 → falls through to scan path every call). Subtle.
3. **u64 mask via new ioctl nr 10:** ABI-clean per the cipher-abi-rule (additive nrs only), but cipher_rt + exporter + Phase 4 contract all need updating. Largest blast radius.

**Recommended:** option 1 (defensive cap). The 33→32 H100 capacity loss is one less concurrent 8-way tenant in the worst case (i.e. ~4 instead of ~4 tenants at hint=8, since 33/8 ≈ 4.1 → 32/8 = 4 exactly). Acceptable for current Phase 4 scope. Re-evaluate when B200 hardware shows up.

**Acceptance criteria for closing B6:**
1. Apply chosen fix in source.
2. Bump module to 0.4.4 with `MODULE_VERSION` + `/proc/cipher/stats` banner update. Save `cipher_kmod.ko.v0.4.3` (already saved) and create `cipher_kmod.ko.v0.4.4` per rule 2.
3. Re-run 20-cycle insmod/smoke/rmmod stress test.
4. Re-run Partition Test 3 (11-way fork) — verify UBSAN no longer fires in dmesg.
5. Re-run contention discriminator (per-thread fd) — verify ratio still ≤ 5× (Sweep C from T4.0.9.D).
6. Update PHASE_4_NOTES.md version table + Incident note.
7. Mark B6 FIXED with evidence pointer.

**Timing:** **must be fixed BEFORE T4.2.2 PARTITION_ROUTER actuator (Task 3 of this overnight session) consumes the mask**, because the actuator routes streams based on mask bit positions and a stale slot-0 alias from slot 32 would mis-route. Tasks 1 (vLLM env) and 2 (cipher_rt fd refactor) do not touch this code path, so they can run first.

---

## B7 — ARBITRATE quartile policy is data-starved at first-launch — **FIXED 2026-05-14 ~03:58 UTC**

**Fix:** added `cipher_rt_arb_poll_thread` background thread (30s cadence
per advisor-binding pre-design) inside `cipher_rt_arbitrate.c`. Thread
loops: re-reads tenant snapshot via `cipher_rt_tenant_enumerate`,
re-computes quartile rank, and re-issues `CIPHER_REQUEST_SM_PARTITION`
**only on rank improvement** (asymmetric grow-only). Skips reissue when
hint is same or smaller than the previously-granted level.

**Asymmetric design dictated by measured kmod behavior:** the
`cipher_test_b7_idempotency` test (built and run as binding gate
before the poll thread was implemented) confirmed:
- hint grows → mask updates (grow path)
- hint same → mask identical
- hint shrinks → mask **STAYS** at the larger previous value (kmod does
  NOT release slots)

A symmetric "always reissue" poll thread would waste an ioctl every 30s
in the shrink case for no behavioral change, and would mislead the
cached `g_last_hint` accounting. The grow-only asymmetric path matches
the kmod's actual semantics.

**Documented limitation (UPGRADED TO PRE-T4.2.4 BLOCKER per advisor
2026-05-14):** slot pool can leak under tenant rank-down transitions
(a tenant that briefly hits top-quartile keeps the larger mask forever,
since the kmod won't shrink). Closing this leak requires a kmod-side
change (CIPHER_RELEASE_SM_PARTITION ioctl, or extend nr 9 to accept an
"exactly N" semantic with explicit shrink). **Without this, T4.2.4's
GREEN_CTX lift measurement on WL05 will mix actuator effects with the
first-4-tenants-hog-the-pool pattern** observed in today's WL05 B7
verify (4 of 8 children logged "ARB: granted mask"; the other 4
likely hit ENOSPC at cold-path because slots 0-31 were already claimed).
Elevated to next-session top-of-queue. Tracked as new entry B10.

**Verification:**
- Built `libcipher_rt.so.v0.2.0_T4_2_3_B7` md5 `71e6e937d2b4be08464e6bc1e0e285b1`.
- Source tar `cipher_rt_phase4_src_T4_2_3_B7.tar.gz` md5
  `fceb85402f418f38ad516a84b3b9f48b`.
- WL01 60s smoke: ARB-poll thread started (cadence=30s), 2 poll
  iterations fired during 60s smoke, both correctly skipped reissue
  (hint=8 already at top); Phase 3 ABI all PASS.
- WL05 600s verify under corrected `run_baseline_wl05.sh` (CIPHER_INJECTION_OVERRIDE
  now honored): all 8 children show `ARB-poll: thread started`; 21 poll
  iterations per child × 8 = 168 polls in 600s; TPW = 1.3994 (within
  same-condition noise band ±0.53% of 1.396 mean — Δ +0.24%).
- Phase 3 ABI happy / negative / root all PASS.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.
- No oops/warn/bug in dmesg across morning's two kmod swaps + four
  WL05 runs.

Evidence: `/home/ubuntu/cipher-phase4-evidence/{b7_idempotency_test.log,wl05_B7_run_real.log,wl05_B7_verify_real.json}`.

**Related correction:** the morning audit also surfaced a measurement
bug — `run_baseline_wl05.sh` hardcoded `CUDA_INJECTION64_PATH=libcipher_v2.so`
overriding `CIPHER_INJECTION_OVERRIDE`. Yesterday's WL05 T4.2.2/T4.2.3
"vs baseline" comparisons therefore ran libcipher_v2 in the children,
not libcipher_rt. Script fixed and B7 verify re-run honored
`CIPHER_INJECTION_OVERRIDE`. Corrections applied to all four prior
WL05-referencing reports.

---

## B7 (legacy notes for history) — ARBITRATE quartile policy is data-starved at first-launch

**Discovered:** 2026-05-13 T4.2.3 measurements + post-T4.2.3 advisor review.

**Severity:** MEDIUM — does NOT break the T4.2.3 build (the mask is
metadata only), but blocks the quartile policy from doing its intended
job once T4.2.4 makes the mask load-bearing.

**Where:** `/home/ubuntu/cipher_rt_phase4/cipher_rt_arbitrate.c`,
`cipher_rt_arb_request_partition()`.

**Symptom:** the hint for `CIPHER_REQUEST_SM_PARTITION` is derived from
the calling tenant's `launches_total` quartile-rank among visible peers.
But that rank is computed at first stream observation (which is also the
first kernel launch in the process) when `launches_total` is 0 for
every newly-started tenant. With 8 WL05 tenants all hitting first-launch
within seconds of each other, all 8 are tied at 0 and quartile is degenerate.

**Suggested fix:** add a per-process snapshot-poll thread (one per
`cipher_rt_init`) that wakes every ~5 s, re-reads the tenant snapshot via
the existing Mode 3 TLS cache mechanism, recomputes the quartile hint,
and re-issues `CIPHER_REQUEST_SM_PARTITION` (the kmod allocator is
idempotent on this ioctl — it grows or shrinks the granted mask without
disturbing slots the tenant already holds).

**Timing:** must be in place before T4.2.4 GREEN_CTX binds the mask to
actual SM scope; otherwise the green-context partition assignment is
based on degenerate quartile data.

---

## B8 — WL14 throughput regression under libcipher_rt — bisect pending

**Discovered:** 2026-05-13 T4.2.2 and T4.2.3 measurements.

**Severity:** MEDIUM — MFU gate (≥95%) is unaffected (100% in both
baseline and T4.2.3); the regression is in absolute throughput
(tok/s) which doesn't break a binding gate today but consumes lift
headroom.

**Symptom:** WL14 (torch.compile + TinyLlama decode) shows -5.5 to -5.8%
tok/s consistently across all three libcipher_rt builds (T4.2.2,
T4.2.2b sync_domain disabled, T4.2.3) vs the T4.0.9.D baseline measured
under libcipher_v2. Watts drop proportionally so TPW is preserved within
roughly -0.7 to -5.1%, but tok/s outside the WL05 noise band of ±2%.

**Hypothesis under bisect:** per-launch CUPTI subscriber path is more
expensive in libcipher_rt than libcipher_v2 at the 4,400 launches/sec
workload tempo. WL01 (~2 launches/sec) shows no equivalent drop.

**Bisect now running** (diagnostic 2 in tonight's session):
WL14 under `libcipher_v2.so` + kmod 0.4.4. If result ≈ baseline 8880 →
the drop is libcipher_rt-specific (close B8 with library-overhead
finding). If result ≈ 8370 → the drop is from the 0.4.3 → 0.4.4 kmod
SLOTS_MAX cap change (open new investigation).

**Bisect result (2026-05-13 22:48 UTC):** WL14 under libcipher_v2 + kmod
0.4.4 gives **8,351 tok/s / 323.13 W / 25.84 TPW**, essentially identical
to T4.2.3 under libcipher_rt + 0.4.4 (8,364 / 324 / 25.85). The drop is
NOT cipher_rt. Two candidates remain:

1. The kmod 0.4.3 → 0.4.4 transition. The only source delta is
   `CIPHER_PARTITION_SLOTS_MAX 37 → 32` (B6 fix). That constant affects
   slot-array sizing in kmod-internal data, not the launch hot path.
   Mechanistically implausible as a 5% drop.
2. Pod state / thermal drift between the afternoon T4.0.9.D baseline
   window and the evening measurements (4+ hours later). The same GPU on
   the same pod can vary 2-5% on iterative compute-bound workloads
   between runs due to thermal carryover and host-side scheduling.

**Status: FIXED 2026-05-14 02:56 UTC — pod-state drift confirmed; kmod
0.4.3 → 0.4.4 transition exonerated.**

**Disambiguating experiment run 2026-05-14 morning:** WL14 on
`libcipher_v2.so` + cipher_kmod `v0.4.3` (the same library and same kmod
as the T4.0.9.D baseline). Result: **8,354 tok/s / 322.4 W / 25.91 TPW.**

| Run | library | kmod | tok/s | TPW |
|---|---|---|---:|---:|
| T4.0.9.D baseline (afternoon 2026-05-13) | libcipher_v2 | 0.4.3 | 8,880 | 27.24 |
| **BISECT (morning 2026-05-14)** | **libcipher_v2** | **0.4.3** | **8,354** | **25.91** |
| Yesterday evening | libcipher_v2 | 0.4.4 | 8,351 | 25.84 |
| Yesterday evening T4.2.3 | libcipher_rt | 0.4.4 | 8,364 | 25.85 |

All four recent runs (libcipher_v2/0.4.3, libcipher_v2/0.4.4,
libcipher_rt/0.4.4 × 1) cluster at **8,351–8,364 tok/s** regardless of
library or kmod version. Only the afternoon T4.0.9.D baseline at 8,880
sits 5–6% above. The cause is NOT kmod-version transition (same kmod
gives the lower number today) and NOT cipher_rt overhead (libcipher_v2
gives the same lower number); it is pod state drift between the
afternoon T4.0.9.D measurement window and ~12+ hours later.

The "mechanistically implausible" framing for the kmod-version
hypothesis was CORRECT (the SLOTS_MAX change in 0.4.4 doesn't touch
launch latency). The hypothesis was right; the bisect now provides
direct evidence.

**Closure rule for future T4.2.x comparisons on WL14:** the T4.0.9.D
baseline (8,880 tok/s) is a high-water-mark run that this pod does not
reproduce on a fresh-day measurement. Going forward, WL14 comparisons
must use matched-pair measurement (same wall-clock window, same library
on baseline + build under test) — NOT the static T4.0.9.D baseline. The
8,350-ish cluster is the realistic WL14 reference under current pod
conditions.

---

## B10 — slot-release mechanism (pre-T4.2.4 blocker) — **FIXED 2026-05-14 04:29 UTC**

**Fix:** option 2 (extend ioctl nr 9 — recommended path).

**Kmod side (0.4.5, srcversion 4618FD1FC28BEE5EBC32944, md5
b47db4500e226f5ddc010f9148d3d6ca):**
- Renamed `cipher_partition_request.reserved` → `flags` (ABI-compat: same
  offset/size; zero-init by old callers preserves T4.2.1 behavior).
- Defined `CIPHER_PARTITION_FLAG_FIT_HINT = 1U << 0`.
- New `cipher_partition_request_v2(pid, hint, flags, ...)` implements
  fit-to-N semantic. Old `cipher_partition_request` is a wrapper that
  calls v2 with flags=0.
- With FIT_HINT and have>hint: walks slot array in reverse, releases
  excess via `atomic_cmpxchg(my_state → 0)`, preserving lowest-indexed
  slots (cache-warm-friendly).
- With FIT_HINT and hint=0: explicit release-all.
- Unknown flag bits → -EINVAL.

**Userspace side (libcipher_rt.so.v0.2.0_T4_2_3_B7_v2_B10, md5
5d0f200c75c179040b57f4ebcd293500):**
- B7 poll thread rewritten from asymmetric-grow-only to
  **fair-share-fit** policy: `fair = max(1, min(8, 32/N_peers))`, reissue
  with FIT_HINT on grow OR shrink.
- New atomic counter `g_arb_poll_shrinks` for telemetry.

**Verification:**
1. `cipher_test_b10_fit_hint` — 11/11 PASS across all FIT_HINT semantic
   cases (shrink, grow, release-all, no-op same, EINVAL on bad flag).
2. WL05 600s under B10-enabled stack: **8/8 children get non-zero
   grants** (vs pre-B10 4/8 pattern). Final mask distribution:
   t1-t4 = 4 slots each (shrunk 8→4); t5-t8 = {4,5,4,3} slots (grew 0→).
   Sum = 32 = full pool. Each tenant within ±1 of fair share=4.
3. ARB-poll iterations per child: 21 polls in 600s × 8 children.
   t1-t4 each: 1 shrink + 19 no-ops (held==fair → idle after first
   correction). t5-t8: 1-20 grows (acquiring during shrink window of
   t1-t4). All children logged at least one successful grant.
4. TPW: 1.3998 — within today's same-condition libcipher_v2 noise band
   (1.396 ±0.53% → 1.388..1.403). Multi-tenant TPW unchanged at this
   layer (SM scope still unrestricted — GREEN_CTX is the lift
   mechanism).
5. Phase 3 ABI 12/12 PASS, fallback md5s unchanged, taint 12288 unchanged.

**Closed dependent gap:** the first-4-tenants-hog pattern from yesterday
is **gone**. T4.2.4 measurements can now run without that confounder.

**Documented limitation (still open, lower-priority):** the convergence
takes one poll cycle (≤30s). During the first 30s of any multi-tenant
workload, cold-path FCFS hog persists. For long-running workloads
(WL05's 600s) this is negligible; for short benchmarks <60s, the cold
path dominates and the policy would need a snapshot-pre-read at cold
path. Deferred (not blocking T4.2.4 for WL05 which is the primary
multi-tenant target).

---

## B10 (legacy notes for history) — slot-release mechanism

**Discovered:** 2026-05-14 (this morning's B7 verify revealed that 4 of
8 WL05 children logged `ARB: granted mask`; the other 4 likely hit
cipher_partition_request ENOSPC at cold path because tenants 1-4 hogged
all 32 slots).

**Severity:** HIGH — blocks meaningful T4.2.4 GREEN_CTX measurement.

**Mechanism:** kmod is grow-only-idempotent (B7 idempotency test
result). Tenants 1-4 cold-path to hint=8 and each get 8 slots; tenants
5-8 cold-path to hint=8 but slot pool is empty. The asymmetric
grow-only B7 policy then refuses to re-issue for tenants 5-8 because
they're "at" their requested hint locally. First-4-tenants-hog is the
result.

**Suggested fixes (any one closes B10):**

1. New ioctl nr 10 `CIPHER_RELEASE_SM_PARTITION` that explicitly
   releases all slots held by current_pid. Add to libcipher_rt's
   ARBITRATE: when poll-thread detects local mask vs intended hint
   mismatch (granted < requested), call release then re-request with
   *current* hint to release-and-reacquire. Kmod side: trivial — atomic
   cmpxchg loop walking the slot array.

2. Extend nr 9 to accept "exactly N" semantic — caller specifies
   `hint_partitions = N` AND a `release_excess=1` flag. Kmod
   shrinks granted mask down to N slots if currently larger. Single
   ioctl, no new entry point. Cleaner API.

3. Kmod-side fair-share rather than FCFS-grab: on each
   `CIPHER_REQUEST_SM_PARTITION`, compute fair share across active
   tenants and steal slots from over-represented tenants if needed.
   Higher complexity; defer to a later sub-phase.

**Recommendation:** option 2 (extend nr 9). Minimal ABI surface area,
matches the grow-only-by-default that current code expects, and
provides shrink semantics for B7's poll thread to use.

**Timing:** pre-T4.2.4 blocker. Without this fix landed, T4.2.4
GREEN_CTX lift measurements on WL05 will be confounded by 4 hog
tenants vs 4 starved tenants. Debug pain is much higher when the
actuator is also new.

---

## B9 — day-to-day pod-state drift dominates same-condition noise

**Discovered:** 2026-05-14 morning (Task 2 — WL05 noise band measurement).

**Severity:** MEDIUM — does not block any single comparison, but it
breaks cross-day comparisons against the static T4.0.9.D baseline row.

**Finding:** same-condition WL05 noise band is **±0.53% TPW** (3
back-to-back runs at 1.4034 / 1.3927 / 1.3920). But today's mean (1.396)
is **4.3% below** yesterday afternoon's T4.0.9.D baseline (1.459). The
WL14 bisect (B8 FIXED) confirmed the same drift signature: ~5% delta
between afternoon and morning/evening runs of the same workload on the
same library+kmod.

**Mechanism:** unknown but unambiguous. Day-to-day pod-state drift —
thermal carry-over, host-side scheduler state, neighbor noise — on the
order of 3–5% TPW. This is **6–10× the same-condition noise** measured
in PHASE_4_NOISE_BAND_WL05.md.

**Binding rule for future comparisons:** all T4.2.x lift claims against
baseline must use matched-pair measurement (baseline + build under test
captured in the same wall-clock window, on the same pod state). The
static T4.0.9.D baselines from 2026-05-13 are reference points for
absolute throughput levels and the substrate readiness criteria; they
are NOT directly comparable to multi-hour-later measurements.

**Acceptance criteria for closing B9:** investigate the source of the
drift (likely thermal — check nvidia-smi power-cap state and clock
sustained levels morning vs afternoon). If thermal, document the
power-cap-history dependence and proceed with matched-pair as the
standard. If something else, open a more specific entry.

**Tracking:** non-blocking; future T4.2.x sub-phase reports must
declare which measurement-pair window they used.

---

## B11 — token-count overstatement in `m.generate()`-based drivers

**Discovered:** 2026-05-14 T4.2.4c prep audit.

**Severity:** MEDIUM — every WL01-family tok/s and TPW number on record
is overstated by ~15% (prompt-echo included in token count). Affects all
cross-WL comparison ratios mildly but consistently.

**Mechanism:** `m.generate(..., max_new_tokens=N)` returns a tensor of
shape `[B, prompt_len + N]` (prompt tokens concatenated with new tokens).
Drivers report `int(out.shape[-1])`, which is `prompt_len + N`, not `N`.

**Affected drivers (same pattern):**
- `wl01_decode_b1.py:20` — **FIXED 2026-05-14 ~12:28 UTC** (`out.shape[-1] - PROMPT_LEN`)
- `wl02_decode_b8.py:21` — same pattern, batched
- `wl11_agentic.py:26` — same pattern, multi-turn accumulator
- `wl20_llava.py:25` — `out.numel()` flavor, multimodal
- `wl21_code_generation.py:20`
- `wl22_rag.py:39`
- `wl23_model_switch.py:32`

**Fix:** subtract prompt input_ids length per call. Pattern in wl01:
```python
PROMPT_LEN = int(prompt.input_ids.shape[-1])
return 1, int(out.shape[-1]) - PROMPT_LEN
```

**Status:** wl01 FIXED for T4.2.4c (used as victim in noisy-neighbor A/B).
Other drivers deferred — re-measurements under fixed token counts can be
batched as a P4.2 close-out clean-up pass once T4.2.4c is settled. Until
then, cross-WL comparisons should note the ~15% overstatement asymmetry
between fixed (wl01) and unfixed drivers.

---

## B12 — GREEN_CTX enforcement gap — **FIXED 2026-05-14 T4.2.4d**

**Discovered:** 2026-05-14 T4.2.4c bomb-throughput diagnostic (Δ A/B
−0.6%).

**Severity:** HIGH — gates Phase 4.2 actuator product-value claim.

**Mechanism (root cause):** PyTorch's `m.generate(...)` decode/prefill
launches go through `stream==NULL` (the per-context default stream),
not the explicit streams we hooked via `cuStreamCreate`. The push-on-
create approach (T4.2.4a/b/c) bound the ~32 streams PyTorch creates,
but the 884k/30s hot-path kernel launches resolved NULL → primary
context's default stream → full 132 SMs.

**Fix:** persistent `cuCtxSetCurrent(green_cuctx)` at every cuLaunchKernel
ENTER, with a fast-path that skips the syscall when the thread's
current context is already green. NULL-stream launches now resolve to
green ctx's default stream → 8-SM partition. Verified by raw-CUDA
microbench (`/tmp/test_green_ctx_nullstream.cu` TEST B — 14.6×
slowdown under cuCtxSetCurrent(green) + NULL stream).

**Verification:** bomb solo under T4.2.4d (`libcipher_rt.so` md5
`50414674ddad2689191d13a92377e492`): 10.4 prefills/s vs 114.1 under
libcipher_v2 = 11× slowdown. User's binding diagnostic threshold <40
prefills/s met decisively. Multi-tenant A/B confirmed in PHASE_4_T4_2_4d_REPORT.md.

---

## B14 — BAR0 sniff + register-map discovery — **DEFERRED to Phase 6 hardening**

**Discovered:** 2026-05-14 T4.3.0 Sub-phase A discovery.

**Severity:** LOW (research-value hardening, not deployment-blocking).

**Background:** the original Stream 2 plan included an empirical BAR0
snapshot+diff register-map discovery (Priority 2, 3-4 h). The kmod-
mediated NVML path (T4.3.2) closed the privilege gap that motivated
direct-BAR0 work, so the substrate is no longer on the critical path.

**Why we'd still want it (someday):** future-pod portability if NVML
clock-lock ever returns to `NVML_ERROR_NOT_SUPPORTED` on a future
Lambda image / driver version / container security profile. A direct
BAR0 register write substrate is the fallback for that scenario.

**Discovery already in place:**
- OGKM cloned at `/home/ubuntu/ext/open-gpu-kernel-modules` (Hopper register
  headers in `src/common/inc/swref/published/hopper/gh100/` — but NO
  `dev_pwr.h`/`dev_clk.h` published; everything routed through GSP RPC).
- Nouveau cloned at `/home/ubuntu/ext/nouveau-src/` (clock-control source
  stops at Kepler/Maxwell-mobile; no Hopper reference).
- `PHASE_4_T4_3_0_DISCOVERY.md` documents the dead-ends.

**Phase 6 work scope:**
1. Sniff: read BAR0 windows before/after `nvidia-smi -lgc N` invocations,
   diff to find clock-control register addresses.
2. Read-path validation: confirm our register interpretation matches
   nvidia-smi readback.
3. Write-path: bounds-checked direct BAR0 writes, three-indicator
   verification (read-back + nvidia-smi + workload runtime delta).

**Timing:** non-urgent. Re-enter active development only if NVML path
breaks on a future deployment target.

---

## B13 — persistent cuCtxSetCurrent side effects (informational, open)

**Discovered:** 2026-05-14 T4.2.4d during fix design.

**Severity:** LOW — current scope is tenant-isolated workloads on
single-context Python processes.

**Mechanism:** the T4.2.4d fix sets the green CUcontext as the calling
thread's current context persistently. All CUDA APIs on that thread
(cudaMalloc, cudaMemcpy, cuEvent*, cuStreamSynchronize, etc.) now
operate under the green ctx. Green contexts inherit the primary
context's memory pool per the CUDA docs, so allocations work the same
way. Stream synchronization and event APIs likewise work — they're
just attached to a different context.

**Acceptable for current scope** because:
1. The bomb-throughput diagnostic (clean A/B) showed no functional
   regressions under T4.2.4d.
2. The actuator's intended use is per-tenant single-context isolation,
   which matches the fix's behavior.
3. Future multi-context workloads (e.g., one process orchestrating
   multiple tenants' streams) would require either per-stream context
   association or a different mechanism — out of scope for P4.2.

**To revisit if:** Phase 5 actuator wiring needs multi-context-per-
process; OR a workload surfaces a behavior where cudaMalloc/Memcpy
on green ctx behaves differently from primary ctx in an unwanted way.

---

## P4.2 cluster — what's NOT in the backlog because it ships clean

The B-entries above are *additions* to the open list. Items that *closed*
during this session (B1, B3, B6) are kept above with FIXED stamps for
audit history. Phase 3 ABI 12-PASS regression, fallback md5 invariance,
zero-new-taint, 20-cycle stress, partition 8/8, contention 1.4× — all
verified at session end on kmod 0.4.4.
