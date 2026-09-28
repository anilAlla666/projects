# CIPHER DEEP INSPECTION REPORT — line-by-line verification of CIPHER_REENGINEERING_PLAN.md

**Date:** 2026-05-20
**Type:** Code inspection. No design. Read what exists, document every trap.
**Scope:** Five sections (A–E) verifying the reengineering plan against the on-disk codebase.
**Companion document:** [CIPHER_REENGINEERING_VERIFICATION_PHASE_B.md](./CIPHER_REENGINEERING_VERIFICATION_PHASE_B.md) — full 771-line per-op dependency walk (Section B detail).
**Cross-reference:** [CIPHER_REENGINEERING_PLAN.md](./CIPHER_REENGINEERING_PLAN.md) — v1.0, md5 `e1f047d1...` — claims verified or corrected below.

---

## EXECUTIVE SUMMARY

Across 5 sections of line-by-line code inspection, the reengineering plan is **substantially accurate**, but verification produced **9 specific corrections** to the plan's claims and **6 new findings** the plan missed. None are show-stoppers for v1; one is a critical clarification (the `cipher_dispatch.cpp` PASS_THROUGH line numbers cited in the plan came from the *src/ shadow* file, not the *top-level live* file — the plan's port instruction is correct, but the citations must be updated). The actuator-registry pattern that the plan declares as the unification join point has been verified end-to-end, but it has **5 semantic divergences between the matmul and attn instances** that the plan didn't surface — the new classifier substrate must pick a contract, and §A.3 below names which divergences to resolve which way.

Plan claims **verified**: 16. Plan claims **corrected**: 9. **New findings** (traps not in plan): 6. **Show-stoppers**: 0.

---

<a id="section-a"></a>
## SECTION A — ACTUATOR REGISTRY CONTRACT VERIFICATION

The plan asserts (§4.0) that the priority-ordered actuator registry pattern on `cipher_rt_matmul_dispatch.c` and `cipher_rt_attn_dispatch.cpp` is the unification join point and that the AUDIT op is the canonical priority-0 observer template. Verified end-to-end below.

### A.1 — cipher_rt_matmul_dispatch.c — full read

**File:** `/home/ubuntu/cipher_rt_phase4/cipher_rt_matmul_dispatch.c` (124 LOC) + `cipher_rt_matmul_dispatch.h` (130 LOC).

**Contract semantics — confirmed by direct read:**

| Property | Location | Verified |
|---|---|---|
| Registry capacity | `cipher_rt_matmul_dispatch.c:15` — `CIPHER_RT_MATMUL_MAX_ACTUATORS 16` | ✓ |
| Result enum (3 values) | `cipher_rt_matmul_dispatch.h:70-74` — `HANDLED=0, PASSTHROUGH=1, ERROR=2` | ✓ |
| Registration signature | `cipher_rt_matmul_dispatch.c:52-83` — `cipher_rt_matmul_register_actuator(const struct *actuator)` | ✓ |
| Registration locks `reg_lock` mutex | `cipher_rt_matmul_dispatch.c:60, 81` | ✓ |
| Registry full → return -1 | `cipher_rt_matmul_dispatch.c:62-67` | ✓ |
| Insertion-sort by priority (lower first, STABLE for equal priorities) | `cipher_rt_matmul_dispatch.c:50-51, 70-74` — comparison `priority <= actuator->priority` means equal-priority shifts the new entry to slot `i` (after existing) | ✓ |
| `maybe_handle(call, *out_status)` signature | `cipher_rt_matmul_dispatch.h:80-87` — TWO arguments | ✓ |
| Dispatch walks actuators **without** holding lock | `cipher_rt_matmul_dispatch.c:91-107` — direct iteration of `g_disp.actuators[]`, comment L93 says "registry append-only after init" | ✓ |
| HANDLED → return `*out_status` from actuator immediately | `cipher_rt_matmul_dispatch.c:97-100` | ✓ |
| ERROR → **break loop**, fall through to real cuBLAS | `cipher_rt_matmul_dispatch.c:101-105` — `break;` exits the for-loop | ✓ |
| PASSTHROUGH → try next actuator | `cipher_rt_matmul_dispatch.c:106` (comment) | ✓ |
| All-PASSTHROUGH → call `passthrough_fn` with original args | `cipher_rt_matmul_dispatch.c:109-115` | ✓ |
| Telemetry: `atomic_ulong total / handled / passthrough` | `cipher_rt_matmul_dispatch.c:24-26, 91, 98, 109` | ✓ |
| atexit diagnostic emitter | `cipher_rt_matmul_dispatch.c:31-38, 43` | ✓ |
| Init is idempotent | `cipher_rt_matmul_dispatch.c:42` — `atomic_exchange(&g_disp.inited, 1)` | ✓ |

### A.2 — cipher_rt_attn_dispatch.cpp — full read

**File:** `/home/ubuntu/cipher_rt_phase4/cipher_rt_attn_dispatch.cpp` (422 LOC) + `cipher_rt_attn_dispatch.h` (139 LOC).

**Contract semantics — confirmed by direct read:**

| Property | Location | Verified |
|---|---|---|
| Registry capacity | `cipher_rt_attn_dispatch.h:108` — `CIPHER_RT_ATTN_MAX_ACTUATORS 16` | ✓ |
| Result enum (**4** values) | `cipher_rt_attn_dispatch.h:91-96` — `HANDLED=0, PASSTHROUGH=1, REDIRECTED=2, ERROR=3` | ✓ |
| Registration signature | `cipher_rt_attn_dispatch.cpp:211-231` — `cipher_rt_attn_register_actuator(const cipher_rt_attn_actuator* a)` | ✓ |
| Registration locks `g_reg.mu` mutex | `cipher_rt_attn_dispatch.cpp:213, 227` | ✓ |
| Registry full → return -1 | `cipher_rt_attn_dispatch.cpp:214-217` | ✓ |
| Insertion sort by priority (strictly-lower first) | `cipher_rt_attn_dispatch.cpp:218-225` — comparison `a->priority < g_reg.entries[i].priority` strict < | ✓ |
| `maybe_handle(call)` signature — **NO** status arg | `cipher_rt_attn_dispatch.h:102` — ONE argument | ✓ |
| Dispatch **snapshots actuators under lock**, then iterates | `cipher_rt_attn_dispatch.cpp:143-148` — memcpy of `cipher_rt_attn_actuator[16]` under `pthread_mutex_lock`, then unlock | ✓ |
| HANDLED → counted, return HANDLED | `cipher_rt_attn_dispatch.cpp:154-156` | ✓ |
| **REDIRECTED is logged and treated as PASSTHROUGH** (T4.6.1 — substitution mechanism deferred) | `cipher_rt_attn_dispatch.cpp:157-165` | ✓ |
| ERROR → log, **continue to next actuator** (not break!) | `cipher_rt_attn_dispatch.cpp:166-169` — `break;` exits the switch only; outer for-loop continues | ✓ |
| All-PASSTHROUGH → counted, return PASSTHROUGH | `cipher_rt_attn_dispatch.cpp:175-176` | ✓ |
| **Trampoline ALWAYS calls `orig(...)` regardless of route() result** (T4.6.1 observe-only) | `cipher_rt_attn_dispatch.cpp:321, 359, 398` — return is `return orig(...)` unconditional, outside the if-tensors_are_real-try block | ✓ — see A.3 finding #5 |
| Telemetry: `std::atomic<uint64_t>` total / handled / passthrough / redirected / per_backend[4] / tramp_calls / tramp_fake | `cipher_rt_attn_dispatch.cpp:70-84` | ✓ |
| Destructor diagnostic emitter | `cipher_rt_attn_dispatch.cpp:245-256` | ✓ |
| Init is idempotent (g_init_announced compare-exchange) | `cipher_rt_attn_dispatch.cpp:237-241` | ✓ |
| FakeTensor / dynamo-tracing guard | `cipher_rt_attn_dispatch.cpp:90-104` — `tensors_are_real` dispatch-key check | ✓ |
| Lazy symbol resolution via `dlopen("libtorch_cpu.so", RTLD_NOLOAD) + dlsym` | `cipher_rt_attn_dispatch.cpp:189-205` | ✓ |
| `abort()` on dlsym failure (better than self-call infinite loop) | `cipher_rt_attn_dispatch.cpp:298-302, 333-335, 371-373` | ✓ |

### A.3 — Contract diff: matmul vs attn — **5 SEMANTIC divergences**

The reengineering plan §4.0 implies the two registries follow the same contract. They do not. Five divergences, classified:

| # | Divergence | matmul behavior | attn behavior | Class |
|---|---|---|---|---|
| 1 | Result enum cardinality | 3 values (HANDLED/PASSTHROUGH/ERROR) | **4 values** (HANDLED/PASSTHROUGH/REDIRECTED/ERROR) — REDIRECTED at value 2 | **SEMANTIC** — actuator code that returns `2` means PASSTHROUGH on attn vs ERROR on matmul. Unified classifier substrate must pick one canonical enum. |
| 2 | `maybe_handle` signature | `(call, *out_status)` — actuator writes cuBLAS-style status | `(call)` — actuator returns enum only | **SEMANTIC** — attn cannot propagate a return status because T4.6.1 ships observe-only and the original is always called anyway. The classifier substrate doesn't need to propagate status, so it should follow the **attn** signature (simpler). |
| 3 | Lock-during-dispatch | NO lock (assumes registry append-only after init) | LOCKS to memcpy snapshot, then releases | **SEMANTIC** — matmul's no-lock pattern is faster but UNSAFE if any actuator registers after init. Attn's snapshot-under-lock is the safer pattern. The classifier substrate should follow **attn** (snapshot pattern, ~16 × sizeof(actuator) memcpy per call is acceptable). |
| 4 | ERROR handling | `break` — exits the loop, falls through to real cuBLAS | **continues** to next actuator (the `break;` inside the switch case exits only the switch, not the for-loop) | **SEMANTIC** — matmul stops at first ERROR (defensive); attn keeps trying actuators (resilient). The classifier substrate should follow **matmul** (an ERROR from a classifier means classification is unreliable; fall through to default routing). |
| 5 | HANDLED bypass semantics | The substrate returns `*out_status` directly; the real cuBLAS is **not** called | **The trampoline ALWAYS calls orig(...)** regardless of route() returning HANDLED — T4.6.1 observe-only | **BLOCKING for future actuators** — when the attn substrate moves from T4.6.1 observe-only to T4.6.3 actuator-handled, the trampolines at L321/L359/L398 must change to conditionally call orig. Currently a HANDLED return is discarded. |
| 6 | Insertion sort comparison | `<= actuator->priority` (existing equal-priority items stay first; new equal goes AFTER) | `< g_reg.entries[i].priority` (strict; equal goes AFTER) | **COSMETIC** — same effective stable-sort behavior. Both place new equal-priority items after existing equal-priority items. |

**Which contract should the classifier substrate match?** A hybrid:
- **Enum**: follow attn (4 values: HANDLED/PASSTHROUGH/REDIRECTED/ERROR) — superset is safer.
- **maybe_handle signature**: follow attn (single-arg) — classifier doesn't need status propagation.
- **Lock-during-dispatch**: follow attn (snapshot under lock) — safer.
- **ERROR handling**: follow matmul (break loop, fall through to default routing) — classifier errors should be terminal not retried.
- **HANDLED semantics**: define explicitly — classifier HANDLED means "this classifier produced a confident result, downstream routing accepts it"; PASSTHROUGH means "no opinion, try next or default."

**Recommended classifier-substrate contract spec** (for Week 1 port):

```c
enum cipher_rt_classify_result {
    CIPHER_RT_CLASSIFY_HANDLED     = 0,  /* classifier produced confident result */
    CIPHER_RT_CLASSIFY_PASSTHROUGH = 1,  /* no opinion, try next */
    CIPHER_RT_CLASSIFY_REDIRECTED  = 2,  /* reserved for v2 — surrogate path */
    CIPHER_RT_CLASSIFY_ERROR       = 3,  /* classifier failed; substrate stops loop, returns default kernel_class */
};

struct cipher_rt_classifier {
    const char *name;
    int         priority;
    int       (*maybe_classify)(const struct cipher_rt_classify_call *call,
                                struct cipher_rt_classify_out *out);
};
```

### A.4 — Third registry instance? — **NO**

Grep across `cipher_rt_phase4/`, `cipher_kmod/`, `cipher-may13-evidence/src/` for `register_actuator` and priority-ordered patterns finds exactly two instances:

```
cipher_rt_matmul_dispatch.c:52  cipher_rt_matmul_register_actuator
cipher_rt_attn_dispatch.cpp:211 cipher_rt_attn_register_actuator
```

Callers (all in `cipher_rt_phase4/`):
- `cipher_rt_audit.c:190-191` — registers AUDIT on both
- `cipher_rt_attn_test_actuator.c:98` — registers test actuator on attn
- `cipher_rt_marlin_actuator.c:198` — registers MARLIN on matmul

**Verified: no hidden third registry instance.** The unification plan adding a classifier substrate is a clean third instance, not duplicating an existing one.

### A.5 — AUDIT priority-0 always-PASSTHROUGH template — **VERIFIED**

The plan §2.2 / §4.0 / §4.4 claims AUDIT is the canonical priority-0 always-PASSTHROUGH observer template. Verified by reading `cipher_rt_audit.c` in full:

| Plan claim | Code evidence | Verified |
|---|---|---|
| Registers at priority 0 on matmul | `cipher_rt_audit.c:132-134` — `audit_matmul_act = { .name = "audit", .priority = 0, ... }` | ✓ |
| Registers at priority 0 on attn | `cipher_rt_audit.c:135-137` — `audit_attn_act = { .name = "audit", .priority = 0, ... }` | ✓ |
| Both registrations happen at init | `cipher_rt_audit.c:190-191` | ✓ |
| matmul handler always returns PASSTHROUGH | `cipher_rt_audit.c:114` — `return CIPHER_RT_MATMUL_PASSTHROUGH;` | ✓ |
| attn handler always returns PASSTHROUGH | `cipher_rt_audit.c:129` — `return CIPHER_RT_ATTN_PASSTHROUGH;` | ✓ |
| Env-gated by `CIPHER_AUDIT` | `cipher_rt_audit.c:182-185` — bails if env not set | ✓ |
| HMAC-SHA256 chain is in the observation work, not the hot-path return | `cipher_rt_audit.c:78-97, 101-130` | ✓ — `pthread_mutex_lock` taken inside `cipher_rt_audit_record`, not on the dispatch fast path |
| Observation runs **inside** the hot path (synchronous) | `cipher_rt_audit.c:101-115, 117-130` | ✓ — note this is NOT off-thread; the hot-path cost is the HMAC + 1 mutex acquisition per launch |

**Finding A.5.NEW** (not in plan): the AUDIT observation is **synchronous on the hot path** — every cuBLAS call and every SDPA call acquires `g.mu`, runs FNV-1a + HMAC-SHA256, then releases. With multiple tenants making concurrent calls, this is a **single global lock** on the audit ring. The plan's Section 5 budget of "~50 ns observer" per AUDIT call is optimistic if HMAC-SHA256 hardware acceleration (SHA-NI / Intel SHA Extensions) is not present on the host CPU. **Recommendation:** verify SHA-NI is available on the production CPU (Lambda H100 pod CPU is Intel Xeon, has SHA-NI); document the per-call cost.

**Severity: MINOR.** The pattern works. The lock contention may show up at high tenant counts; AUDIT is env-gated off by default so production deployments that don't enable CIPHER_AUDIT incur zero cost.

### A.6 — Summary of Section A

- Two registry instances exist; both verified.
- Five semantic divergences between them — the classifier substrate must explicitly pick a contract per A.3.
- AUDIT priority-0 PASSTHROUGH template verified end-to-end. One minor finding (synchronous HMAC on hot path under global lock — acceptable when env-gated off).
- No third hidden registry instance.

**Plan corrections needed:**
1. §4.0 must specify the classifier substrate contract per A.3 (the plan currently implies both registries are uniform; they are not).
2. §4.0 should note that attn substrate's T4.6.1 observe-only ALWAYS calls orig regardless of HANDLED — future T4.6.3 actuator work must change this (the plan defers SUBSTITUTE Koopman to v2, which is correct, but the SDPA-actuator port path needs the trampoline fix).

---

<a id="section-b"></a>
## SECTION B — OP DEPENDENCY CHAIN VERIFICATION

Section B is **deferred to the companion document**: `cipher-fusion-evidence/CIPHER_REENGINEERING_VERIFICATION_PHASE_B.md` (771 lines, ~9,270 words). It walks all 27 NOT-PORTED ops (plus 3 verified additions for 30 total) with full dependency chains.

**Summary of Phase B findings (for this report):**

### B.1 — 6-of-33-ops-fire claim — **5 of 6 verified, 1 correction**

| Op | Plan claim | Verification |
|---|---|---|
| SUBSTITUTE / Marlin | `cipher_rt_marlin_actuator.c:198` registration; M≤64, K%128==0, STABILITY_THRESHOLD=4 | ✓ confirmed at L198 / L47 / L106 / L118 / L48 |
| ARBITRATE / CP54 | `cipher_cp54_sched.c` 15×8-SM ledger, NRs 13/14/15 + 16-20 | ✓ confirmed at L60-62 + NR handlers |
| VOLT | `cipher_rt_volt.c` NVML→kmod fall-through, [210, 1980] MHz | ✓ confirmed at L293 / L300-321 |
| AUDIT | priority-0 on both substrates, env-gated CIPHER_AUDIT | ✓ confirmed at L133, L136, L182 |
| COMMIT FSM token | Plan says "kmod cp54_sched (Track 3 DSM ACK-commit, Track 2 weight-arena commit-on-publish) — not as a separately-named op" | ✓ partially confirmed: `cp54_commit_migration` at `cipher_cp54_sched.c:383`, `CIPHER_CP54_MIGOUT_COMMITTED` at L395. **Plan correction:** in `cipher_weight_arena.c` "commit" appears only as a code comment at L262, NOT as a state-machine token. Drop the weight-arena half of the COMMIT claim. |
| CLASSIFY at cipher_cupti.c:166 | "comment-only stream-routing diagnostic" | ✓ confirmed comment-only; substantive may13 OpClass classifier (`include/cipher_classify.hpp`, 250 LOC, 7 classes) NOT ported |

### B.1.7 — **CRITICAL PLAN CORRECTION**: cipher_dispatch.cpp line numbers

The plan's §1.1 and §2.1 cite PASS_THROUGH lines as `541, 560-561, 570-573, 577-579, 584` and says "file is 543 lines."

**Reality:**
- `cipher-may13-evidence/cipher_dispatch.cpp` (TOP-LEVEL, the file the plan instructs to port) is **543 LOC** ✓ matches plan length claim.
- BUT the line citations 541/560-561/570-573/577-579/584 map to `cipher-may13-evidence/src/cipher_dispatch.cpp` (the **616-LOC src/ shadow** that is silently excluded by Makefile L29).

The same PASS_THROUGH/SUBSTITUTED logic lives in the TOP-LEVEL file at:
- Main path: L448 (Layer 3 not initialized), L468 (ORACLE DENY), L479-480 (registry MISS — the documented "L3.5 EDMD pipeline — wired in Week 4-5"), L486 (error_bound > 0.01), L500 (substitute failed), L515 (CIPHER_SUBSTITUTED — the lone success).
- Plus an additional 3 PASS_THROUGH paths in the **classify-only branch** (the `if (!g_cipher.initialized)` fallback at L338-426): L373, L425, L445. Total = 8 PASS_THROUGH + 4 SUBSTITUTED returns.

**Plan correction:** Update plan §1.1 and §2.1 to cite top-level line numbers (448, 468, 480, 486, 500) and acknowledge the classify-only branch has 3 additional paths (373, 425, 445). The plan's Week-1 instruction to port the TOP-LEVEL file is **correct** — only the line citations need updating.

### B.2 — 30 NOT-PORTED ops documented

Phase B agent produced structured rows for all 30 ops (the plan's 27 + 3 verified extras). Per-op detail in the companion document. Summary:

| Tier | Op count | TRIVIAL | MODERATE | HARD |
|---|---|---|---|---|
| Classifier (CLASSIFY, ORACLE, SENSE, PREDICT, DETERMINISM) | 5 | 1 (DETERMINISM — 88 LOC, no deps) | 3 (SENSE 340 LOC, PREDICT 333 LOC, CLASSIFY backing) | 1 (ORACLE — depends on liquid state + EMA + N≤4 counter) |
| Actuator (SUBSTITUTE-Koopman, ORCHESTRATE, GENERATE, RING_WRITE, REMEMBER, VALIDATE, SPECULATE, ADAPT) | 8 | 0 | 2 (GENERATE, RING_WRITE) | 6 (Stage 1/2 chain — all depend on Koopman runtime dead code) |
| Observability (SHIELD, SUSTAIN, THERMOSTAT, PULSE, HIBERNATE, LOOP, CONTINUITY, PIPELINE, GUARD, TOPOLOGY, TRACE, FAIRNESS, CARBON, RECEIPT, COMPLY) | 15 | 4 (TOPOLOGY, DETERMINISM, TRACE, CARBON) | 8 | 3 (THERMOSTAT — needs dedicated `std::thread` + NVML dlsym; PULSE — NVML dlsym; HIBERNATE — NVML SetPowerManagementLimit NOT_SUPPORTED on this pod) |
| Other (OVERLAP, STRAGGLER) | 2 | 0 | 0 | 2 (Phase 6 NCCL family deferral) |

### B.3 — Dependency DAG — **shallow, no cycles**

Phase B agent's DAG:
```
Tier 1 (leaves — port standalone):
  CLASSIFY, ORACLE, RING_WRITE, TOPOLOGY, DETERMINISM
Tier 2 (depends on Tier 1):
  SENSE (needs CLASSIFY) — 10 OPS DEPEND ON SENSE (the brain hub)
Tier 3 (depends on SENSE + others):
  SHIELD, FAIRNESS, GUARD, LOOP, PIPELINE, RECEIPT, CARBON, SUSTAIN,
  CONTINUITY, PREDICT, THERMOSTAT, PULSE, HIBERNATE, TRACE
Tier 4 (aggregators):
  COMPLY (fans-in to GUARD/DETERMINISM/CARBON/RECEIPT/FAIRNESS/TOPOLOGY)
Tier 5 (deferred):
  REMEMBER → VALIDATE → SPECULATE → ADAPT → SUBSTITUTE-Koopman (chain into dead code)
```

**No circular dependencies in v1 scope.** SENSE is the load-bearing brain hub. Plan's Week 1→5 sequence is consistent with this DAG.

### B.4 — Show-stoppers (v2 deferrals) — **8 ops named**

Phase B agent confirms the plan's v2 deferrals:
1. **SUBSTITUTE-Koopman lane** — depends on the empty `cipher_recipes.cpp` registry being populated, which requires the dead `cipher_koopman_runtime.cpp`.
2. **REMEMBER** — chains to dead Koopman runtime.
3. **VALIDATE** — chains to REMEMBER.
4. **SPECULATE** — chains to REMEMBER.
5. **ADAPT** — chains to REMEMBER + EDMD.
6. **OVERLAP** — Layer-2 NCCL family (Phase 6).
7. **STRAGGLER** — NCCL cross-rank attribution (Phase 6).
8. **NCCL_P2P** — multi-node (Phase 6).

Plus **4 observer-only-by-hardware ports** (build but cannot fully actuate on this Lambda pod):
- HIBERNATE — NVML SetPowerManagementLimit NOT_SUPPORTED.
- PULSE Signal 2 — code structurally deferred at the impl level.
- SUSTAIN compression flag — actuator not implemented.
- SHIELD Protections 2/3 — flag-only, downstream consumer not implemented.

**See companion document for the full Phase B walk.**

---

<a id="section-c"></a>
## SECTION C — KMOD ABI ENUMERATION + v1 EXTENSION SCOPE

### C.1 — Current ABI (24 NRs) — handler line numbers verified

Read `/home/ubuntu/cipher_kmod/cipher_dev.c:251-303` (the unlocked_ioctl switch) and the per-feature handler files. Every NR's handler line confirmed:

| NR | Name | Handler function | File:line | What it does | Userspace caller |
|---|---|---|---|---|---|
| 1 | `CIPHER_REGISTER_TENANT` | `cipher_dev_register_tenant` | `cipher_dev.c:58-96` | Anti-spoof tenant identity stamp + identity-bridge (FNV-64) | `cipher_rt_phase4/cipher_tenant.c` |
| 2 | `CIPHER_SNAPSHOT` | — | `cipher_dev.c:297-300` | Reserved, returns `-ENOSYS` | (none) |
| 3 | `CIPHER_RESET` | — | `cipher_dev.c:297-300` | Reserved, returns `-ENOSYS` | (none) |
| 4 | `CIPHER_GET_VERSION` | — | `cipher_dev.c:297-300` | Reserved, returns `-ENOSYS` | (none) |
| 5 | `CIPHER_SUBMIT_GPU_STATE` | `cipher_dev_submit_gpu_state` | `cipher_dev.c:162-194` | CAP_SYS_ADMIN. Device-wide GPU state from cipher-gpustate daemon | `cipher-gpustate` daemon |
| 6 | `CIPHER_SUBMIT_PROCESS_UTIL` | `cipher_dev_submit_process_util` | `cipher_dev.c:196-223` | CAP_SYS_ADMIN. Per-process NVML GPM | `cipher-gpustate` daemon |
| 7 | `CIPHER_SUBMIT_LAUNCH_STATS` | `cipher_dev_submit_launch_stats` | `cipher_dev.c:225-249` | Anti-spoof CUPTI snapshot from workload | `cipher_rt_phase4/cipher_cupti.c` |
| 8 | `CIPHER_GET_TENANT_SNAPSHOT` | `cipher_dev_get_tenant_snapshot` | `cipher_dev.c:118-160` | Cross-process snapshot query | `cipher_rt_phase4/cipher_rt_tenant.cpp` |
| 9 | `CIPHER_REQUEST_SM_PARTITION` | `cipher_dev_request_sm_partition` | `cipher_dev.c:107-113` | **DEACTIVATED** — returns `-ENOSYS` (CP 5.4 supersession) | (none — was `cipher_rt_arbitrate.c`, retired) |
| 10 | `CIPHER_SET_CLOCK_MHZ` | `cipher_dev_set_clock_mhz` | `cipher_clock.c` | DVFS via `call_usermodehelper("nvidia-smi -lgc")` | `cipher_rt_phase4/cipher_rt_volt.c` |
| 11 | `CIPHER_SUBMIT_FLOP_SAMPLE` | `cipher_flops_submit` | `cipher_flops.c` | CAP_SYS_ADMIN. CUPTI PM Sampling daemon | `cipher_flopd` daemon |
| 12 | `CIPHER_QUERY_FLOPS` | `cipher_flops_query` | `cipher_flops.c` | Read device + per-tenant FLOP/MFU | tools |
| 13 | `CIPHER_CP54_ALLOCATE` | `cipher_cp54_ioctl_allocate` | `cipher_cp54_sched.c` | 8-SM group allocation | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 14 | `CIPHER_CP54_FREE` | `cipher_cp54_ioctl_free` | `cipher_cp54_sched.c` | Release allocation | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 15 | `CIPHER_CP54_QUERY` | `cipher_cp54_ioctl_query` | `cipher_cp54_sched.c` | Ledger snapshot | tools / kvdedup test paths |
| 16 | `CIPHER_CP54_SUBSCRIBE_MIGRATE` | `cipher_cp54_ioctl_subscribe_migrate` | `cipher_cp54_sched.c` | Track 3 opt-in | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 17 | `CIPHER_CP54_POLL_MIGRATE` | `cipher_cp54_ioctl_poll_migrate` | `cipher_cp54_sched.c` | Tenant polls migration state | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 18 | `CIPHER_CP54_START_MIGRATE` | `cipher_cp54_ioctl_start_migrate` | `cipher_cp54_sched.c` | PROPOSED → MIGRATING | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 19 | `CIPHER_CP54_ACK_MIGRATE` | `cipher_cp54_ioctl_ack_migrate` | `cipher_cp54_sched.c` | COMMIT or NACK | `cipher_rt_phase4/cipher_rt_green_ctx.c` |
| 20 | `CIPHER_CP54_COMPACT_MIGRATE` | `cipher_cp54_ioctl_compact_migrate` | `cipher_cp54_sched.c` | Force compaction (operator/test) | tools |
| 21 | `CIPHER_ARENA_REGISTER` | `cipher_wa_ioctl_register` | `cipher_weight_arena.c` | Producer fd custodian | `cipher_kv_bridge.cpp` |
| 22 | `CIPHER_ARENA_IMPORT` | `cipher_wa_ioctl_import` | `cipher_weight_arena.c` | Consumer fd import | `cipher_kv_bridge.cpp` |
| 23 | `CIPHER_ARENA_LEAVE` | `cipher_wa_ioctl_leave` | `cipher_weight_arena.c` | Voluntary deregister | `cipher_kv_bridge.cpp` |
| 24 | `CIPHER_ARENA_QUERY` | `cipher_wa_ioctl_query` | `cipher_weight_arena.c` | Enumerate arenas | tools |

**Plus the separate `/dev/cipher_kvdedup` device (magic 'K'):** 5 NRs (INIT/PUT/CONFIRM/FREE/STATS) handled in `cipher_kvdedup.c`. Wired through `cipher_rt_phase4/cipher_rt_kv_alloc.c:481, 488, 559, 600, 620, 648, 669` — verified, not stubs.

### C.2 — v1 (5-week integration) kmod ABI extension scope

Re-read each week of the plan's Section 7 against the kmod ABI:

| Week | Activity | New kmod NR needed? | Rationale |
|---|---|---|---|
| **Week 1** | Compile-port classifier ops (CLASSIFY, ORACLE, SENSE) | **NO** | These ops operate on userspace state (per-tenant TLS, in-process cache). No kernel-side store needed. |
| **Week 2** | Hot-path observe-only (CLASSIFY/SENSE/ORACLE fire telemetry) | **NO** | Telemetry counters live in `/proc/cipher/classify_stats` — would be a new /proc node, not a new ioctl NR. |
| **Week 3** | Dispatch routing live + fleet-DVFS policy | **NO** | Uses existing NR 10 (SET_CLOCK_MHZ). Fleet-policy worker is userspace, reads `/proc/cipher/stats` and writes via nr 10. |
| **Week 4** | Observability tier port (FAIRNESS, CARBON, RECEIPT, etc.) + KV-dedup live-wire + Nemotron + Prometheus | **POSSIBLY YES — see C.3** | FAIRNESS needs per-tenant quota state; the kmod snapshot already has `fairness_quota_remaining_pct` field (`cipher_internal.h:161`) but `cipher_state_updater.c:129-130` explicit comment "FAIRNESS table (P4.7); the kthread is the right place to compute it once FAIRNESS is [implemented]" → the state-updater needs to populate this, which may need a new NR for userspace FAIRNESS to push per-tenant quota updates. |
| **Week 5** | CP 5.5 100-tenant benchmark | **NO** | Measurement only. |
| Post-v1 — L2 budget enforcement (per plan §3.5) | New NR 25 | **YES** | `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` per-tenant requires kmod-side accounting (depth win #3). |

### C.3 — Proposed new NRs (v1 + v1.5)

| Proposed NR | Name | Direction | Payload | Purpose | Userspace caller | When |
|---|---|---|---|---|---|---|
| **25** | `CIPHER_FAIRNESS_SET_QUOTA` | `_IOW` | `struct cipher_fairness_quota` (tenant_id_str, quota_units, period_ns) | Push per-tenant FAIRNESS quota for state_updater to track | userspace FAIRNESS op | **v1 Week 4** (only if FAIRNESS goes beyond observe-only) |
| 26 | `CIPHER_PREDICT_PUSH_HOT_REGION` | `_IOW` | `struct cipher_hot_region` (devptr, size_kb, layer_idx) | PREDICT informs kmod of hot L2 regions; state_updater writes `predicted_hot_regions[8]` slots | userspace PREDICT op | **v1.5** |
| 27 | `CIPHER_L2_BUDGET_SET` | `_IOW` | `__u32 budget_kb` | Per-tenant L2 budget (CU_LIMIT_PERSISTING_L2_CACHE_SIZE) | userspace runtime | **post-v1 depth-win program** |
| 28 | `CIPHER_CARBON_REGION_SET` | `_IOW` | `struct cipher_carbon_region` (region_name[32], grams_co2_per_kwh) | Operator-supplied marginal carbon factor | userspace CARBON op | v2 |
| 29 | `CIPHER_KOOPMAN_SURROGATE_ADD` | `_IOW` | `struct cipher_surrogate` (op_class, shape_hash, kernel_id, error_bound) | Dynamic registry insertion for Goal-4 v2 work | userspace EDMD pipeline | v2 |

**Plan Section 3.5 had this list shorter and less specific.** Update plan §3.5 with the C.3 table above. Note specifically:
- **NR 25 FAIRNESS_SET_QUOTA is the only one v1 might need.** And only IF FAIRNESS moves past observe-only (the plan's Section 7 Week 4 description suggests it does: "Per-tenant accounting via AUDIT chain becomes the customer-facing artifact"). If FAIRNESS ships observe-only in v1, no new NR needed.
- **NR 27 L2_BUDGET_SET is post-CP-5.5**, aligns with the depth-win program (Section 5.1 corrected timeline).

### C.4 — kvdedup ABI integration — VERIFIED

The plan says the 5 kvdedup NRs (magic 'K') are wired through `cipher_kv_bridge.cpp` (vLLM bridge). Verified by grep:

| NR | Name | Handler | Wired in userspace |
|---|---|---|---|
| 1 | `CIPHER_KVDEDUP_INIT` | `kvd_ioctl_init` | `cipher_rt_kv_alloc.c:490` — `ioctl(d.kvd_fd, CIPHER_KVDEDUP_INIT, &ini)` |
| 2 | `CIPHER_KVDEDUP_PUT` | `kvd_ioctl_put` | `cipher_rt_kv_alloc.c:563, 621` — both first-PUT and FORCE_NEW PUT |
| 3 | `CIPHER_KVDEDUP_CONFIRM` | `kvd_ioctl_confirm` | `cipher_rt_kv_alloc.c:602` |
| 4 | `CIPHER_KVDEDUP_FREE` | `kvd_ioctl_free` | `cipher_rt_kv_alloc.c:650` |
| 5 | `CIPHER_KVDEDUP_STATS` | `kvd_ioctl_stats` | `cipher_rt_kv_alloc.c:669` |

**No TODOs or stubs in the kvdedup path.** The kvdedup substrate is **substrate-complete on Tree B**. The plan's R-A3 risk ("kvdedup live-wiring breaks bit-identical correctness, T4.6.5 not done") is about wiring this PROVEN PUT/CONFIRM/FREE substrate to the **live decode path** in vLLM — i.e. the vLLM model integration layer, not the substrate itself.

**Verified: the kmod substrate is ready. The integration risk is upstream (cipher_vllm_plugin layer).**

---

<a id="section-d"></a>
## SECTION D — PASS_THROUGH + DEAD-CODE RE-VERIFICATION

### D.1 — cipher_dispatch.cpp PASS_THROUGH paths — **PLAN CORRECTED**

Read `/home/ubuntu/cipher-may13-evidence/cipher_dispatch.cpp` (TOP-LEVEL, the file the plan says to port) in full. The file is **543 LOC**.

**Plan claim (§1.1):** "Falls through to `CIPHER_PASS_THROUGH` at lines 541/560-561/570-573/577-579/584. Line 480 is the documented 'L3.5 EDMD pipeline — wired in Week 4-5' deferral. Only one code path reaches `return CIPHER_SUBSTITUTED` at line 589."

**Reality (top-level file, verified by `grep -n "return CIPHER_PASS_THROUGH\|return CIPHER_SUBSTITUTED"`):**

8 PASS_THROUGH returns and 4 SUBSTITUTED returns at:

| Line | Path | What triggers |
|---|---|---|
| L373 | classify-only branch (`if (!g_cipher.initialized)`) | ORACLE DENY in classify-only path |
| L388 | classify-only branch | `return CIPHER_SUBSTITUTED` — GEMM registry HIT in classify-only path |
| L403 | classify-only branch | `return CIPHER_SUBSTITUTED` — non-GEMM Chebyshev registry HIT |
| L418 | classify-only branch | `return CIPHER_SUBSTITUTED` — synthetic-entry Chebyshev |
| L425 | classify-only branch end | Default fall-through if no apply_recipe succeeded |
| L445 | Main path | Fast exit: ITERATIVE_CUSTOM with confidence < 60 |
| L448 | Main path | Layer 3 not initialized |
| L468 | Main path | ORACLE DENY |
| **L480** | **Main path** | **Registry MISS — "L3.5 EDMD pipeline — wired in Week 4-5" — THE LOAD-BEARING GOAL-4 PASS-THROUGH** |
| L486 | Main path | entry->error_bound > 0.01f |
| L500 | Main path | substitute failed |
| L515 | Main path | `return CIPHER_SUBSTITUTED` — the lone success path |

**Plan's line citations match a DIFFERENT FILE:**

`/home/ubuntu/cipher-may13-evidence/src/cipher_dispatch.cpp` (the silently-excluded shadow copy, **616 LOC**) has the same logic at lines 541, 560-561, 570-573, 577-579, 584. The plan's Section 1.1 says "file is 543 lines" — this matches the TOP-LEVEL file, but the line numbers cited match the SRC/ shadow.

**Conclusion**: The plan was reading the shadow file's line numbers while documenting the top-level file's purpose. The plan's Week-1 instruction to port the TOP-LEVEL file is CORRECT — only the line citations must be updated.

**The conceptual claim — Goal-4 never fires because the EDMD pipeline at the registry-miss gate was never wired (the "wired in Week 4-5" comment) — is VERIFIED.** L480 of the top-level file is that gate. The registry is seeded with 32+ static entries that no real workload's M/N/K hash matches.

### D.1.NEW — Additional finding the plan missed: classify-only-branch PASS_THROUGHs

The plan's PASS_THROUGH inventory covers only the main path. The TOP-LEVEL file's L338-426 is an entire SECOND dispatch path: the "classify-only" branch that fires when `!g_cipher.initialized` — i.e. on every kernel before `cipher_init()` completes. This branch has its own:
- Lazy registry init (L351-356)
- Oracle gate (L358-374) — PASS_THROUGH at L373
- GEMM registry lookup with apply_recipe (L377-391) — SUBSTITUTED at L388
- Non-GEMM Chebyshev path (L392-421) — SUBSTITUTED at L403/L418
- Default fall-through at L425

**Implication for the port**: when the top-level `cipher_dispatch.cpp` ports to `cipher_rt_phase4/cipher_rt_dispatch.cpp`, the classify-only branch must port too — otherwise the runtime loses its pre-init dispatch path. Add to plan §7 Week 1 file-port list.

**Severity: MINOR.** The plan's port instruction implicitly carries this branch (entire file ports as one). But the line-citation correction in D.1 should also note the existence of this second dispatch path so the reviewer doesn't think L373/L425 are spurious extras.

### D.2 — Koopman runtime dead-code claim — **VERIFIED**

Grep across all CIPHER source trees for callers of `cipher_kr_*`:

```
/home/ubuntu/cipher-may13-evidence/tests/test_l11.cpp                  # TEST CODE ONLY
/home/ubuntu/cipher-may13-evidence/include/cipher_koopman_runtime.h    # HEADER (declarations)
/home/ubuntu/cipher-may13-evidence/src/cipher_koopman_runtime.cpp      # IMPL (self-references)
```

Self-references inside `cipher_koopman_runtime.cpp`: 15 (normal — internal function calls). **External callers outside the file + header + test: ZERO.**

**Plan claim VERIFIED.** The "Runtime Koopman derivation" module is built into `libcipher_rt.so.preroadmap` (per the may13 Makefile) but no production code path invokes it. The plan's deferral of Goal-4 to v2 is correctly grounded.

### D.3 — Recipes registry seeding — **PLAN CORRECTED (count)**

Read `/home/ubuntu/cipher-may13-evidence/src/cipher_recipes.cpp:346` onward in full.

**Plan claim (§2.1):** "32 hardcoded Llama-3-70B-scale shapes seeded at init (L346); no real workload emits those exact shapes; therefore `apply_recipe()` is never reached."

**Reality:** The registry seeds **up to 42 entries** in a mix of categories:

| Entry range | Category | Count |
|---|---|---|
| 0 | Square GEMM (M=N=K=4096) | 1 |
| 1-3 | Llama-3-70B attention + FFN shapes | 3 |
| 4 | SOMA motor control (CfC) | 1 |
| 5-10 | 6 Llama-3 attention shapes (decode/prefill/seq8k) | 6 |
| 11-16 | 6 RMSNorm Chebyshev shapes | 6 |
| 17-22 | 6 elementwise activation Chebyshev shapes | 6 |
| 23-26 | 4 HyperFlux gaming physics surrogates | 4 |
| 27-31 | Up to 6 A100 (sm_80) versions of core shapes (bounded `reg->count < 32`) | up to 5 |
| 32-41 | 10 "shape-parametric" entries with `hash_shape(0, K, N)` — added beyond the 32 cap (uses `CIPHER_REGISTRY_MAX_ENTRIES`) | 10 |
| **Total** | | **~42 entries** |

**Plan correction:** §2.1 should say "**32+ hardcoded entries** spanning gemm/Chebyshev/HyperFlux categories, not exclusively Llama-3-70B shapes."

### D.3.NEW — Latent bug: shape-parametric entries NEVER MATCH

The shape-parametric block at the end (entries 32-41) is keyed on `hash_shape(0, K, N)` — i.e. with M=0. The lookup at `cipher_dispatch.cpp:153` calls `fnv_shape(m, n, k)` with the **actual** M from runtime. Since real workloads always have M ≥ 1, **these 10 shape-parametric entries never match in practice**. They appear to be a partial attempt to add 7B FFN shape coverage (gemm-Kx4096-N14336 etc.) that was never finished.

**Severity: COSMETIC** — these entries are inert. They don't break anything, but they also can never lift performance. The plan's Goal-4-deferral correctly skips this work; the entries can be deleted in the v1 port or left in place as a v2 work item ("complete the shape-parametric matcher").

### D.3 — Default error_bound for seeded entries

| Entry category | error_bound | Will L486 (`> 0.01f`) PASS_THROUGH? |
|---|---|---|
| Llama-3-70B GEMMs (0-3) | 0.005f | NO — passes (<0.01) |
| SOMA (4) | 0.001f | NO — passes |
| Llama-3 attention (5-10) | 0.005f | NO — passes |
| RMSNorm Chebyshev (11-16) | 0.001f | NO — passes |
| Elementwise Chebyshev (17-22) | 0.002f | NO — passes |
| HyperFlux (23-26) | 0.01f | **YES — exactly at boundary** — `> 0.01f` is false, so passes; but operator strictly `>` not `>=` |
| A100 versions (27-31) | 0.005f | NO — passes |
| Shape-parametric (32-41) | 0.01f | YES — same boundary case |

**Plan claim (§5.6 et al.):** correct that error_bound > 0.01 triggers PASS_THROUGH — but this is moot because of D.3.NEW (the high-error-bound entries are the shape-parametric ones that never match anyway).

### D.4 — Stage-1/2 thread spawn-gate — **VERIFIED**

The plan §2.1 says: "Lazy-spawn-gated: at line 949-965, threads spawn only if `observers_enabled > 0` (sum of 20 observer `_init()` returns); default-off observers → no threads → 6 core ops NEVER FIRE in production."

Direct read of `cipher_10ops_impl.cpp:949-965`:

```cpp
// Lazy spawn: only start Stage 1/2 threads if at least one observer is
// enabled. CIPHER_NO_BG_THREADS=1 forces skip for diagnostic bisects
// regardless of observer state.
const char* no_bg = getenv("CIPHER_NO_BG_THREADS");
int skip_bg = (no_bg && (no_bg[0] == '1' || no_bg[0] == 'o' || no_bg[0] == 'O'));
if (!skip_bg && observers_enabled > 0) {
    pthread_create(&t1, &attr, stage1_shadow,     NULL);
    pthread_create(&t2, &attr, stage2_background, NULL);
    fprintf(stderr, "[CIPHER 10ops] Stage 1/2 threads spawned "
                    "(observers_enabled=%d)\n", observers_enabled);
} else {
    (void)t1; (void)t2;
    fprintf(stderr, "[CIPHER 10ops] Stage 1/2 threads skipped — "
                    "%s\n",
            skip_bg ? "CIPHER_NO_BG_THREADS=1"
                    : "no observers enabled (lazy-start)");
}
```

**Plan claim VERIFIED.** Two-clause gate: `!skip_bg && observers_enabled > 0`. The 6 core Stage 1/2 ops (REMEMBER, VALIDATE, AUDIT, SPECULATE, ADAPT, ARBITRATE) all depend on these threads firing. In production with no `CIPHER_*=1` observer envvars set, `observers_enabled == 0` and the threads never start.

**Stage banner at L971-975 lists** Stage 1: REMEMBER+VALIDATE+AUDIT+SPECULATE; Stage 2: ADAPT+ARBITRATE. Matches plan.

### D.5 — src/ shadow vs top-level diff — **PLAN CORRECT, with one nuance**

Plan §1.1: "src/cipher_dispatch.cpp ... canonical version is top-level. Action: discard, use top-level."

Direct `diff /home/ubuntu/cipher-may13-evidence/cipher_dispatch.cpp /home/ubuntu/cipher-may13-evidence/src/cipher_dispatch.cpp` shows:
- TOP-LEVEL includes `<dlfcn.h>` (for weak symbol resolution)
- SRC/ includes `<math.h>` (different — older form)
- TOP-LEVEL has `cipher_tls_get_gemm_types` weak symbol + `cipher_edmd_live_collect` weak hook + ~50 LOC of `edmd_live_post_relaunch_hook` function
- SRC/ does NOT have these

**The src/ copy is a strictly OLDER version.** Plan's Action ("discard, use top-level") is correct. The 73 LOC delta (616 - 543) is the absence of the EDMD-live hook code in the older src/ version — meaning the src/ copy ALSO has the PASS_THROUGH gates at different line numbers (541/560-561/570-573/577-579/584 per plan citation, which means src/ has them spread out over more lines because of the missing inline expansion).

**Plan action correct; line-number citation needs to come from the top-level file (D.1).**

---

<a id="section-e"></a>
## SECTION E — INTEGRATION BLOCKERS NOT YET SURFACED

### E.1 — Makefiles inspection — **clean**

Read in full:
- `cipher-may13-evidence/Makefile` (190 LOC) — uses gcc/g++/nvcc, includes `-I./include`, builds 3 DSOs as documented in plan §1.1.
- `cipher_rt_phase4/Makefile` (137 LOC) — uses gcc/g++ + libpthread + libcupti + libcuda + libdl + libcrypto + libc10 (NOT libtorch_cpu). 16 OBJS as documented in plan §1.2.
- `cipher_kmod/Kbuild` (23 LOC) — 14 obj files, `ccflags-y := -Wall`.
- `cipher_kmod/Makefile` — wrapper for kernel-out-of-tree build.

**No surprise compile-time dependencies surfaced beyond plan §3.3.**

One nuance worth noting: the plan's Section 3.3 says "Adding nvcc is required only if Koopman O(1) substitute kernels are ported. For v1 — deferred." Verified: the Tree B Makefile has zero CUDA `.cu` rules. Adding nvcc to Tree B's build for v1 op ports is unnecessary; all the may13 ops being ported in v1 (classifier, observability) are pure `.cpp`/`.c`. **Plan §3.3 correct.**

### E.2 — LD_PRELOAD init order — **single-threaded, no race**

Plan §1.2 lists 14 init steps in `cipher_v2_init_body`. Verified by reading `cipher_inject.c` in full (76 LOC):

```c
static pthread_once_t cipher_v2_init_once = PTHREAD_ONCE_INIT;

static void cipher_v2_init_body(void)
{
    cipher_dbg("init body running");
    (void)cipher_v2_tenant_register();      /* ioctl nr 1 */
    (void)cipher_rt_green_ctx_cp54_init();  /* CP 5.4 ioctl nr 13 */
    (void)cipher_rt_smp_init();
    (void)cipher_rt_pr_init();
    (void)cipher_v2_cupti_init();           /* CUPTI subscribe */
    (void)cipher_rt_volt_init();            /* env-gated DVFS */
    (void)cipher_rt_matmul_dispatch_init();
    (void)cipher_rt_marlin_init();          /* env-gated CIPHER_MARLIN */
    (void)cipher_rt_attn_dispatch_init();
    (void)cipher_rt_attn_test_actuator_init(); /* env-gated CIPHER_ATTN_TEST */
    (void)cipher_rt_audit_init();           /* env-gated CIPHER_AUDIT */
    cipher_rt_cublas_shim_register_got();
    cipher_rt_attn_register_got();
    (void)cipher_rt_got_patch_init();
}

int InitializeInjection(void *pfnGetExportTable) {
    (void)pfnGetExportTable;
    pthread_once(&cipher_v2_init_once, cipher_v2_init_body);
    return 1;
}

int InitializeInjection2(void) {
    pthread_once(&cipher_v2_init_once, cipher_v2_init_body);
    return 1;
}
```

**The entire init body runs under `pthread_once` — single-threaded by design.** No race between the 14 steps. They run sequentially in the listed order.

**Plan §1.2 init order matches the code exactly.** Adding 7 new classifier-init calls (Section 4 of inspection plan) is straightforward: append them to `cipher_v2_init_body` in the right slot. Recommended slot for classifier inits is between `cipher_rt_pr_init()` and `cipher_v2_cupti_init()` — before CUPTI subscribes, after partition router is ready.

### E.2.NEW — One subtle dependency the plan didn't name

`cipher_v2_cupti_init` (line 5 in the init sequence) subscribes to CUPTI callbacks that **immediately start firing** on `cudaLaunchKernel` / `cuLaunchKernel`. If the dispatch routing (Week 3) goes live before the CUPTI subscription has stable telemetry, the per-tenant snapshot may be stale and the dispatcher may make decisions on stale data.

**Mitigation**: Week 3 dispatch-routing-live should add a CUPTI-warmup delay (e.g. wait for ≥1 second of telemetry, or ≥100 launch counters) before flipping `CIPHER_DISPATCH_LIVE=1`. Add to plan §7 Week 3 risk register (R-W3.4).

### E.3 — do_exit reaper — **plan correct; new ops do not extend it**

Read `cipher_kmod/cipher_probe.c:170-221` in full.

```c
static int cipher_do_exit_pre(struct kprobe *p, struct pt_regs *regs)
{
    pid_t pid = current->pid;
    struct cipher_pid_stats *e;

    /* CP 5.4: release any 8-SM-group allocation held by this tenant —
     * UNCONDITIONALLY, before the cipher_pid_stats fast-path guard below.
     * ... */
    cipher_cp54_release(pid);

    rcu_read_lock();
    e = cipher_pid_lookup_rcu(pid);
    rcu_read_unlock();
    if (likely(!e))
        return 0;

    /* Release any SM partition slots held by this tenant before tearing
     * down the cipher_pid_stats entry. ... */
    cipher_partition_release_slots_only(pid);

    spin_lock(&cipher_pid_insert_lock);
    e = cipher_pid_lookup_rcu(pid);
    if (e) {
        hash_del_rcu(&e->node);
        kfree_rcu(e, rcu);
        atomic64_inc(&cipher_reaped_count);
    }
    spin_unlock(&cipher_pid_insert_lock);
    return 0;
}
```

Order of cleanup:
1. CP 5.4 ledger release (unconditional, lock-free).
2. RCU fast-path: if no `cipher_pid_stats` entry, return.
3. Legacy partition slot release (`cipher_partition_release_slots_only`).
4. Lock + hash_del_rcu + kfree_rcu.

**Plan claim VERIFIED.** Cleanup is correctly ordered to avoid races with active CUDA streams (the kmod ledger is independent of cipher_pid_stats; the legacy slot release happens before the entry is freed).

**Do v1 new ops need to extend the reaper?**
- Classifier ops (CLASSIFY, ORACLE, SENSE, PREDICT, DETERMINISM): per-tenant state is in `cipher_pid_stats` or in libcipher_rt-process local memory. When the process exits, both vanish naturally. **No reaper extension needed.**
- Observability ops (FAIRNESS, CARBON, RECEIPT, etc.): per-session state in `cipher_pid_stats`. **No reaper extension needed.**
- KV-dedup wire (Week 4): uses `/dev/cipher_kvdedup` separate device. That device has its own `release` fop that handles per-fd tenant cleanup. **No reaper extension needed.**

**Plan correct: v1 reaper unchanged.**

### E.4 — Env vars, runtime gates, TODOs — **mostly complete; one gap surfaced**

Plan §1.5 mentions per-thread fd rule and CIPHER_AUDIT/CIPHER_MARLIN. Full env-var inventory by grep:

| Env var | Used in | What it does | Plan mentions? |
|---|---|---|---|
| `CIPHER_TENANT_ID` | `cipher_v2_internal.h:22`, `cipher_tenant.c` | Required for tenant registration | ✓ implicit |
| `CIPHER_V2_DEBUG` | `cipher_v2_internal.h:23` | Verbose logging | ✗ plan doesn't mention |
| `CIPHER_QOS_CLASS` | `cipher_rt_green_ctx.c:121` | PARTITION / SHARED / POOL qos | ✓ |
| `CIPHER_SM_COUNT` | `cipher_rt_green_ctx.c:148` | Partition SM count request | ✓ |
| `CIPHER_MIGRATABLE` | `cipher_rt_green_ctx.c:176` | Opt-in to Track 3 DSM | ✓ |
| `CIPHER_SC3_FAULT` | `cipher_rt_green_ctx.c:430` | Track 3 SC3 fault injection (test hook) | ✗ test-only, ok |
| `CIPHER_MARLIN` | `cipher_rt_marlin_actuator.c:178` | Enable Marlin actuator | ✓ |
| `CIPHER_MARLIN_VERBOSE` | `cipher_rt_marlin_actuator.c:179` | Verbose Marlin | ✗ |
| `CIPHER_VOLT` | `cipher_rt_volt.c:255` | Enable DVFS | ✓ |
| `CIPHER_VOLT_BATCH` | `cipher_rt_volt.c:256` | Batch-aware DVFS | ✗ |
| `CIPHER_VOLT_MHZ` | `cipher_rt_volt.c:257` | Manual clock-MHz | ✓ |
| `CIPHER_ATTN_TEST` | `cipher_rt_attn_test_actuator.c:81, 92` | Enable attn test actuator | ✗ |
| `CIPHER_AUDIT` | `cipher_rt_audit.c:182` | Enable AUDIT | ✓ |
| `CIPHER_AUDIT_DUMP` | `cipher_rt_audit.c:153` | Path to dump audit ring at exit | ✗ |
| `CIPHER_NO_BG_THREADS` | `cipher_10ops_impl.cpp:952` | Force-skip Stage 1/2 thread spawn (diagnostic) | ✓ |
| `CIPHER_PER_THREAD_FD` | (per plan §4.1, used in test harnesses) | Verify per-thread fd contention | ✓ |

**Env vars the plan should document:**
- `CIPHER_V2_DEBUG` — verbose logging toggle.
- `CIPHER_MARLIN_VERBOSE`, `CIPHER_VOLT_BATCH`, `CIPHER_AUDIT_DUMP` — operator knobs.
- `CIPHER_ATTN_TEST` — diagnostic actuator (already covered in plan §4.0).

**Severity: COSMETIC.** Plan §1.5 / §4.x can be augmented with a complete env-var table for the operator/DD reader.

**TODOs/FIXMEs in deployed runtime:** `grep -n "TODO\|FIXME\|XXX"` across `cipher_rt_phase4/` produces **zero hits** in active source files. The deployed runtime is comment-clean. (TODOs exist in may13 src/ tree at expected locations: `cipher_dispatch.cpp:478` "L3.5 EDMD pipeline — wired in Week 4-5" comment is structurally a deferred TODO; `cipher_attn_koopman.cpp:495-499` "fused_path_not_yet_wired"; etc. — all documented in audit_section_4.)

### E.4.NEW — Module parameters (kmod sysfs equivalent of env vars)

`module_param` declarations in the kmod (writable post-load via `/sys/module/cipher_kmod/parameters/`):

| Module param | File:Line | Default | Permission | Purpose |
|---|---|---|---|---|
| `sm_count` | `cipher_partition_allocator.c:73` | 132 | 0444 (read-only) | Legacy allocator SM count override |
| `cipher_cp54_mig_gap_min_grps` | `cipher_cp54_sched.c:99` | 1 | 0644 | Track 3 DSM compaction trigger |
| `cipher_cp54_mig_sustain_ms` | `cipher_cp54_sched.c:100` | 2000 | 0644 | Track 3 sustain window |
| `cipher_cp54_mig_ratelimit_ms` | `cipher_cp54_sched.c:101` | 10000 | 0644 | Track 3 per-tenant rate limit |
| `cipher_cp54_mig_verbose` | `cipher_cp54_sched.c:102` | 0 | 0644 | Track 3 dmesg verbosity |

**Finding E.4.NEW:** Plan doesn't mention these are operator-tunable at runtime via sysfs. For the data-room writeup, add a note: "DSM policy parameters are runtime-tunable without kmod reload via `/sys/module/cipher_kmod/parameters/`." This is operator-friendly and a competitive feature.

**Severity: COSMETIC.**

### E.5 — Compile-time gates surfacing

Per CLAUDE.md context: `CIPHER_CPU_STUB` is a conditional for Mac/CPU builds (cipher_stubs.h). Verified by grep — present in ~10 may13 files. Tree B does not use it. **Plan §3.3 implicitly handles this** (porting may13 ops to Tree B drops the stub guard).

`CIPHER_HAVE_OPENSSL` toggle in `cipher_10ops_impl.cpp:978-982` selects HMAC-SHA256 vs XOR chain. Tree B's `cipher_rt_audit.c` always uses OpenSSL (links `-lcrypto`). **No plan impact** — the AUDIT port in Week 4 retires the may13 `cipher_10ops_impl.cpp` AUDIT block entirely in favor of `cipher_rt_audit.c`'s OpenSSL path.

---

## CROSS-CUTTING SUMMARY

### Plan claims verified (16)

1. Two actuator-registry instances (matmul + attn). [§A.4]
2. AUDIT priority-0 always-PASSTHROUGH on both registries. [§A.5]
3. 24 NRs on `/dev/cipher` with reserved 2/3/4. [§C.1]
4. NR 9 (REQUEST_SM_PARTITION) returns -ENOSYS. [§C.1]
5. CP54 ledger + Track 3 migration FSM at NRs 13-20. [§C.1]
6. Weight arena fd custodian at NRs 21-24. [§C.1]
7. kvdedup `/dev/cipher_kvdedup` 5 NRs, magic 'K', wired through cipher_rt_kv_alloc.c. [§C.4]
8. Koopman runtime is dead code (zero callers outside file + header + test). [§D.2]
9. Stage-1/2 thread spawn gate at `cipher_10ops_impl.cpp:954` on `!skip_bg && observers_enabled > 0`. [§D.4]
10. src/ shadow vs top-level diff: top-level has EDMD-live hook, src/ does not. [§D.5]
11. `pthread_once` makes init order single-threaded — no race between 14 init steps. [§E.2]
12. do_exit reaper at `cipher_probe.c:184` order is correct and lock-free fast path. [§E.3]
13. No TODOs/FIXMEs in deployed runtime active source. [§E.4]
14. SUBSTITUTE/Marlin gates (M≤64, K%128, STABILITY_THRESHOLD=4). [§B.1]
15. VOLT bounds [210, 1980] MHz. [§B.1]
16. Top-level `cipher_dispatch.cpp` is 543 LOC; src/ shadow is 616 LOC. [§D.1]

### Plan claims corrected (9)

1. **§1.1 / §2.1 cipher_dispatch.cpp line citations** map to src/ shadow (616 LOC), not top-level (543 LOC). Update to top-level: L448, L468, L480 (the load-bearing one), L486, L500. [§B.1.7, §D.1]
2. **§2.1 PASS_THROUGH count** — main path has 5 PASS_THROUGH + 1 SUBSTITUTED; classify-only branch adds 3 more PASS_THROUGH + 3 SUBSTITUTED returns (total 8 + 4). [§D.1.NEW]
3. **§4.0 actuator-registry uniformity** — 5 semantic divergences between matmul and attn. Classifier substrate must pick a contract per §A.3. [§A.3]
4. **§2.3 COMMIT FSM token** — only the kmod cp54_sched "COMMIT" state machine token is real; the weight_arena "commit" is just a comment string at L262. Drop the weight-arena half. [§B.1]
5. **§2.1 Recipe registry seed count** — "32 hardcoded Llama-3-70B-scale shapes" is wrong. Actually 32+ mixed entries (gemm/Chebyshev/HyperFlux/A100/shape-parametric). [§D.3]
6. **§5.6 R-A3 KV-dedup risk** — the kmod substrate is COMPLETE; the risk is the vLLM-bridge wiring (cipher_vllm_plugin), not the substrate itself. [§C.4]
7. **§3.5 ABI extension scope** — the proposed extensions list is shorter than reality. C.3 table provides the full v1/v1.5/v2 NR proposals (25-29). [§C.3]
8. **§4.0 attn substrate HANDLED bypass** — the trampoline ALWAYS calls orig regardless of HANDLED. T4.6.3 SDPA-actuator work must fix this. [§A.3 finding #5]
9. **§5.1 / §4.5 SDPA dispatch** — REDIRECTED enum value (2 on attn, ERROR on matmul) is a hidden cross-substrate semantic mismatch. [§A.3 finding #1]

### New findings (6) — traps the plan missed

| # | Finding | Severity | Section |
|---|---|---|---|
| F1 | Top-level `cipher_dispatch.cpp` has a SECOND dispatch path (the classify-only branch at L338-426) that the plan didn't document — the port must carry it. | MINOR (port instruction implicitly carries) | §D.1.NEW |
| F2 | The 10 shape-parametric registry entries (entries 32-41) use `hash_shape(0, K, N)` but the lookup uses `hash_shape(M, N, K)` — they NEVER match real workloads. Latent v1.5/v2 cleanup. | COSMETIC | §D.3.NEW |
| F3 | AUDIT observation is synchronous on the hot path under a global mutex. At high tenant counts the lock contention may exceed the "~50 ns" budget. Verify SHA-NI is available. | MINOR (env-gated off by default) | §A.5.NEW |
| F4 | Week 3 dispatch-routing-live should add a CUPTI-warmup delay before flipping `CIPHER_DISPATCH_LIVE=1` — otherwise routing decisions are made on stale telemetry. Add R-W3.4 to plan §7 / §8. | MINOR | §E.2.NEW |
| F5 | DSM policy parameters (`cipher_cp54_mig_*`) are operator-tunable at runtime via `/sys/module/cipher_kmod/parameters/`. Document as a competitive feature. | COSMETIC | §E.4.NEW |
| F6 | Env-var inventory has 7 undocumented vars (CIPHER_V2_DEBUG, CIPHER_MARLIN_VERBOSE, CIPHER_VOLT_BATCH, CIPHER_AUDIT_DUMP, CIPHER_ATTN_TEST, CIPHER_SC3_FAULT, CIPHER_PER_THREAD_FD). | COSMETIC | §E.4 |

### Show-stoppers — **NONE**

No finding blocks the 5-week v1 plan. The contract divergences (A.3) require a deliberate classifier-substrate spec; the line-number corrections (D.1) are pure-documentation; the v2 deferrals (B.4) match the plan. **The plan is trustworthy after the 9 corrections and 6 new findings above are folded back.**

---

## ACTION ITEMS FOR THE REENGINEERING PLAN

The following plan revisions are recommended, in priority order:

1. **§1.1 / §2.1 line citations** — replace src/-shadow line numbers (541/560-561/570-573/577-579/584) with top-level line numbers (448/468/480/486/500). Note the existence of the classify-only branch (L338-426) with 3 additional PASS_THROUGH paths.
2. **§4.0 classifier-substrate contract spec** — adopt the hybrid per §A.3 (4-value enum, single-arg maybe_handle, snapshot-under-lock, break-on-error).
3. **§7 Week 3 add R-W3.4** — CUPTI-warmup delay before dispatch-routing-live.
4. **§3.5 / §C.3 ABI extension table** — adopt the 5-row NR proposal (NR 25-29) with v1/v1.5/v2 scoping.
5. **§5.6 R-A3** — clarify that the kvdedup substrate is COMPLETE; the wiring risk is upstream at the cipher_vllm_plugin layer.
6. **§2.3 COMMIT FSM token** — restrict the claim to cp54_sched (drop weight_arena half).
7. **§2.1 Recipe registry** — update to "32+ mixed entries" and note the latent shape-parametric bug as v2 cleanup.
8. **§4.0 attn HANDLED trampoline bypass** — note that T4.6.3 SDPA-actuator port must fix the always-call-orig pattern.
9. **§4.x env-var inventory** — add the 7 missing vars.
10. **§4.x operator-tunable kmod params** — add the DSM policy parameter table.

---

## CLOSING NOTE

This inspection was line-by-line verification, not design. The reengineering plan is **trustworthy after 10 surgical corrections** documented above. The deployed runtime is in a healthier state than the plan suggested (zero TODOs, clean Makefiles, init order race-free, do_exit reaper correct). The v1 5-week plan is feasible.

The single biggest surprise was the actuator-registry contract divergence (§A.3). The plan declared the matmul and attn registries as "the same pattern" — they are nearly the same but not identical. The classifier-substrate port in Week 1 must explicitly pick a contract, not assume uniformity. This document provides the picked contract (§A.3 final block).

Reading the code first matters. Every claim above is grounded in a specific `file:line` citation, not in summary documents or memory pointers.

**File md5 at this snapshot:** to be computed by the user out-of-band after final read-through (writing the md5 changes the md5 — pre-stamp baseline captured separately).

— end of CIPHER_DEEP_INSPECTION_REPORT.md (v1.0, 2026-05-20) —
