# Week 3 Step 4 Option II — End-to-End Readiness Audit (Pre-Flight)

**HEADLINE STATUS: SCOPE-COMPRESSED.**

Sub-deliverables 1, 2, 3 are CLEAN-WIRE (~50 LOC total). Sub-deliverables 5 and 6 are SHIM-REQUIRED (~110 LOC; mechanical mirrors of Week 2 Step 5/6 ioctl patterns). **Sub-deliverable 4 is DESIGN-MISMATCH**: SENSE does NOT produce migration proposals — it classifies sessions (HUMAN/AGENT/BATCH/UNKNOWN). Wave 5 §5.5 W3's "SENSE → DSM PROPOSE" prescribes a transition-detection wrapper that does NOT exist. Building it adds ~200-300 LOC of new code and 2-3h on top of the original Option II budget.

| sub-deliverable | category | LOC | risk | notes |
| --- | --- | ---:| --- | --- |
| 1 — CUPTI → ring bridge | **CLEAN-WIRE** | ~20 | low | Stack-construct `CipherRingEntry`; no `g_cipher_10ops` needed; `cipher_10ops_impl.cpp` port NOT required |
| 2 — `cipher_sense_init` wiring | **CLEAN-WIRE** | ~5 | low | Idempotent + env-gated (`CIPHER_SENSE` default OFF); call from existing `cipher_v2_init_body` |
| 3 — Per-launch `cipher_sense_observe` | **CLEAN-WIRE** | ~10 | low | Single relaxed atomic load when env-OFF (~3 cycles); ~30ns hot-path even when ON |
| 4 — SENSE → proposal accessor | **DESIGN-MISMATCH** | ~250 NEW | **medium-high** | SENSE classifies sessions; doesn't propose migrations. Wave 5's "proposal" must be a NEW transition-detection wrapper (~200-300 LOC of fresh code) |
| 5 — `CIPHER_DSM_PROPOSE` ioctl ABI | SHIM-REQUIRED | ~30 | low | nr 26 free; mirror Week 2 Step 6 pattern |
| 6 — Kmod handler + proposals queue | SHIM-REQUIRED | ~80 | low | Pattern (a) observability queue; mirror Week 2 Step 5/6 |

**Date:** 2026-05-20
**Read-only diagnostic.** No source modifications.

---

## Part 1 — CUPTI → may13-ring bridge — CLEAN-WIRE (revised)

### 1.1 Ring buffer infrastructure on disk

- `include/may13/cipher_10ops.h:23-37` defines `CipherRingEntry` (128 B, cache-line aligned)
- `cipher_10ops.h:75-80` defines `CipherRing` (65536-entry SPMC Disruptor)
- `cipher_10ops.h:83-100` defines static-inline `cipher_ring_write()` (lock-free, header-only)
- `cipher_10ops.h:103-115` defines `Cipher10OpsRuntime g_cipher_10ops` (extern, storage in cipher_10ops_impl.cpp)
- `cipher_10ops.h:117` declares `cipher_10ops_init/teardown/report()` (impl in cipher_10ops_impl.cpp)

### 1.2 Initial concern — `g_cipher_10ops` requires cipher_10ops_impl.cpp port

Symbol-closure analysis on `cipher_10ops_impl.cpp`:

```
undef count: 77
non-system: 61
in current A4 11-file closure: 3
NOT in A4 (would require new ports): 50
no in-tree provider: 8
```

Porting `cipher_10ops_impl.cpp` pulls in 23 unique providers (cipher_edmd, cipher_lnn, cipher_hibernate, cipher_pulse, cipher_shield, cipher_thermostat, cipher_volt, …) — same "port the brain" pattern Week 2 Step 2 surfaced.

### 1.3 Resolution — bypass `g_cipher_10ops`

`cipher_sense.cpp` (the consumer of CipherRingEntry) has only 2 non-system undefs: `__fprintf_chk` and `__stack_chk_fail` — both fortified-libc, **resolved at link time**. **It does NOT reference `g_cipher_10ops`.** The 5 SENSE functions (`init`, `observe`, `get_type`, `current_session`, `report`) are already exported by libcipher_rt.so today (ported in Week 2 Step 2 D1 as inert TU).

Bridge approach: stack-allocate a `CipherRingEntry` in `cipher_cupti.c` (~50 lines below the CLASSIFY block), populate from CUPTI launch params + classify_out, pass to `cipher_sense_observe(&entry)`. **No need to port `cipher_10ops_impl.cpp`. No need for `g_cipher_10ops` storage.**

LOC estimate: ~20 in `cipher_cupti.c`.

### 1.4 Verification

```bash
$ nm -D libcipher_rt.so | grep cipher_sense_
T cipher_sense_current_session
T cipher_sense_get_type
T cipher_sense_init
T cipher_sense_observe
T cipher_sense_report
```

All five present in the current `0b6effdb..4f1a86ab` build. Confirmed callable.

---

## Part 2 — `cipher_sense_init` wiring — CLEAN-WIRE

### 2.1 Function body (`cipher_sense.cpp:154-170`)

```cpp
extern "C" int cipher_sense_init(void) {
    int already = g_sense_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_sense_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_SENSE");
    int on = (env && (strcmp(env,"on")==0 || strcmp(env,"1")==0
                                          || strcmp(env,"ON")==0));
    g_sense_enabled.store(on, std::memory_order_release);
    if (on) fprintf(stderr, "[CIPHER Op13] SENSE enabled — slots=%u idle_ms=%llu\n", ...);
    return on;
}
```

- **Idempotent** via atomic exchange flag
- **Default OFF** (`CIPHER_SENSE` env unset → returns 0; observe path is a single relaxed atomic load + branch when OFF)
- **No CUDA / NVML / kmod touch**
- **No allocation**
- Safe to call from any context

### 2.2 Static-init safety (re-verified)

Week 2 Step 2 v3 pre-flight static-init scan classified `cipher_sense.cpp` as CLEAN (zero `.init_array` entries). Re-confirmed: no constructor functions registered.

### 2.3 Call site

`cipher_inject.c::cipher_v2_init_body()` is the canonical libcipher_rt init body (the same place Step 6 added CLASSIFY substrate init). Add one line:

```c
(void)cipher_sense_init();   /* Op 13 SENSE; default OFF via CIPHER_SENSE env */
```

LOC estimate: 5 (1 actual + 4 surrounding comment).

---

## Part 3 — Per-launch `cipher_sense_observe` — CLEAN-WIRE

### 3.1 Body inspection (`cipher_sense.cpp:171-197`)

```cpp
extern "C" void cipher_sense_observe(const CipherRingEntry* ev) {
    if (!g_sense_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    uint64_t now = ev->timestamp_ns;
    if (now == 0) { clock_gettime(CLOCK_MONOTONIC_RAW, &ts); now = ...; }
    uint64_t prev = g_last_ts_ns.exchange(now, std::memory_order_relaxed);
    int slot = g_current_slot.load(std::memory_order_relaxed);
    /* ... session-boundary detection + slot allocation ... */
}
```

- Fast-exit when SENSE env-OFF (~3 cycles).
- When ON: `clock_gettime`, atomic exchanges, internal session-table maintenance.
- Empirical cost per cipher-may13-evidence/SENSE banner: "Stage 1, ~30 ns" hot-path.

### 3.2 Latency budget impact

At default-OFF (production), cost is ~3 cycles per launch — negligible. At ON, +30 ns per launch atop the existing CLASSIFY route (~100-200 ns) = ~150-250 ns total per launch. At 10K launches/sec, ~2-3 ms/sec overhead. Acceptable.

### 3.3 Data shape construction (in cipher_cupti.c after CLASSIFY block)

```c
{
    CipherRingEntry entry = {0};
    entry.timestamp_ns   = 0;  /* let cipher_sense_observe fill via clock_gettime */
    entry.func_ptr_hash  = (uint64_t)(uintptr_t)fn;  /* CUPTI fn pointer */
    entry.kernel_class   = out.op_class;
    entry.grid_x = gX; entry.grid_y = gY; entry.grid_z = gZ;
    entry.block_x = bX; entry.block_y = bY; entry.block_z = bZ;
    entry.confidence     = (float)out.confidence / 100.0f;
    entry.decision       = (uint8_t)(r == CIPHER_RT_CLASSIFY_HANDLED);
    /* sequence + params_hash + output_hash zeroed; cipher_sense uses
     * func_ptr_hash + timestamp for session boundary detection. */
    cipher_sense_observe(&entry);
}
```

LOC estimate: ~10 in `cipher_cupti.c` (after the existing CLASSIFY block at Step 6 line ~189).

---

## Part 4 — SENSE proposal accessor — **DESIGN-MISMATCH**

### 4.1 Public surface (`include/may13/cipher_sense.h`)

```c
typedef enum {
    CIPHER_SESSION_UNKNOWN           = 0,
    CIPHER_SESSION_HUMAN_INTERACTIVE = 1,
    CIPHER_SESSION_AGENT_AUTONOMOUS  = 2,
    CIPHER_SESSION_BATCH_BACKGROUND  = 3,
} CipherSessionType;

int                cipher_sense_init(void);
void               cipher_sense_observe(const CipherRingEntry* ev);
CipherSessionType  cipher_sense_get_type(uint64_t fingerprint);
uint64_t           cipher_sense_current_session(void);
void               cipher_sense_report(void);
unsigned           cipher_sense_session_count(void);
```

**SENSE produces session classifications, not migration proposals.** The output is "this session is HUMAN" or "this session is AGENT". There is no `cipher_sense_propose_migration()`, no `cipher_sense_get_proposal()`, no proposal-related API at all.

### 4.2 Wave 5's "SENSE → DSM PROPOSE" prescription

Wave 5 §5.5 W3 L738-739 says:
> "SENSE phase transitions trigger DSM PROPOSE for tool-idle detection (Wave 3 `cipher_cp54_sched.c::COMPACT_MIGRATE` nr 20)."

This implies a wrapper that:

1. Polls `cipher_sense_current_session()` to get the current session's fingerprint
2. Calls `cipher_sense_get_type(fingerprint)` to get the type
3. Tracks per-tenant session-type state (HUMAN ↔ AGENT ↔ BATCH transitions)
4. On a transition that indicates "tool-idle" (e.g., AGENT_AUTONOMOUS staying agent-y for N seconds), emits a migration proposal
5. Pushes proposal via the new `CIPHER_DSM_PROPOSE` ioctl

**This wrapper does NOT exist in cipher-may13-evidence.** Wave 5's prescription assumes a behavior that's not in the may13 code.

### 4.3 Scope of building the wrapper

~200-300 LOC of new C++ in cipher_rt_phase4 native:

- Per-tenant fingerprint table (map tenant_id → last session fingerprint)
- Per-tenant session-type cache + last-transition timestamp
- Transition-detection state machine (debounce; e.g., 5-second hysteresis)
- "Tool-idle" heuristic (when AGENT_AUTONOMOUS persists for N kernel launches with no human-class kernels mixed in)
- Proposal-emit call site (likely in the 256-launch flush piggyback alongside `cipher_rt_classify_push_to_kmod`)

Plus the userspace-side ioctl wrapper (~30 LOC, mirrors `cipher_rt_classify_push_to_kmod`).

### 4.4 Risk surfaces

- **Heuristic correctness**: what threshold defines "tool-idle worth migrating"? Tuning required.
- **Hysteresis**: prevent thrashing (existing `cipher_cp54_mig_ratelimit_ms = 10000` provides 10s rate limit, but the wrapper needs its own debounce).
- **Per-tenant state**: requires tenant identity at observation time. cipher_cupti.c sees per-thread launches; tenant identity comes from `cipher_rt_tenant.cpp`. Wiring is doable but adds dependency.
- **Production-readiness**: SENSE's classifications are research-grade per may13 README ("Classifies the active session into HUMAN_INTERACTIVE / AGENT_AUTONOMOUS / BATCH_BACKGROUND / UNKNOWN using only ring-buffer timing and shape sequences"). Migration decisions based on this carry the same research-grade caveat.

### 4.5 Verdict

DESIGN-MISMATCH at the spec level: Wave 5 named a behavior that requires NEW code. Not a code-on-disk gap but a behavior-on-disk gap.

---

## Part 5 — `CIPHER_DSM_PROPOSE` ioctl ABI — SHIM-REQUIRED

### 5.1 Reserved nr availability

Next-free nr = 26 (per Week 2 Step 6 pre-flight analysis; nrs 21, 22, 25 also free but 25 was taken by `CIPHER_PUSH_CLASSIFY_STATS`, 21+22 are inactive). Cb.2 discipline preserved (nrs 2/3/4 still -ENOSYS).

### 5.2 Existing CP 5.4 migration ioctls (nrs 16-20)

```
nr 16: SUBSCRIBE_MIGRATE — register for migration notifications
nr 17: POLL_MIGRATE       — poll for pending migrate (cipher_cp54_migrate_poll)
nr 18: START_MIGRATE      — begin migration to target_mask
nr 19: ACK_MIGRATE        — confirm migration complete
nr 20: COMPACT_MIGRATE    — force compaction-evaluation pass (operator)
```

None of these have PROPOSE semantics. PROPOSE is a NEW ioctl.

### 5.3 Proposed payload struct

```c
/* Week 3 Step 4 (Option II) — SENSE → DSM PROPOSE bridge. Userspace
 * pushes session-transition signals to kmod; kmod queues them for
 * operator-side adjudication. Pattern (a) observability queue. */
struct cipher_dsm_propose {
    __u32 tenant_id;        /* tenant whose session transitioned */
    __u32 session_band;     /* CipherSessionType: 0=UNK, 1=HUMAN, 2=AGENT, 3=BATCH */
    __u32 reason;           /* SENSE-internal reason code */
    __u32 timestamp_secs;   /* CLOCK_MONOTONIC at transition */
    __u64 fingerprint;      /* SENSE session fingerprint */
    __u64 reserved[4];      /* Cb.2 future expansion */
};                          /* 56 bytes; 8-byte aligned */

#define CIPHER_DSM_PROPOSE \
    _IOW(CIPHER_IOCTL_MAGIC, 26, struct cipher_dsm_propose)
```

LOC estimate: ~30 in `cipher_kmod/cipher_ioctl.h`.

### 5.4 Verdict

SHIM-REQUIRED. No structural blocker. Mechanical mirror of Week 2 Step 6's `CIPHER_PUSH_CLASSIFY_STATS` pattern.

---

## Part 6 — Kmod handler + proposals queue (Pattern a) — SHIM-REQUIRED

### 6.1 Existing CP 5.4 migration state machine

- 5 ioctl handlers exist (`cipher_cp54_ioctl_poll/start/ack/compact_migrate` at `cipher_cp54_sched.c:726/754/773/797`)
- No proposals queue (grep clean)

### 6.2 Pattern (a) — observability queue (no auto-action)

Recommended for v1:

- New `cipher_dsm_proposals_queue` struct in `cipher_internal.h`: bounded queue (16 slots? 64?) of `cipher_dsm_propose` entries with atomic head/tail.
- New `cipher_dsm_propose_ioctl()` handler in `cipher_proc.c` or new `cipher_dsm.c`: push to queue, drop oldest on overflow (bounded observability semantic).
- New `/proc/cipher/dsm_proposals` proc node (mirror Week 2 Step 5 pattern): emit recent proposals.
- Operator (or future automation) reads `/proc/cipher/dsm_proposals` and acts (or doesn't).

### 6.3 Pattern (b) — auto-action

NOT recommended for v1. Would feed PROPOSE directly into `cipher_cp54_ioctl_start_migrate` without operator review. SENSE is research-grade (per §4.4); auto-acting would be premature.

### 6.4 LOC estimate (Pattern a)

| component | LOC |
| --- | ---:|
| struct cipher_dsm_proposals_queue | ~15 |
| cipher_dsm_propose_ioctl handler | ~25 |
| /proc/cipher/dsm_proposals show callback + proc_create | ~30 |
| dispatch case in cipher_dev.c | ~5 |
| cipher_internal.h declarations | ~10 |
| **total** | **~85** |

### 6.5 Verdict

SHIM-REQUIRED. Pattern (a) mirrors Week 2 Step 5 `classify_stats` + Step 6 ioctl handler patterns exactly. No structural blocker.

---

## Part 7 — Cumulative scope + time estimate

| sub-deliverable | LOC | est time |
| ---:| ---:| ---:|
| 1 — CUPTI → ring bridge | ~20 | 30 min |
| 2 — `cipher_sense_init` wiring | ~5 | 15 min |
| 3 — `cipher_sense_observe` wiring | ~10 | 30 min |
| **4 — SENSE → proposal wrapper (NEW)** | **~250** | **2-3h** |
| 5 — `CIPHER_DSM_PROPOSE` ioctl ABI | ~30 | 30 min |
| 6 — Kmod handler + proposals queue | ~85 | 1-1.5h |
| LIVE flip (1-line default) | ~5 | 10 min |
| Build + nm + smoke per tree | — | 30 min |
| TinyLlama SC6 + Mistral-7B SC6 | — | 15 min |
| Commits + tags (2 trees) | — | 20 min |
| Buffer / iteration | — | 30 min |
| **total** | **~405** | **~6.5-9h** |

vs scope-lock budget of 4-6h: ~50% over (driven by Sub-4 wrapper work).

### Risk surfaces

| sub-deliverable | risk |
| ---:| --- |
| 1, 2, 3 | LOW (stack-construct, idempotent, env-OFF default) |
| 4 | **MEDIUM-HIGH** (heuristic correctness, hysteresis tuning, per-tenant state plumbing) |
| 5 | LOW (mechanical ioctl) |
| 6 | LOW (queue + proc emit) |
| LIVE flip | MEDIUM (Mistral-7B SC6 is the binding gate; existing Step 3 substrate ought to hold bit-identity but it's the first time at LIVE=1) |

---

## Part 8 — Recommendation: SCOPE-COMPRESSED

**Option II as originally scoped (all 6 sub-deliverables): feasible but undersized by ~50%.** Three responses available:

### II-a — Accept the overrun; ship all 6 + LIVE flip

- ~6.5-9h actual vs 4-6h budget
- Wave 5 W3 fully shipped
- Risk: Sub-4 wrapper is NEW code with no testing precedent; tuning may need iteration
- Recommendation: only viable if Sub-4 wrapper is treated as research/scaffolding (queue proposals but don't act on them); production tuning deferred

### II-b — Drop Sub-4 (proposal wrapper); ship Subs 1-3 + LIVE flip + Subs 5-6 as deferred

- ~3-4h actual; closer to original budget
- Wave 5 W3 partially shipped: LIVE flip + SENSE observation pipeline + ioctl infrastructure all live, but no proposals are emitted (the queue is empty)
- DSM PROPOSE ioctl + kmod handler ship as stubs (accept payload, never receive any) → Wave 5 nominally complete from infrastructure perspective; behavioral observation deferred to Week 4 (when LOOP/PIPELINE port lands; those may be the actual proposal producers)
- Lowest risk; cleanest deliverable boundary

### II-c — Drop Subs 4-6 entirely; ship Subs 1-3 + LIVE flip only

- ~2-3h actual
- Wave 5 W3 only partially shipped (no DSM PROPOSE infrastructure at all)
- Same as Option I in terms of LIVE flip; adds SENSE observation pipeline for diagnostic value
- DSM PROPOSE deferred to Week 4 entirely
- Smallest delta; easiest to roll back if Mistral SC6 fails

### Note on Mistral-7B SC6 gate

Regardless of II-a/b/c, the LIVE flip is the load-bearing change. Mistral-7B SC6 must PASS bit-identical at LIVE=1. Sub-deliverables 1-3 add observation but don't change kernel routing; Subs 5-6 add kmod-side ioctl but don't change kernel routing. Only the LIVE flip changes routing. If SC6 fails post-flip, the issue is in the LIVE-driven routing path (Marlin shape gate, classify→dispatch_lookup, etc.), not in the SENSE/DSM additions.

---

## Telemetry on disk

- `/home/ubuntu/cipher_rt_phase4/include/may13/cipher_sense.h` — SENSE public API (5 functions; no proposal accessor)
- `/home/ubuntu/cipher_rt_phase4/src/may13/cipher_sense.cpp` — SENSE impl (init + observe + get_type + report)
- `/home/ubuntu/cipher_rt_phase4/include/may13/cipher_10ops.h` — `CipherRingEntry` type + `cipher_ring_write` inline (g_cipher_10ops storage NOT linked)
- `/home/ubuntu/cipher_rt_phase4/cipher_cupti.c:134-191` — CLASSIFY landing site; Sub-1 lands ~50 lines below
- `/home/ubuntu/cipher_rt_phase4/cipher_inject.c::cipher_v2_init_body()` — Sub-2 init call site
- `/home/ubuntu/cipher_kmod/cipher_ioctl.h` — Sub-5 add site (nr 26)
- `/home/ubuntu/cipher_kmod/cipher_proc.c` — Sub-6 handler + proc node mirror site (Week 2 Step 5/6 patterns)
- `/tmp/symbol_closure.py` (+ a4/v3) — cross-TU symbol audit tooling used here

---

## Discipline notes

- Read-only diagnostic; no source-tree changes.
- Sub-1's INFRASTRUCTURE-MISSING first-read flipped to CLEAN-WIRE after verifying `cipher_sense.cpp`'s undef closure is self-contained.
- Sub-4 DESIGN-MISMATCH is a spec/code gap: Wave 5 named a behavior that doesn't exist; building it is feasible but expands scope.
- No recommendation between II-a / II-b / II-c per discipline.
- Failure-mode-of-record carried forward: Wave 5 prescriptions that say "X → Y" should have explicit pre-flight checks on both the producer's actionable-output exposure AND the consumer's ioctl/symbol existence. Sub-4 is the second example this week (Sites 1+2 cuBLAS/SDPA was the first).

---

## Awaiting

User adjudication on II-a / II-b / II-c. After adjudication:

- **II-a**: I draft a Step 4-full brief (~6.5-9h, all 6 sub-deliverables + LIVE flip + Mistral SC6 gate).
- **II-b**: I draft a Step 4-medium brief (~3-4h, Subs 1-3 + 5-6 + LIVE flip; Sub-4 wrapper deferred to Week 4 alongside LOOP/PIPELINE port).
- **II-c**: I draft a Step 4-minimal brief (~2-3h, Subs 1-3 + LIVE flip; Sub 5-6 ioctl + Sub-4 wrapper deferred to Week 4).

The Mistral-7B SC6 bit-identical gate at LIVE=1 is binding under all three sub-options.
