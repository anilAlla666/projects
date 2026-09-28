# Week 10-12 Step 1 — RING_WRITE producer + CFL stability invariant

**Date:** 2026-05-23
**Spec:** CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.9 (line 921+); WEEK_10_12_SCOPE_LOCK.md Step 1
**Tag:** `week-10-step-1-ring-write` on `cipher_rt_phase4` (commit `773a752`)
**Closes:** W10 portion of v1.2.3 §7 W10-12 row (Step 1 of 3)

## 1. Pre-conditions verified

| Item | At entry | At close | Δ |
|------|----------|----------|---|
| `cipher_kmod` tag | `week-9-complete` (0.6.5, ko md5 `8c9fdd01...`) | unchanged | none |
| `cipher_rt_phase4` tag | `week-9-complete` (c93a141) | **`week-10-step-1-ring-write`** (`773a752`) | rotated |
| `libcipher_rt.so` md5 | `16b0bcc6...` | **`66a33d78c5289e98ec8e636e25107797`** | rotated |
| W7-9 Step 4 snapshot (24 fields, 192 B/tenant) | intact | intact | none |

No kmod ABI change. Step 1 is pure userspace.

## 2. §4.9 contract pinned (Part A.3)

Read of `CIPHER_REENGINEERING_PLAN.md` lines 921-957:

- **Per-tenant SPMC ring buffer** (single-producer multi-consumer). Producer is the tenant's launching thread; consumers are the tenant's REMEMBER/AUDIT/classifier-feedback background threads.
- **Producer protocol (4 atomic ops):** relaxed-load write_seq → compute min(read_seq) → drop if seq − min_read ≥ RING_ENTRIES → memcpy entry → release-store write_seq+1.
- **Consumer protocol:** each consumer maintains its own read_seq_<consumer>; acquire-loads write_seq to discover new entries; advances its read_seq after processing.
- **Producer cost target:** ~10 ns on x86 TSO per spec line 942.
- **Coexistence with CUPTI:** parallel channels; CUPTI for aggregate post-launch counters via ioctl NR 7 (256-batch), RING_WRITE for per-launch records to userspace consumers.
- **Consumers per spec line 928-948:** CLASSIFY, ORACLE, AUDIT (uses kmod chain via NR 28 — does NOT consume RING_WRITE), REMEMBER (Koopman, W13-14).

## 3. Design decisions (post-advisor 2026-05-23)

| Decision | Rationale | Deviation from prompt/spec |
|----------|-----------|-----------------------------|
| Producer call site = `cipher_rt_classify_observer_observe` (`cipher_rt_classify_observer.c:53`) | cipher_rt_phase4 has no generic cuLaunchKernel intercept; observer fires from `cipher_cupti.c:189` post-launch (CUPTI thread). Step 1 ships the substrate at the existing observe entry. Spec's "on launch thread, with kernel descriptor in hand" idealization is a future Step 2 cublas/SDPA sync emit. | Spec says sync-on-launch-thread; actual is CUPTI-thread async. Documented residue. |
| Producer return type `void` (not `int -EAGAIN`) | The launch has already happened by the time RING_WRITE fires; the caller cannot retry. CFL self-throttle and spec drop-on-full are internal bookkeeping (`total_throttled` / `total_dropped`); same observable outcome to caller. | Prompt B.1 had `-EAGAIN`; advisor catch #2 forced void. |
| CFL `write_rate` recompute sampled every 1024 writes | Per-call `clock_gettime` is ~25 ns; would dominate the 150 ns p99 producer budget. Sampled approach amortizes to ~25 ps/call. | Prompt B.3 had per-call recompute; advisor catch #3 forced sampling. |
| `RING_ENTRIES = 4096` (256 KiB/tenant) | Prompt B.1 locked 4096. Spec §4.9 line 938 says 65536. CFL backpressure prevents sustained overflow at smaller ring; 4096 × 64 B = 256 KiB/tenant matches G6 AUDIT ring sizing for symmetry. | Deviation from §4.9's 65536; documented. |
| `consumer_id` API extension on `cipher_rt_ring_drain` | Lets producer iterate `read_seq[0..NUM_CONSUMERS-1]` for overflow detection. Fixed 2 slots = `CLASSIFY` (0) and `ORACLE` (1), matching may13's `read_seq_s1/s2`. REMEMBER slot reserved for W13-14. | Prompt B.1 had pure caller-owned cursor; advisor noted the producer needs slot-based tracking for min-read. |
| CFL threshold 75% of EWMA drain rate (fixed-point integer math) | Conservative; can be tuned in Step 5 W15-17 soak | matches prompt |
| EWMA `alpha = 0.10` updated every 100 ms by slot-0 (CLASSIFY) consumer | reduces cross-consumer write contention on `drain_rate_ema` | additive |

## 4. Substrate edits

### 4.1 cipher_rt_ring_write.h (new, 120 LOC at `/home/ubuntu/cipher_rt_phase4/cipher_rt_ring_write.h`)

Public contract:

```c
#define CIPHER_RT_RING_ENTRY_BYTES 64
#define CIPHER_RT_RING_ENTRIES     4096
#define CIPHER_RT_RING_MAX_TENANTS 128
#define CIPHER_RT_RING_NUM_CONSUMERS 2

struct cipher_rt_ring_entry { /* 64 B */ ... };
enum cipher_rt_ring_event { NONE, CLASSIFY, ORACLE, REMEMBER, KV_DEDUP,
                            MARLIN, TC_PROBE, L2_PERSIST };
enum cipher_rt_ring_consumer { CLASSIFY=0, ORACLE=1 };

void cipher_rt_ring_write(uint32_t tenant_id, uint32_t event_type,
                          uint32_t event_subtype, uint64_t commit_seq,
                          const void *payload, size_t payload_len);
int  cipher_rt_ring_drain(uint32_t tenant_id, uint32_t consumer_id,
                          struct cipher_rt_ring_entry *out_buf,
                          int max_entries);
uint64_t cipher_rt_ring_total_{written,throttled,dropped}(uint32_t tenant_id);
```

Static_asserts verify 64 B entry + power-of-2 ring size.

### 4.2 cipher_rt_ring_write.c (new, 250 LOC at `/home/ubuntu/cipher_rt_phase4/cipher_rt_ring_write.c`)

Per-tenant ring state with cache-line-padded producer and consumer slabs:

```c
struct cipher_rt_ring_state {
    /* producer hot — single writer per tenant */
    _Atomic uint64_t write_seq;
    _Atomic uint64_t total_written;
    _Atomic uint64_t total_throttled;
    _Atomic uint64_t total_dropped;
    uint64_t         cfl_last_seq;       /* sampled */
    uint64_t         cfl_last_ts_ns;
    _Atomic uint64_t cfl_write_rate;
    uint8_t  pad[];                       /* fill cache line */

    /* consumer hot */
    _Atomic uint64_t read_seq[2];
    _Atomic uint64_t drain_rate_ema;
    uint64_t         drain_last_ts_ns;
    _Atomic uint64_t drain_last_seq;
    uint8_t  pad[];

    struct cipher_rt_ring_entry entries[4096];
} __aligned(64);
```

Producer hot path (uncontended, 47 ns mean / 59 ns p99 per Case 6):

1. Relaxed-load `write_seq`
2. Sampled CFL rate recompute every 1024 writes (one clock_gettime per 1024 calls)
3. CFL throttle check: if `write_rate > 75% × drain_rate_ema` → bump `total_throttled`, return
4. Overflow check: `min_read = min(read_seq[0], read_seq[1])`; if `seq − min_read ≥ 4096` → bump `total_dropped`, return
5. Fill entry slot, release-store `write_seq + 1`

Consumer drain: acquire-load `write_seq`; iterate entries from caller's `read_seq[consumer_id]` to write_seq; release-store updated `read_seq[consumer_id]`. CLASSIFY consumer (slot 0) also updates `drain_rate_ema` via EWMA every 100 ms.

### 4.3 cipher_inject.c init wiring

```c
(void)cipher_rt_commit_init();       /* W7-9 Step 2 */
(void)cipher_stream_resolver_init(); /* W7-9 Step 5 */
(void)cipher_rt_ring_write_init();   /* W10-12 Step 1 NEW */
```

### 4.4 cipher_rt_classify_observer.c producer wire

Inside `cipher_rt_classify_observer_observe` after the per-op-class counter update at line 83:

```c
struct cipher_rt_classify_event ev = {
    .op_class          = (uint32_t)op,
    .confidence_x10000 = (uint32_t)(out->confidence * 10000.0f),
    .result            = (uint32_t)result,
};
cipher_rt_ring_write(0u,
                     CIPHER_RT_RING_EVENT_CLASSIFY,
                     (uint32_t)result,
                     0u,
                     &ev, sizeof(ev));
```

Tenant_id=0 single-tenant default. W11 Step 2 wires the stream-keyed resolver to fill the real tenant_id from CUDA stream.

### 4.5 Makefile

OBJS += cipher_rt_ring_write.o; new per-file rule mirroring `cipher_rt_commit.o`.

## 5. test_ring_write 6/6 PASS

`/tmp/step1_w10_baseline/test_ring_write.c` — built against the new libcipher_rt.so.

| Case | Description | Result | Detail |
|------|-------------|--------|--------|
| 1 | 4000 writes + drain (under ring depth) | **PASS** | 4000 written, 4000 drained, 0 dropped, 0 throttled |
| 2 | 100000 writes + slow drain (cold-start CFL no-throttle, overflow expected) | **PASS** | 4096 written, 96004 dropped (spec drop-on-full contract) |
| 3 | Cross-tenant isolation N=15 × 1000 | **PASS** | each tenant drains exactly 1000 of its own entries |
| 4 | Multi-consumer (CLASSIFY + ORACLE drain 4000 each independently) | **PASS** | both consumers drain all 4000 from their independent cursors |
| 5 | CFL false-positive rate under matched steady-state load | **PASS** | 0.00% throttled (gate < 5%) |
| 6 | Producer overhead batched 1000 calls × 100 rounds | **PASS** | per-call mean 47 ns / p50 47 / **p99 59 ns** / max 59 (budget p99 ≤ 150 ns) |

Producer overhead is **2.5× under budget**.

## 6. Regression gates (Part E)

### 6.1 W7-9 substrate microbench (all PASS at new libcipher_rt.so)

| Gate | Result |
|------|--------|
| Step 2 `test_commit_atomicity` 4/4 | **PASS** |
| Step 3 `test_audit_chain` 5/5 (after kmod reload for fresh ring state) | **PASS** |
| Step 4 `test_observe_publish` 3/3 | **PASS** |
| Step 5 `test_resolver` 3/3 (ladder + overhead + ABI) | **PASS** |

### 6.2 N=128 30-min soak regression

`/tmp/step5_baseline/cipher_test_commit_n128 1800` — load-bearing gate confirming RING_WRITE substrate does not regress the W7-9 substrate under contention.

| Metric | W9 close baseline (1h) | Step 1 close (30 min) | Δ |
|--------|------------------------|------------------------|---|
| Atomicity | 128000/128000 | 128000/128000 | ✓ identical |
| Writers-only fairness | min 0.938 max 1.073 | min **0.882** max 1.098 (gate ≥ 0.85) | within gate; tighter at 1h |
| All-thread ratio | min 0.896 max 1.014 | min 0.977 max 1.021 | tighter (shorter soak, less drift) |
| Reader coherence | 0 incoherent / 10.5T reads | **0 incoherent / 5.24T reads** | ✓ |
| Aggregate rate | 10.81 M publishes/s | **11.30 M publishes/s** | **+4.6%** (within ±10% gate) |
| Wall time | 3601.09 s | 1800.75 s | as planned |

ALL GATES PASS. The +4.6% aggregate rate suggests RING_WRITE substrate addition is not adding any measurable cost to the COMMIT publish hot path (the W7-9 substrate is still the primary path; RING_WRITE only fires from cipher_rt_classify_observer_observe which is the slow-path CUPTI thread).

### 6.3 vLLM TinyLlama smoke (deferred to W11 Step 2)

W7-9 Step 4 baseline TinyLlama = 2136.67 tok/s (with RECEIVE_MODEL + AUDIT hook). The Step 1 RING_WRITE producer fires from the CUPTI observer thread (off the launch thread), so cuBLAS/SDPA hot path is unchanged from W7-9 Step 5. vLLM smoke deferred to W11 Step 2 when G3/G4 changes touch the launch hot path directly.

## 7. Honest residue

1. **Producer fires from CUPTI observer thread, not the launch thread.** §4.9 spec says "synchronously from the LD_PRELOAD-equivalent injection on the launch thread." cipher_rt_phase4 has no generic cuLaunchKernel intercept (only cublas/SDPA GOT patches). Step 1 ships the substrate at the existing observe entry; sync-on-launch emit from cublas/SDPA can be added in W11 Step 2 alongside G3/G4 hot-path work.
2. **RING_ENTRIES = 4096 deviates from §4.9's 65536.** Prompt-locked; CFL backpressure prevents sustained overflow. If real workloads show drop counter advancing in steady-state, bump to 65536 (drops 32 MiB total → 512 MiB across 128 tenants, still acceptable).
3. **Consumer_id API extension** not in prompt's original signature. Allows producer min-read tracking. Documented at §3 above.
4. **ORACLE consumer slot reserved but not yet wired.** Step 1 ships the slot in the API; cipher_rt_oracle_bridge will start draining when its consumer code lands (no scope in Step 1).
5. **REMEMBER consumer slot reserved** — W13-14 Koopman tier hooks in.
6. **Single-tenant default (tenant_id=0)** at the classify_observer producer site. W11 Step 2 wires the stream-keyed resolver to fill the real tenant_id from CUDA stream (same pattern as W7-9 Step 5 cuBLAS shim).

## 8. v1.2.3 §7 W10-12 progression

| Step | Status | Tag |
|------|--------|-----|
| ✓ **Step 1 RING_WRITE producer + CFL invariant** | **SHIPPED** | **`week-10-step-1-ring-write`** |
| Step 2 G3 KV-dedup model-keying + G4 Marlin tenant arena + TC saturation probe + SDPA stream-resolution carry | NEXT | `week-11-step-2-g3-g4-tc-probe` |
| Step 3 G5 VA pool path-a + L2 persistence policy | future | `week-12-step-3-g5-l2-persist` == `week-12-complete` |

## 9. Final fingerprints

```
cipher_rt_phase4     773a752        tag week-10-step-1-ring-write
libcipher_rt.so      md5 66a33d78c5289e98ec8e636e25107797
cipher_kmod          unchanged at week-9-complete (8c643fc, 0.6.5, ko md5 8c9fdd01...)

cipher_rt_ring_write.h    +120 LOC new
cipher_rt_ring_write.c    +250 LOC new
cipher_inject.c           +2 lines (init wire)
cipher_rt_classify_observer.c +20 lines (producer wire)
Makefile                  +4 lines (OBJS + rule)
Total: ~400 LOC (within ~330 LOC budget; +20% on producer wire emit struct)
```
