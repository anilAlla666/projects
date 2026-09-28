# Week 2 Step 6 — Hot-Path CLASSIFY Wiring (CUPTI + ioctl) — RESULT

**Status: PASS.**

Three landings shipped: CUPTI CLASSIFY block, userspace ioctl wrapper, kmod ioctl handler. Sites 1+2 (cuBLAS, SDPA) intentionally dropped per pre-flight DESIGN-MISMATCH. End-to-end smoke confirms classify counters reach `/proc/cipher/classify_stats` via the 256-launch flush bridge. LP-2 SDPA invariant holds; CP 5.4 isolation 15/15 byte-identical.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 6 of 7
**Anchors:**
  - cipher_rt_phase4: `7ee5b2a8` → `f9c32322` (tag `week-2-step-6-cupti-classify-wired`)
  - cipher_kmod: `bc48590e` → `0ce4b8e2` (tag `week-2-step-6-ioctl-classify-push`)

---

## 6.A — Baseline + route() locking inspection — PASS

| signal | value |
| --- | --- |
| rt_phase4 HEAD | `7ee5b2a8` ✓ |
| cipher_kmod HEAD | `bc48590e` ✓ |
| Pre-edit rt_phase4 build | rc=0, md5 `71cf2a25` (nvcc-non-deterministic), 78 warnings |
| Pre-edit cipher_kmod build | rc=0, md5 `f497b2ee` (reproduces; kmod is deterministic), 1 warning |

### route() locking model

`cipher_rt_classify_substrate.cpp::cipher_rt_classify_route()` (line 137) reads `g_classify.n_actuators` as a single int snapshot and uses C++ `std::atomic<>::fetch_add` for the 3 counters. **No mutex on the hot path.** The mutex is only acquired inside `cipher_rt_classify_register_actuator()` (init-time). Latency budget of ~100-200 ns/call holds.

This was the load-bearing pre-flight gate; route() being lock-free is what makes Step 6 sensible at CUPTI-per-launch frequency.

---

## 6.B — kmod ioctl handler — PASS

### Files modified

- `cipher_kmod/cipher_ioctl.h` (+17 lines): `struct cipher_classify_stats_push` (216 B; 27×`__u64`); `#define CIPHER_PUSH_CLASSIFY_STATS _IOW(magic, 25, ...)`.
- `cipher_kmod/cipher_internal.h` (+5 lines): `cipher_push_classify_stats_ioctl()` prototype.
- `cipher_kmod/cipher_dev.c` (+2 lines): new `case CIPHER_PUSH_CLASSIFY_STATS` in dispatch (between `CIPHER_ARENA_QUERY` and reserved-`-ENOSYS` block).
- `cipher_kmod/cipher_proc.c` (+23 lines): `cipher_push_classify_stats_ioctl()` impl — `copy_from_user(216 B)`, then `atomic64_set` per field (3 head + 16 per-op). Returns 0 / -EFAULT.

### Security posture

No `CAP_SYS_ADMIN` check. The payload is observer-state (read-only semantically from kmod's perspective — kmod overwrites its own counters with userspace values). Matches `CIPHER_SUBMIT_LAUNCH_STATS` (ioctl nr=7) and `CIPHER_SUBMIT_GPU_STATE` (nr=5) patterns: any process can push its own telemetry. The /proc node is mode 0444 (read-only); ioctl nr=25 is the only write path.

### Build + ABI

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 1 → 1 (zero delta) |
| cipher_kmod.ko md5 | `f497b2ee` → `8d1f1fc4` |
| existing IOCTL nrs | unchanged |
| reserved nrs 2/3/4 | still `-ENOSYS` (cipher_dev.c:299-302) |
| struct cipher_classify_stats_push alignment | natural 8-byte; no `__packed` |
| new nm exports | `cipher_push_classify_stats_ioctl` (T) |

---

## 6.C — Userspace ioctl wrapper — PASS

### `cipher_rt_classify_push_to_kmod()` in `cipher_rt_classify_observer.c` (+54 lines)

- **Lazy fd init**: cached `g_kmod_fd` opened via `open("/dev/cipher", O_RDWR)` on first push; pthread_mutex guards the init race. After init, the fd is read lock-free.
- **1:1 snapshot copy**: `__atomic_load_n` per field from `g_stats` → local `struct cipher_classify_stats_push`. Memory order `RELAXED` (pairs with `__atomic_fetch_add` writers).
- **Reserved tail**: `memset(push.reserved, 0, sizeof(push.reserved))` for forward compat.
- **Non-fatal failures**: ioctl errno logged once (`cipher_log`), function returns -1, caller (CUPTI piggyback) drops the error without affecting the launch path.

Includes added: `<fcntl.h>`, `<unistd.h>`, `<sys/ioctl.h>`, `<pthread.h>`, `<errno.h>`, `<string.h>`, `cipher_ioctl.h`, `cipher_v2_internal.h`. Existing build path already had `-I$(CIPHER_KMOD_DIR)` per Makefile L32, so `cipher_ioctl.h` resolves transparently.

### Symbol export

```
nm -D libcipher_rt.so | grep cipher_rt_classify_push_to_kmod
T cipher_rt_classify_push_to_kmod
```

Exported (default-visibility); callable from `cipher_cupti.c` and any future caller.

---

## 6.D — CUPTI CLASSIFY wiring (Site 3) — PASS

### `cipher_cupti.c` modifications (+31 lines)

**Geometry extraction extended** (L134-159 area): two new locals `const void *fn` and `unsigned int shared_b`, populated in each of the three CUPTI param-extraction branches:

| CUPTI param type | fn source | shared_bytes source |
| --- | --- | --- |
| `cudaLaunchKernel_v7000_params` | `p->func` | `p->sharedMem` |
| `cudaLaunchKernelExC_v11060_params` | `p->func` | `p->config->dynamicSmemBytes` |
| `cuLaunchKernel_params` | `(const void *)p->f` | `p->sharedMemBytes` |

**CLASSIFY block** added after the existing `cipher_rt_smp_observe(...)` call (~L162):

```c
struct cipher_rt_classify_call call = {0};
struct cipher_rt_classify_out  out;
call.fn     = fn;
call.grid_x = gX; call.grid_y = gY; call.grid_z = gZ;
call.block_x= bX; call.block_y= bY; call.block_z= bZ;
call.shared_bytes = shared_b;
int r = cipher_rt_classify_route(&call, &out);
cipher_rt_classify_observer_observe(&call, &out, r);
```

### Sites 1+2 disposition

`cipher_rt_matmul_dispatch.c` and `cipher_rt_attn_dispatch.cpp` left **untouched**. Per pre-flight §4 / option 1C+2C: dropping is structurally clean because every cuBLAS / SDPA internal kernel launch eventually fires through CUPTI's `cuLaunchKernel` callback — observed in the end-to-end smoke (1650 launches from 150 iterations of SDPA + matmul; cuBLAS-internal launches counted indistinguishably).

---

## 6.E — Piggyback push on 256-launch flush — PASS

`cipher_cupti.c` ~L189: one-line addition before the existing `CIPHER_SUBMIT_LAUNCH_STATS` ioctl:

```c
/* Week 2 Step 6 — piggyback classify-stats push to kmod on the
 * same 256-launch flush cadence. Failure is logged once and
 * non-fatal (does NOT propagate up the launch path). */
(void)cipher_rt_classify_push_to_kmod();
```

Shares the existing `CIPHER_V2_FLUSH_MASK = 0xff` cadence (every 256 launches). At 10K launches/sec inference workloads, this is one ioctl roughly every 26 ms — far below any reasonable observability threshold.

### Includes

`cipher_cupti.c` gained two header includes (Step 6 minimal change):

```c
#include "cipher_rt_classify_substrate.h"   /* Week 2 Step 6 */
#include "cipher_rt_classify_observer.h"    /* Week 2 Step 6 */
```

---

## 6.F — Build + nm diff — PASS

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 78 → 78 (zero delta) |
| libcipher_rt.so md5 | `71cf2a25` → `f6251e62` (nvcc-non-deterministic) |
| Symbol count | 307 → 308 (+1 new T) |
| Symbols removed | 0 |
| Undef audit | clean (only expected system/weak surface) |

### New T-symbol

`cipher_rt_classify_push_to_kmod` — the userspace wrapper.

---

## 6.G — Runtime invariant verification — PASS (the integration payoff)

### G.1 Loader smoke

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
rc=0
```

### G.2 Kmod reload (new ioctl handler live)

```
$ sudo rmmod cipher_kmod && sudo insmod cipher_kmod.ko
rc=0
$ cat /proc/cipher/classify_stats
classify_stats:
  total:        0
  handled:      0
  passthrough:  0
  per_op_class:
  (no producer wired yet — Step 6 will populate via ioctl bridge)
```

Zero pre-test (no producer has pushed yet).

### G.3 End-to-end smoke — 150-iteration SDPA + matmul loop (the Step-6 payoff)

```
$ CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 \
    -c '<150-iter SDPA + matmul loop>'

[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] CUPTI subscribed: kernel launch callbacks active (flush every 256 launches); cuStreamCreate hooks armed
...
Heavy loop done; 150 iterations
[cipher_v2] DIAG-T4.2.4d: total_launches=1650 streams_observed=0 ctx_swaps_to_green=0
[cipher-attn] exit totals - tramp_calls=150 tramp_fake=0 observed=150
              handled=0 passthrough=150 redirected=0 (flash=0 eff=0 cudnn=150)
```

**1650 total launches** → 6 full flushes at 256 each = 1536 pushed.

`cat /proc/cipher/classify_stats` AFTER the test:

```
classify_stats:
  total:        1536
  handled:      1536
  passthrough:  0
  per_op_class:
    ELEMENTWISE        1397
    ITERATIVE_CUSTOM   139
```

- **`total` and `handled` agree (1536):** every launch classified by may13_default; zero passthroughs (only one actuator registered).
- **`per_op_class` distribution makes sense**: SDPA generates many small elementwise kernels (1397) and a sprinkling of unclassified custom kernels (139). At cuLaunchKernel geometry granularity, what we see is the actual kernel mix Mistral SDPA + matmul generates, not the higher-level "this is GEMM" view.
- **The "no producer wired yet" hint is ABSENT** — confirms the proc emitter's `total == 0` branch correctly suppresses the hint when populated.
- **Launches between last flush boundary and python exit (1650 - 1536 = 114) are LOST.** Acceptable per SET semantics design — Step 6 explicitly chose 256-launch flush; partial batches are not flushed at process exit. v1.2.2 single-tenant scope tolerates this; ADD semantics + on-exit flush would be the multi-tenant improvement.

### LP-2 SDPA trampoline invariant — HELD

```
tramp_calls=150 handled=0 passthrough=150 (flash=0 eff=0 cudnn=150)
```

150 SDPA calls, 0 actuator HANDLED returns, 150 passthrough to orig() — Step 1 LP-2 invariant intact.

---

## 6.H — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Step 5 baseline. CLASSIFY runs at CUPTI time, independent of the /dev/cipher kmod path CP 5.4 tests exercise.

---

## 6.I — Commits + tags — PASS

### cipher_kmod commit `0ce4b8e2` (tag `week-2-step-6-ioctl-classify-push`)

```
 cipher_dev.c      |  2 ++
 cipher_internal.h |  5 +++++
 cipher_ioctl.h    | 17 +++++++++++++++++
 cipher_proc.c     | 23 +++++++++++++++++++++++
 4 files changed, 47 insertions(+)
```

Rollback: `git -C cipher_kmod reset --hard week-2-step-5-classify-proc-node`

### cipher_rt_phase4 commit `f9c32322` (tag `week-2-step-6-cupti-classify-wired`)

```
 cipher_cupti.c                | 31 +++++++++++++++++++++++++++
 cipher_rt_classify_observer.c | 54 +++++++++++++++++++++++++++++++++++++++++++
 cipher_rt_classify_observer.h |  7 ++++++
 3 files changed, 92 insertions(+)
```

Rollback: `git -C cipher_rt_phase4 reset --hard week-2-step-4-classify-observer-stub`

---

## Discipline notes

- Two commits (one per tree). Tag chain extended on both: `week-2-step-5-classify-proc-node → week-2-step-6-ioctl-classify-push` (kmod), `week-2-step-4-classify-observer-stub → week-2-step-6-cupti-classify-wired` (rt_phase4).
- Sites 1+2 left untouched per pre-flight DESIGN-MISMATCH adjudication. CUPTI catches their downstream launches naturally.
- Honest partial-batch loss flagged in §G.3 (last 0-255 launches before process exit are not flushed); acceptable for v1.2.2 SET semantics.
- The integration payoff is realized: `/proc/cipher/classify_stats` shows non-zero counters end-to-end through the Step 1-5 substrate stack.

---

## Anchors at close

| tree | HEAD | tag |
| --- | --- | --- |
| cipher_rt_phase4 | `f9c32322` | `week-2-step-6-cupti-classify-wired` |
| cipher_kmod | `0ce4b8e2` | `week-2-step-6-ioctl-classify-push` |
| cipher-may13-evidence | `fc8a9ae6` | (unchanged) |

Week 2 progress: **6/7 steps complete**. Step 7 (closeout) follows.
