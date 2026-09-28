# Week 3 Step 4 Option II-a — LIVE Flip + SENSE + Transition Wrapper — RESULT

**Status: PASS.**

Four variables changed in one step (LIVE flip + SENSE init/observe wiring + invented transition-detection wrapper + DSM PROPOSE ioctl). Mistral-7B SC6 bit-identical gate cleared. Pattern (a) DSM proposals queue ships; CIPHER_SENSE=1 end-to-end smoke produces 5 transition proposals visible at `/proc/cipher/dsm_proposals`. CP 5.4 15/15 byte-identical.

**Date:** 2026-05-20
**Phase:** v1.2.2 §7 Week 3, Step 4 of 5 (Option II-a per user adjudication)
**Anchors:**
  - rt_phase4: `4f1a86ab` → `79c1b4f9` (tag `week-3-step-4-opt2a-dispatch-live-sense`)
  - kmod: `0ce4b8e2` → `a21a45ee` (tag `week-3-step-4-opt2a-dsm-propose`)

---

## A — Pre-edit verification + snapshot — PASS

| signal | value |
| --- | --- |
| rt_phase4 HEAD | `4f1a86ab` ✓ |
| kmod HEAD | `0ce4b8e2` ✓ |
| rt_phase4 pre-build | rc=0, md5 `7994a586`, warnings 78, 311 dyn symbols |
| kmod pre-build | rc=0, md5 `8d1f1fc4`, warnings 1 |
| step4_pre snapshots | preserved at `/tmp/week3_step4_opt2a/` |

---

## B+C+D — Subs 1+2+3: CUPTI → SENSE bridge (~70 LOC in cipher_cupti.c)

Stack-allocated `CipherRingEntry` constructed after the CLASSIFY block; lazy `cipher_sense_init()` via static atomic CAS (first-reader-wins); conditional `cipher_sense_observe(&entry)` under `CIPHER_SENSE` env (default OFF; ~3 cycles when OFF). Bypasses `g_cipher_10ops` entirely (per pre-flight Sub-1 reversal: `cipher_sense.cpp` is self-contained; no `cipher_10ops_impl.cpp` port needed).

```c
{
    static atomic_int g_sense_env_inited = 0;
    static atomic_int g_sense_env_on     = 0;
    /* CAS-cache cipher_sense_init() first-reader-wins */
    /* if CIPHER_SENSE on: build CipherRingEntry, observe, transition_observe */
}
```

Tenant identity proxy: `gettid()`. Fingerprint: `cipher_sense_current_session()`.

---

## E — Sub-4: invented transition-detection wrapper (NEW ~230 LOC)

Two new files: `cipher_rt_sense_transition.{c,h}`.

| component | LOC | semantics |
| --- | ---:| --- |
| Tenant table (256 slots, direct-mapped hash) | ~30 | Atomic load/store; collision = evict-and-reset; lock-free reads |
| Hysteresis state machine | ~50 | N=8 stability + M=4 transition-debounce thresholds (best-guess v1) |
| Tool-idle heuristic | ~15 | AGENT class stable + gap > K=100ms since last observation |
| Proposal queue (64 slots, monotonic atomic head) | ~30 | Drop-oldest-on-overflow; observability-only |
| Lazy /dev/cipher fd cache | ~25 | pthread_mutex on init only; lock-free reads after |
| `..._observe()` hot-path entry | ~50 | Lock-free; ~50-100 ns added per launch under SENSE=ON |
| `..._flush()` ioctl push | ~30 | Drains queue; mirrors `cipher_rt_classify_push_to_kmod` |
| 2 diagnostic accessors | ~5 | queued/pushed counts |

**Research-grade caveat documented inline:** thresholds (N=8, M=4, K=100ms) are v1 best-guesses; measurement-driven tuning deferred to future weeks. Proposals are observability-only (Pattern a); no auto-action.

---

## F+G — Subs 5+6: kmod CIPHER_DSM_PROPOSE ioctl + proposals queue (~175 LOC)

| component | LOC |
| --- | ---:|
| `cipher_ioctl.h`: `struct cipher_dsm_propose_push` (56 B) + `CIPHER_DSM_PROPOSE` _IOW(magic, 26, ...) | 19 |
| `cipher_internal.h`: `CIPHER_PROC_DSM_PROPOSALS` macro + ring struct + extern + prototype | 29 |
| `cipher_main.c`: ring storage `cipher_dsm_proposals` (BSS zero) | 6 |
| `cipher_proc.c`: ioctl impl + show callback (with reason-name table) + proc_create unrolled-cleanup + proc_remove | 119 |
| `cipher_dev.c`: dispatch case | 2 |

Ring size 256 (~14 KiB). Pattern (a) observability only — operator reads, no auto-action.

---

## H — Sub-7: userspace push pathway

Folded into `cipher_rt_sense_transition.c::cipher_rt_sense_transition_flush()`. Piggybacks on existing 256-launch flush in `cipher_cupti.c:189`. Pattern mirrors Week 2 Step 6's `cipher_rt_classify_push_to_kmod`.

---

## I — Sub-8: CIPHER_DISPATCH_LIVE=1 flip

`cipher_rt_dispatch.cpp::read_live_env_once()`:
- Default flipped from 0 to 1 (`int parsed = 1;`)
- Explicit `CIPHER_DISPATCH_LIVE=0` becomes the rollback path
- Load banner updated: "LIVE=1 (1 = classifier-driven routing engaged by default; set CIPHER_DISPATCH_LIVE=0 to disable)"

---

## J — Build + symbol audit — PASS

### cipher_kmod

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 1 → 1 (zero delta) |
| md5 | `8d1f1fc4` → `8401f31a` |

### cipher_rt_phase4

| signal | value |
| --- | --- |
| build rc | 0 |
| warning count | 78 → 78 (zero delta) |
| md5 | `7994a586` → `daaccc40` |
| Symbol count | 311 → 315 (+4) |
| Symbols removed | 0 |
| Undef audit | clean |

### New T-symbols

```
T cipher_rt_sense_transition_observe
T cipher_rt_sense_transition_flush
T cipher_rt_sense_transition_proposals_queued
T cipher_rt_sense_transition_proposals_pushed
```

---

## K — Runtime smoke + LOAD-BEARING SC6 gates — PASS

### K.1 Loader smoke (LIVE=1 banner)

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] DISPATCH: substrate registered, LIVE=1 (1 = classifier-driven routing engaged by default; set CIPHER_DISPATCH_LIVE=0 to disable)
rc=0
```

### K.2 Kmod reload

```
$ sudo rmmod cipher_kmod && sudo insmod cipher_kmod.ko
insmod rc=0
/proc/cipher/: arenas bar0_state classify_stats dsm_proposals flops gpu_state migrations stats
                                                 ^^^^^^^^^^^^^ NEW
$ cat /proc/cipher/dsm_proposals
dsm_proposals:
  total_received: 0
  ring_size:      256
  emitting:       0 (most recent)
  (no proposals yet — userspace transition wrapper has not pushed)
```

### K.3 SC6 TinyLlama (~33s each)

| mode | result |
| --- | --- |
| vanilla | PASS 7/7 bit-identical |
| CIPHER+LIVE=1 | PASS 7/7 bit-identical |

### K.4 SC6 Mistral-7B (LOAD-BEARING, ~57s each)

| mode | result |
| --- | --- |
| vanilla | PASS 7/7 bit-identical |
| CIPHER+LIVE=1 | **PASS 7/7 bit-identical** |

**Load-bearing gate cleared.** Four-variable change preserves byte-identity at the binding correctness check.

### K.5 SENSE+PROPOSE end-to-end (CIPHER_SENSE=1, ~58s)

```
SC6 Mistral-7B/shared: PASS (7/7 bit-identical)

$ cat /proc/cipher/dsm_proposals
dsm_proposals:
  total_received: 5
  ring_size:      256
  emitting:       5 (most recent)
  entries (newest first):
    tenant=1790104  0->3  reason=TRANSITION_DETECTED  conf=85  age=17711 ms
    tenant=1790102  0->3  reason=TRANSITION_DETECTED  conf=85  age=17737 ms
    tenant=1790105  0->3  reason=TRANSITION_DETECTED  conf=85  age=17743 ms
    tenant=1790103  0->3  reason=TRANSITION_DETECTED  conf=85  age=17745 ms
    tenant=1789990  0->3  reason=TRANSITION_DETECTED  conf=85  age=23149 ms
```

Five transition proposals emitted across 5 distinct thread-tenants (4 consumer threads + 1 producer). All transitions are class `0 (UNKNOWN)` → `3 (BATCH_BACKGROUND)`, which is the expected SENSE classification for the SC6 forward-pass workload (no human-interactive latency, no agent-iteration pattern; large-batch forward = BATCH). Confidence 85 from SENSE's stability counter.

Pattern (a) observability path end-to-end functional.

---

## L — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

Byte-identical to Step 3 baseline.

---

## M — Commits + tags — PASS

### cipher_kmod commit `a21a45ee` (tag `week-3-step-4-opt2a-dsm-propose`)

```
 cipher_dev.c      |   2 +
 cipher_internal.h |  29 ++
 cipher_ioctl.h    |  19 ++
 cipher_main.c     |   6 +
 cipher_proc.c     | 119 +++++++++++++++++++++++++
 5 files changed, 175 insertions(+)
```

### cipher_rt_phase4 commit `79c1b4f9` (tag `week-3-step-4-opt2a-dispatch-live-sense`)

```
 Makefile                     |  10 +-
 cipher_cupti.c               |  70 ++++++++++++
 cipher_rt_dispatch.cpp       |  10 +-
 cipher_rt_sense_transition.c | 242 +++++++++++++++++++++++++++++++++++++++++++
 cipher_rt_sense_transition.h |  65 ++++++++++++
 5 files changed, 391 insertions(+), 6 deletions(-)
```

---

## Discipline notes

- Four variables changed in one step (LIVE flip + SENSE wiring + invented wrapper + DSM ioctl). Mistral-7B SC6 cleared as the binding correctness gate.
- Sub-4 wrapper is research-grade; thresholds (N=8, M=4, K=100ms) documented inline as v1 best-guesses requiring future measurement-driven tuning.
- Proposals are observability-only (Pattern a); no auto-action on the kmod side.
- VOLT 1000 vs 1200 MHz discrepancy from Step 3 remains unresolved (existing LUT honors may13-measured optimum).
- Cb.2 invariants preserved: existing ioctl nrs unchanged; reserved 2/3/4 still -ENOSYS; new nr=26 follows the additive-only pattern.

---

## Anchors at close

| tree | HEAD | tag chain |
| --- | --- | --- |
| cipher_rt_phase4 | `79c1b4f9` | week-3-step-4-opt2a-dispatch-live-sense |
| cipher_kmod | `a21a45ee` | week-3-step-4-opt2a-dsm-propose |
| cipher-may13-evidence | `fc8a9ae6` | (unchanged) |
| libcipher_rt.so md5 | `daaccc40` | (nvcc non-deterministic) |
| cipher_kmod.ko md5 | `8401f31a` | (kmod deterministic) |

Week 3 progress: **4/5 steps complete**. Step 5 closeout follows.
