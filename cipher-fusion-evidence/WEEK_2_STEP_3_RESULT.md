# Week 2 Step 3 — `cipher_rt_classify_substrate.{cpp,h}` Scaffolding — RESULT

**Status: PASS.**

Third substrate in the cipher_rt_phase4 actuator-registry family (matmul → attn → classify). Priority-ordered 16-slot registry mirroring `cipher_rt_matmul_dispatch.{c,h}`. may13 default classifier auto-registered at priority 0, wrapping `cipher::classify_launch()`. Registry is quiescent (no hot-path caller yet) — Step 6 wires it in.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 3 of 7
**Anchors:**
  - Pre: `23014c1e` (`week-2-step-2-may13-ports-a4-h1-root`)
  - Post: `7629cf60` (`week-2-step-3-classify-substrate-scaffold`)
  - libcipher_rt.so md5: `bc249c9d` (Step 2 D1 commit-time anchor) → `c20717f3` (post-Step-3 build)

---

## 1. Sub-step A — Pre-edit verification — PASS (with build-non-determinism note)

| signal | value | expected | match |
| --- | --- | --- | --- |
| HEAD | `23014c1e` | `23014c1e...` | ✓ |
| working tree | clean | clean | ✓ |
| Pre-edit build rc | 0 | 0 | ✓ |
| Pre-edit md5 | `e16df0a4...` | `bc249c9d...` | **drift** (see note) |
| Pre-edit warnings | 78 | 78 | ✓ |
| Symbol count | 298 | 298 | ✓ |
| Loader smoke | rc=0 | rc=0 | ✓ |
| Undef set | clean (weak edmd-family only) | clean | ✓ |

### Build-non-determinism note

A pre-edit `make clean && make` produced libcipher_rt.so with md5 `e16df0a4...`, not the Step 2 D1 commit-time anchor `bc249c9d...`. Investigation:

- `git status` clean; `git diff 23014c1e` empty (no source drift)
- Two consecutive clean builds produced two different md5s (`e16df0a4` and `9d9789c1`) — **back-to-back non-determinism**
- All functional invariants hold: loader rc=0, symbol surface clean (only expected weak undefs), CP 5.4 PASS

Root cause: nvcc fatbin embedding `__DATE__` / `__TIME__` (or similar) into the .cu compile output. The 3 ported `.cu` files (`cipher_green_ctx.cu`, `cipher_l2_persist.cu`, `cipher_liquid_state.cu`) compile through `$(NVCC) $(NVCCFLAGS)`; nvcc's fatbin builder timestamps each compile.

**Treatment: note-and-proceed.** Source identity is git-verified (`git rev-parse HEAD` + `git diff <commit>`); md5 reproducibility is not a reliable source-identity check given nvcc behavior. The brief's anchor check is a sanity gate; the underlying intent (no source drift) is verified by git. Functional invariants are the binding correctness check, and all pass.

**Future hardening (not in scope):** can be addressed by `-Xcompiler -frandom-seed=…` + `-D__DATE__=` / `-D__TIME__=` macros for nvcc, or by setting `SOURCE_DATE_EPOCH`. Surfacing only; no fix applied.

---

## 2. Sub-step B — Reference shape from `cipher_rt_matmul_dispatch.{h,c}` — captured

Read both files. The matmul pattern (which classify mirrors exactly):

- `struct cipher_rt_matmul_call` — opaque call descriptor with `uint64_t reserved[4]` tail
- `enum cipher_rt_matmul_result { HANDLED=0, PASSTHROUGH=1, ERROR=2 }`
- `struct cipher_rt_matmul_actuator { name, priority, maybe_handle }`
- `cipher_rt_matmul_register_actuator()` — insertion-sort by priority (lower runs first, stable)
- `cipher_rt_matmul_dispatch()` — atomic counter, iterate, HANDLED short-circuits, ERROR logged + falls through
- Static `g_disp` state: `atomic_int inited`, `pthread_mutex_t reg_lock`, `int n_actuators`, `actuators[16]`, atomic telemetry counters
- `cipher_rt_matmul_dispatch_init()` — atomic-exchange of inited flag, registers `atexit` diagnostic
- 3 diagnostic accessors: `_calls_total/_handled/_passthrough`

attn_dispatch follows the same pattern with REDIRECTED as a 3rd extension.

---

## 3. Sub-step C — `cipher_rt_classify_substrate.h` (120 lines) — designed

Public surface (matches matmul shape; differences documented in §C.4):

```c
struct cipher_rt_classify_call {
    const void *fn;
    uint32_t    grid_x, grid_y, grid_z;
    uint32_t    block_x, block_y, block_z;
    uint32_t    shared_bytes;
    uint32_t    reserved_pad;          /* alignment */
    uint64_t    reserved[6];           /* Cb.2 future fields */
};

struct cipher_rt_classify_out {
    uint8_t     op_class;              /* OpClass enum (0..6) or UNCLASSIFIED=0xFF */
    uint8_t     confidence;            /* 0..100 */
    uint8_t     cache_hit;             /* 0/1 */
    uint8_t     reserved_pad[5];       /* alignment */
    uint64_t    reserved[2];           /* Cb.2 future fields */
};

#define CIPHER_RT_CLASSIFY_UNCLASSIFIED 0xFF

enum cipher_rt_classify_result {
    CIPHER_RT_CLASSIFY_HANDLED     = 0,
    CIPHER_RT_CLASSIFY_PASSTHROUGH = 1,
    CIPHER_RT_CLASSIFY_ERROR       = 2,
};

struct cipher_rt_classify_actuator {
    const char *name;
    int         priority;
    int       (*classify)(const struct cipher_rt_classify_call *call,
                          struct cipher_rt_classify_out        *out);
};

#define CIPHER_RT_CLASSIFY_MAX_ACTUATORS 16

int cipher_rt_classify_register_actuator(const struct cipher_rt_classify_actuator *);
int cipher_rt_classify_dispatch_init(void);
int cipher_rt_classify_route(const struct cipher_rt_classify_call *,
                              struct cipher_rt_classify_out *);

unsigned long cipher_rt_classify_calls_total(void);
unsigned long cipher_rt_classify_calls_handled(void);
unsigned long cipher_rt_classify_calls_passthrough(void);
```

### C.4 — Divergences from `cipher_rt_matmul_dispatch.h`

- **Result enum:** matched matmul order (HANDLED=0, PASSTHROUGH=1, ERROR=2). The brief's "PASSTHROUGH=0, HANDLED=1" was an example; the brief itself said "match matmul patterns", which take precedence.
- **Function naming:** `cipher_rt_classify_register_actuator` (not the brief's `cipher_rt_classify_register`) — exact matmul/attn naming convention. `cipher_rt_classify_route()` (not `cipher_rt_classify_dispatch()`) — chosen to avoid namespace collision with may13's `cipher_dispatch()` symbol.
- **Output struct:** added `CIPHER_RT_CLASSIFY_UNCLASSIFIED` sentinel (0xFF) matching `cipher::OpClass::UNCLASSIFIED`. Route() initializes out → UNCLASSIFIED before iterating so callers see a defined state even if every actuator passes through.

These are matmul/attn-pattern consistency improvements over the brief's example surface, not semantic changes.

---

## 4. Sub-step D — `cipher_rt_classify_substrate.cpp` (180 lines) — implemented

Implementation choices:

### D.1 Registry storage — anonymous namespace, std::atomic + pthread_mutex

C++ idiom (mirroring matmul's C-style `g_disp` static struct):

```cpp
namespace {
struct ClassifySubstrateState {
    std::atomic<int>          inited{0};
    pthread_mutex_t           reg_lock = PTHREAD_MUTEX_INITIALIZER;
    int                       n_actuators = 0;
    cipher_rt_classify_actuator actuators[CIPHER_RT_CLASSIFY_MAX_ACTUATORS];
    std::atomic<unsigned long> total{0}, handled{0}, passthrough{0};
};
ClassifySubstrateState g_classify;
}
```

Matmul uses C11 `atomic_int` + `<stdatomic.h>`; classify uses C++ `std::atomic<>` because the source is `.cpp`. Functional behavior is identical.

### D.2 `cipher_rt_classify_register_actuator()`

Insertion-sort by priority (lower runs first, stable for equal). Identical algorithm to matmul, mutex-guarded.

### D.3 `cipher_rt_classify_route()`

- Atomic total counter increment
- Initialize `out` to UNCLASSIFIED before iterating
- Snapshot `n_actuators` (registry is append-only after init)
- For each actuator: call `classify(call, out)`:
  - HANDLED → atomic handled++, return HANDLED
  - ERROR → log and break (fall through to passthrough)
  - PASSTHROUGH → continue to next
- If no HANDLED: atomic passthrough++, return PASSTHROUGH (caller still sees out->op_class = UNCLASSIFIED)

### D.4 `may13_default_classifier` (10 LOC + null guard)

```cpp
int may13_default_classifier(const cipher_rt_classify_call *call,
                             cipher_rt_classify_out *out) {
    if (!call || !out) return CIPHER_RT_CLASSIFY_ERROR;
    cipher::ClassifyResult r = cipher::classify_launch(
        call->fn,
        call->grid_x, call->grid_y, call->grid_z,
        call->block_x, call->block_y, call->block_z,
        call->shared_bytes);
    out->op_class   = static_cast<uint8_t>(r.op);
    out->confidence = r.confidence;
    out->cache_hit  = r.cache_hit ? 1 : 0;
    return CIPHER_RT_CLASSIFY_HANDLED;
}
```

### D.5 Auto-registration at static-init

```cpp
struct AutoRegister {
    AutoRegister() {
        cipher_rt_classify_actuator a = {};
        a.name     = "may13_default";
        a.priority = 0;
        a.classify = may13_default_classifier;
        cipher_rt_classify_register_actuator(&a);
    }
};
AutoRegister g_auto_register;
```

This is a C++ static-init constructor — fires at `dlopen(libcipher_rt.so)` time before any user code. The constructor only **stores a function pointer**; it does NOT call `cipher::classify_launch()` (which would be a hot-path read, not a static-init action). The pthread_mutex is `PTHREAD_MUTEX_INITIALIZER` so registration is safe at static-init time even before any explicit `cipher_rt_classify_dispatch_init()` call.

Per the impedance check §3 P1: `cipher::classify_launch` is a pure-function geometry classifier with no CUDA/NVML side effects, so even if we wanted to call it at static-init it would be safe. We don't, but the safety margin is built-in.

---

## 5. Sub-step E — Makefile diff

```
+NVCC    ?= nvcc  (already added Step 2)

  cipher_rt_attn_dispatch.o cipher_rt_attn_test_actuator.o \
  cipher_rt_audit.o \
+ cipher_rt_classify_substrate.o \
  cipher_may13_harness.o \
  ...

+# Week 2 Step 3 — classify-routing substrate (3rd in matmul/attn family).
+cipher_rt_classify_substrate.o: cipher_rt_classify_substrate.cpp \
+                                 cipher_rt_classify_substrate.h \
+                                 include/may13/cipher_classify.hpp
+   $(CXX) $(CXXFLAGS) -std=c++17 $(INCLUDES) -c -o $@ $<
```

11 lines added. Pattern matches `cipher_rt_attn_dispatch.o` (also `-std=c++17`).

---

## 6. Sub-step F — Build + nm diff — PASS

| signal | value | notes |
| --- | --- | --- |
| Build rc | 0 | |
| Warning count | 78 → 78 (zero delta after _GNU_SOURCE cleanup) | initial build produced +1 `"_GNU_SOURCE" redefined` cpp-warning; removed from .cpp since transitive header already provides |
| libcipher_rt.so md5 | `bc249c9d` (anchor) → `c20717f3` (post) | changes expected; nvcc non-determinism per §1 means consecutive builds differ |
| Symbol count | 298 → 304 | +6 new T-symbols |
| Symbols removed | **0** | |

### Added symbols (6 new T)

```
T cipher_rt_classify_calls_handled
T cipher_rt_classify_calls_passthrough
T cipher_rt_classify_calls_total
T cipher_rt_classify_dispatch_init
T cipher_rt_classify_register_actuator
T cipher_rt_classify_route
```

(The static `may13_default_classifier` is .cpp-local linkage; `AutoRegister` is hidden inside anonymous namespace. Neither appears in the dynamic symbol table — by design.)

### Undefined-symbol audit

```
$ nm -D --undefined-only libcipher_rt.so | filter system
                 U cuptiSubscribe@libcupti.so.12
```

Only `cuptiSubscribe` (already resolved by existing `-lcupti`). No new CIPHER undefs introduced.

---

## 7. Sub-step G — Runtime invariant verification — PASS

### G.1 Loader smoke (LD_PRELOAD /bin/true)

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
rc=0
```

The CLASSIFY substrate's static-init AutoRegister fires correctly. Banner visible. No symbol-lookup-error.

### G.2 SDPA shim injection smoke (Mistral cuDNN backend)

Reproduces Step 1 / Step 2 D1 pattern:

```
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] GREEN/CP54: ALLOCATE ok ...
[cipher_v2] MATMUL: substrate initialized (max 16 actuators; first registration awaited)
[cipher-attn] substrate active ...
[cipher_v2] GOT: patch applied -- 8 slot(s) across 71 module(s); 4 target(s) registered

SDPA mean: 0.01910400390625
finite: True

[cipher_v2] MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=0)
[cipher-attn] exit totals - tramp_calls=1 tramp_fake=0 observed=1
              handled=0 passthrough=1 redirected=0 (flash=0 eff=0 cudnn=1)

rc=0
```

- 17 cipher banners (was 16; +1 for CLASSIFY registration line)
- SDPA cuDNN backend trampoline: `tramp_calls=1 handled=0 passthrough=1` — Step 1 LP-2 invariant holds
- MATMUL stats unchanged: `calls=0 handled=0 passthrough=0`
- **CLASSIFY substrate is quiescent at runtime** (no exit-totals banner because `cipher_rt_classify_dispatch_init()` is never called — atexit is only registered if init() fires, which the unwired Step 3 substrate doesn't trigger). Expected per scope. Step 6 wiring will activate it.

### G.3 Registry registration verification

Confirmed via banner at .so load: `[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)`. Static-init AutoRegister works; the function pointer is stored at slot 0; route() (when called by Step 6) will dispatch through it.

---

## 8. Sub-step H — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

All 15 kmod-substrate isolation tests pass. Byte-identical behavior to Step 2 baseline.

---

## 9. Sub-step I — Commit and tag — PASS

```
git diff --cached --stat:
 Makefile                         |  11 +++
 cipher_rt_classify_substrate.cpp | 180 +++++++++++++++++++++++++++++
 cipher_rt_classify_substrate.h   | 120 ++++++++++++++++++++++++++
 3 files changed, 311 insertions(+)

HEAD:    7629cf60d31bbaaa9f613247d7af147ce1e1beec
Tag:     7629cf60... (week-2-step-3-classify-substrate-scaffold)
Pre tag: 23014c1e... (week-2-step-2-may13-ports-a4-h1-root)
pending: 0
```

---

## 10. What this unlocks

- **Week 2 Step 4** — `cipher_rt_classify_observer.c` stub. Will register a priority-N observer-only classifier on the classify substrate. Observer's `classify()` always returns PASSTHROUGH after side-effect counting; the route loop still falls through to may13_default at priority 0. Empty / placeholder structure per WEEK_2_SCOPE_LOCK.md §3.
- **Week 2 Step 5** — `/proc/cipher/classify_stats` proc node. Reads from the substrate's counter accessors (`cipher_rt_classify_calls_total/handled/passthrough` exposed by Step 3) and/or kmod-side counters pushed via ioctl. Step 5 design choice.
- **Week 2 Step 6** — Hot-path wiring. F1 cuLaunchKernel intercept calls `cipher_rt_classify_route(call, out)` per-launch; the result populates a per-launch op_class that may13 dispatch + matmul/attn shims can consult. The substrate registry remains the integration point for future classifier actuators.
- **Week 2 Step 7** — closeout.

---

## 11. Discipline notes

- Single commit. Tag chain extended: `week-2-step-2-may13-ports-a4-h1-root` → `week-2-step-3-classify-substrate-scaffold`.
- Rollback: `git reset --hard week-2-step-2-may13-ports-a4-h1-root`
- **Build-non-determinism note** (§1): nvcc fatbin output is timestamped, so consecutive clean builds produce different .so md5s. Source identity is git-verified. Functional invariants (loader, smoke, CP 5.4) are the binding check.
- The substrate is quiescent at runtime in Step 3 — registry exists, may13 default registered, but `cipher_rt_classify_route()` is unreached. Step 6 wiring is the next behavior-changing step.
- Three substrates now exist in cipher_rt_phase4 (matmul, attn, classify) following the same pattern. Future actuators (FP8 matmul, KV-dedup attn, learned-model classify) plug in via the same registration shape.

---

## 12. Anchors at close

```
cipher-fusion-evidence: (to be set when this doc commits)
cipher_rt_phase4:        7629cf60d31bbaaa9f613247d7af147ce1e1beec
                         tag: week-2-step-3-classify-substrate-scaffold
cipher_kmod:             f8572ecf422050a488f1fb45f74534ecd1bde678 (Week 1 close)
cipher-may13-evidence:   fc8a9ae6025cff345e86d054b1df02d07a9fb1fe (Week 1 close)
```

Week 2 progress: **3/7 steps complete** (LP-2 SDPA refactor + may13 ports + classify substrate scaffolding). Awaiting Step 4 brief.
