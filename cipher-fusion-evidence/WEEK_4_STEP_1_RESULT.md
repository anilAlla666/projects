# Week 4 Step 1 — Selective Oracle Init + Observer Permit Upgrade — RESULT

**Status: PASS.**

Real `cipher_oracle_decide()` replaces Week 3 Step 2's hardcoded `permit=1` in `cipher_rt_classify_observer_observe()`. Both load-bearing SC6 gates clear bit-identical (TinyLlama + Mistral-7B, vanilla and CIPHER-injected). CP 5.4 isolation 15/15 byte-identical. Real oracle confirmed live via banner under direct CUPTI smoke.

**Date:** 2026-05-20
**Phase:** v1.2.2 §7 Week 4, Step 1 of 6 (Q2 approach b — selective init)
**Anchors:**
  - cipher_rt_phase4: `79c1b4f9` → `4279461` (tag `week-4-step-1-real-oracle`)
  - cipher_kmod: `a21a45e` (unchanged — Step 1 is userspace-only)
  - cipher-may13-evidence: `fc8a9ae` (unchanged)
  - cipher-fusion-evidence: scope-lock `850bd8b`

---

## A — Pre-edit verification + baseline snapshot — PASS

| signal | value |
| --- | --- |
| rt_phase4 HEAD | `79c1b4f9` ✓ matches scope-lock |
| kmod HEAD | `a21a45e` ✓ matches scope-lock |
| may13-evidence HEAD | `fc8a9ae` ✓ matches scope-lock |
| WEEK_4_SCOPE_LOCK.md | 17891 B, commit `850bd8b` ✓ |
| Snapshot pre | md5 `daaccc40` → `/tmp/week4_step1/libcipher_rt.so.step1_pre` |
| Pre-build clean | rc=0, md5 `81ad96af` (nvcc non-deterministic), 78 warnings |
| Pre-edit SC6 TinyLlama vanilla | PASS 7/7 bit-identical (~33s) |
| Pre-edit SC6 TinyLlama CIPHER | PASS 7/7 bit-identical (~33s) |

Both pre-edit modes PASS; no pod drift before Step 1. Snapshot preserved at `/tmp/week4_step1/`.

---

## B — Oracle API surface inspection — PASS with brief-correction

Read `include/may13/cipher_oracle.h` (9256 B) + `src/may13/cipher_oracle.cpp` (full).

| signal | value | brief vs. reality |
| --- | --- | --- |
| State type | `CipherOracleState` (~3+ KiB; ema arrays + phase + counters + billing) | matches |
| Query type | `CipherOracleQuery` | brief said "CipherOracleInput" — wrong name |
| Result type | `CipherOracleResult { .decision, .reason }` | brief said `.permit` — wrong field |
| Decision enum | `CIPHER_ORACLE_PERMIT=0`, `CIPHER_ORACLE_DENY=1` | matches semantics, not field path |
| `cipher_oracle_init` sig | `void(state*, CipherLiquidStateMgr*, const CipherOracleConfig*)` | matches; NULL for both extra args is supported (line 88-93 + line 85) |
| `cipher_oracle_decide` sig | `CipherOracleResult(state*, const CipherOracleQuery*)` returns by value | matches semantics |
| Threading | reads atomic counters; calls into `cipher_struct_lookup` (cache-only). No mutex internally; per-launch state mutation OK from CUPTI callback thread | safe |

### Pure-function purity verification of `cipher_oracle_init(state, NULL, NULL)`

Read oracle.cpp:80-119 (`cipher_oracle_init`):
- `memset(state, 0, sizeof(*state))` — BSS clear ✓
- `state->liquid = liquid` — sets to NULL ✓ (NULL-tolerant: every later reference to `state->liquid` is guarded by `if (state->liquid && state->liquid->initialized)`, verified at lines 197-199, 249-250, 280-285, 374-375, 396-397, 408-409)
- `if (cfg)` then copy; else apply `CIPHER_ORACLE_DEFAULT_CONFIG` ✓
- Initialize `state->ema.sigma[]` to 1.0f ✓
- `state->phase.detected_phase = 0` (WARMUP) ✓
- `cipher_struct_lookup_init()` call → read structural_lookup.cpp:169-188: pure cache-clear + pre-warm 42 rules + fprintf ✓
- `state->initialized = true` + fprintf banner ✓

**Zero CUDA / NVML / kmod / dlsym / mmap / file-open / network calls.** Safe under cipher_rt_phase4's deferred-init model.

### Behavior delta surface (real oracle vs hardcoded permit=1)

Oracle starts in WARMUP phase (`detected_phase == 0`). On a pure inference workload (only GEMM/ELEMENTWISE/REDUCTION op classes), `topo_detect_inference` (oracle.cpp:30-74) flips phase to CONVERGENCE at the first 64-decision checkpoint (`TOPO_INFER_MIN=64` + `(total & 0x3F) == 0` checkpoint). Until that point: DENY (`reason="warmup"`). After: gates 2-5 (confidence ≥60 default, structural lookup, EMA permanent-demotion, N≤4 rule).

**Correctness preservation argument:** when oracle returns DENY, the observer's hint is not published (`g_tls_hint.valid = 0`); Marlin actuator's binding gate at L100-114 (per task brief) is the actual routing constraint, independent of hint. Hint absence falls back to Marlin's static-priority path. SC6 bit-identity should hold. Verified empirically below in §K.

---

## C — Call-site choice: Option 2 (new bridge TU) — FORCED by language boundary

Brief recommended Option 1 (inline in observer.c). **Forced to Option 2** by the C/C++ boundary:
- `cipher_oracle.h` transitively `#include`s `cipher_classify.hpp`
- `cipher_classify.hpp` uses `namespace cipher`, `enum class OpClass`, `<atomic>`, `<array>` — hard C++
- `cipher_rt_classify_observer.c` is a C TU compiled with `$(CC)` (Makefile:164)

**Option 2:** new `cipher_rt_oracle_bridge.{cpp,h}` exposing a plain-C surface (`init_lazy()` + `decide(uint8_t, uint8_t)`), compiled with `$(CXX) -std=c++17`. Owns the BSS state + `std::atomic<int>` CAS lazy-init flag.

Net cost: +99 LOC (64 .cpp + 35 .h) vs ~+25 inline. Justified — language boundary is non-negotiable.

---

## D — Implementation (3 files modified, 2 new) — PASS

### New: `cipher_rt_oracle_bridge.h` (35 LOC)

Plain-C surface:
```c
void cipher_rt_oracle_bridge_init_lazy(void);
int  cipher_rt_oracle_bridge_decide(uint8_t op_class, uint8_t confidence);
```

### New: `cipher_rt_oracle_bridge.cpp` (64 LOC)

```cpp
namespace {
CipherOracleState g_rt_oracle{};
std::atomic<int>  g_inited{0};
}  /* anonymous namespace */

void cipher_rt_oracle_bridge_init_lazy(void) {
    if (g_inited.load(std::memory_order_acquire) != 0) return;
    int expected = 0;
    if (g_inited.compare_exchange_strong(expected, 1)) {
        cipher_oracle_init(&g_rt_oracle, nullptr, nullptr);
    }
}

int cipher_rt_oracle_bridge_decide(uint8_t op_class, uint8_t confidence) {
    if (g_inited.load(std::memory_order_acquire) == 0) return 0;
    CipherOracleQuery q{};
    q.op_class    = op_class;
    q.confidence  = confidence;
    q.kernel_name = nullptr;
    /* layer_idx/total_layers/is_backward/is_optimizer default-zero */
    CipherOracleResult r = cipher_oracle_decide(&g_rt_oracle, &q);
    return (r.decision == CIPHER_ORACLE_PERMIT) ? 1 : 0;
}
```

Field mapping rationale:
- `op_class` and `confidence` come directly from `out` (the classify result available at observer hot path)
- `kernel_name = nullptr` — CUPTI callback doesn't carry kernel demangled name through the classify substrate; `cipher_struct_lookup` handles NULL safely (uses cache miss → default rule)
- `layer_idx = 0`, `total_layers = 0` — unknown from CUPTI; oracle's N≤4 rule guard `if (q->layer_idx < CIPHER_ORACLE_MAX_LAYERS)` still triggers, but `state->sub_counter[0]` is the only slot mutated; not a correctness issue for inference workload
- `is_backward = is_optimizer = false` — inference-only assumption for v1; Wave 5 W4 doesn't require backward/optimizer support

### Modified: `cipher_rt_classify_observer.c` (44 LOC delta)

- Added `#include "cipher_rt_oracle_bridge.h"`
- Replaced the `int permit = 1` block (lines 107-124) with:
  ```c
  cipher_rt_oracle_bridge_init_lazy();
  permit = cipher_rt_oracle_bridge_decide(out->op_class, out->confidence);
  ```
- Updated the comment block describing the v1 hardcoded-shortcut rationale to document the Week 4 Step 1 wiring + correctness preservation argument

### Modified: `Makefile` (15 LOC delta)

- Added `cipher_rt_oracle_bridge.o` to `OBJS` (between `cipher_rt_dispatch.o` and `cipher_rt_sense_transition.o`)
- Added compile rule using `$(CXX) -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1` (matches `may13_cipher_oracle.o` pattern)
- Added `cipher_rt_oracle_bridge.h` to observer.o dependency list

---

## E — Makefile delta — PASS

| signal | value |
| --- | --- |
| OBJS list | +1 (cipher_rt_oracle_bridge.o) |
| Compile rules | +1 (CXX -std=c++17) |
| Observer.o deps | +1 (cipher_rt_oracle_bridge.h) |
| Total LOC delta | +15 |

Pattern follows existing `cipher_rt_dispatch.o` and `may13_cipher_oracle.o` rules — no new build infrastructure.

---

## F — Build + symbol audit — PASS

| signal | pre | post | delta |
| --- | --- | --- | --- |
| build rc | 0 | 0 | 0 |
| warning count | 78 | 78 | 0 |
| libcipher_rt.so md5 | `daaccc40` | `d60c6256` | changed (nvcc non-deterministic; source identity via git diff) |
| dynsym total | 315 | 317 | +2 |
| dynsym T (defined) | 174 | 176 | +2 |
| dynsym U (undef) | (set) | (set) | unchanged |

### New T-symbols (exactly the 2 bridge functions):
```
T cipher_rt_oracle_bridge_decide
T cipher_rt_oracle_bridge_init_lazy
```

Zero existing symbols removed. Undef set unchanged → linkage clean.

---

## G — Runtime smoke + LOAD-BEARING SC6 gates — PASS

### G.1 — Loader smoke
```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
[cipher_v2] CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)
[cipher_v2] DISPATCH: substrate registered, LIVE=1 (...)
rc=0
```

Banners unchanged from Week 3 Step 4 (oracle init is lazy — fires on first observe() call, not at .so load). PASS.

### G.2 — Direct CUPTI smoke (verify real oracle runs)
```
$ CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 -c "
    import torch
    a = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    b = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    for _ in range(100): c = a @ b
    torch.cuda.synchronize()"
[cipher_v2] DISPATCH: substrate registered, LIVE=1 (...)
[cipher_v2] MARLIN: actuator DISABLED (CIPHER_MARLIN not set)
[CIPHER L3.8] Structural lookup initialized. 42 rules. Name cache pre-warmed.
[CIPHER ORACLE] Initialized.
  N_max:          4
  EMA κ:          0.999
  Divergence:     2.0σ
  Warmup steps:   500
  Min confidence: 60%
```

**Real oracle confirmed live.** Lazy CAS init fires on the first PyTorch GEMM that traverses the CUPTI hot path → observer hot path. No crashes, no hangs.

### G.3 — SC6 TinyLlama gates (33s each)

| mode | result |
| --- | --- |
| vanilla | PASS 7/7 bit-identical |
| CIPHER-injected (real oracle) | PASS 7/7 bit-identical |

### G.4 — SC6 Mistral-7B (LOAD-BEARING, ~57s each)

| mode | result |
| --- | --- |
| vanilla | PASS 7/7 bit-identical |
| CIPHER-injected (real oracle) | **PASS 7/7 bit-identical** |

**Load-bearing gate cleared.** Real oracle runs end-to-end through Mistral-7B 4-tenant N=4 weight-sharing forward pass; correctness preserved. Hint-absence-during-warmup correctness argument from §B validated empirically: even though oracle starts in WARMUP and produces DENY for the first ~64 decisions, hint absence falls back to Marlin's binding gate, leaving Mistral SDPA correctness untouched.

(SC6 child-process logs don't surface oracle stderr because Popen redirects subprocess stderr; G.2 above confirms init does fire in the relevant hot path.)

---

## H — CP 5.4 regression — PASS

```
$ cd /home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3
$ ./cp54_isolation_test
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

15/15 byte-identical to Week 3 closeout baseline. Oracle runs in userspace; kmod unchanged; CP 5.4 unaffected.

---

## I — Commit and tag — PASS

```
[master 4279461] Week 4 Step 1: real cipher_oracle_decide via selective bridge

 Makefile                      | 15 ++++++++--
 cipher_rt_classify_observer.c | 44 ++++++++++++++++-------------
 cipher_rt_oracle_bridge.cpp   | 64 +++++++++++++++++++++++++++++++++++++++++++
 cipher_rt_oracle_bridge.h     | 35 +++++++++++++++++++++++
 4 files changed, 137 insertions(+), 21 deletions(-)
 create mode 100644 cipher_rt_oracle_bridge.cpp
 create mode 100644 cipher_rt_oracle_bridge.h
```

Tag: `week-4-step-1-real-oracle` (cipher_rt_phase4).

Identity: `Anil <anil.0666369@gmail.com>` (per-command GIT_AUTHOR_* env vars).

---

## Anchors at close

| tree | HEAD | tag chain |
| --- | --- | --- |
| cipher_rt_phase4 | `4279461` | week-4-step-1-real-oracle (NEW) |
| cipher_kmod | `a21a45e` | week-3-step-4-opt2a-dsm-propose (unchanged) |
| cipher-may13-evidence | `fc8a9ae` | (unchanged) |
| libcipher_rt.so md5 | `d60c6256` | (nvcc non-deterministic) |
| cipher_kmod.ko md5 | (unchanged from W3) | (no kmod change) |

---

## Discipline notes

- Brief Step C "Option 1 (in observer.c)" was forced to Option 2 (new bridge TU) by the C/C++ language boundary — surfaced and documented; not a mitigation in the bad-discipline sense, a fact about the surface.
- Brief Step D used `oracle_in`/`oracle_out` and `.permit` field names that don't match the on-disk API; corrected to `CipherOracleQuery`/`CipherOracleResult` with `.decision` enum comparison. No semantic deviation.
- Q3 VOLT 1000-vs-1200 MHz discrepancy from Step 3 remains a deferred-residue item, untouched in Step 1.
- Q1 cipher_10ops_impl.cpp continues skipped: this step relied only on the existing may13 oracle ports.
- No new ioctls; ABI Cb.2 invariants preserved (existing nrs unchanged; reserved 2/3/4 still -ENOSYS).

---

## Progress

Week 4: **1/6 steps complete.** Next: Step 2 — Tier A observability ports (cipher_loop, cipher_pipeline, cipher_pulse, cipher_continuity; ~970 source LOC + headers; ~5-7h).

Awaiting adjudication.
