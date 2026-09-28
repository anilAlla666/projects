# Week 2 Step 2 v3 D1 — A4+H1 Port from ROOT Variants — RESULT

**Status: PASS.**

All 9 sub-steps (A.0, A, B, C, D, E, F, G, H, I) executed clean. The v3 PARTIAL source-selection error is corrected. ROOT-variant ports linked cleanly; runtime invariants verified end-to-end (loader smoke, SDPA shim injection with trampoline accounting `tramp_calls=1 handled=0 passthrough=1`, CP 5.4 isolation 15/15 PASS).

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 2 v3 D1 (third Step-2 attempt; A4+H1 scope with ROOT-variant source selection per D3 history audit)
**Anchors:**
  - Pre: `465b244e` (`week-2-step-1-lp2-sdpa-refactor`)
  - Post: `23014c1e` (`week-2-step-2-may13-ports-a4-h1-root`)
  - libcipher_rt.so md5: `56291439c7fd941ca37cf86c482948d7` → `bc249c9dfac9a1d5cb777d9d8a4d7964`

---

## 1. Sub-step A.0 — Full rollback of v3 PARTIAL working tree — PASS

| signal | value |
| --- | ---:|
| Pre-rollback state | 8 untracked + 1 modified (Makefile) |
| Post-rollback state | clean working tree, HEAD unchanged at `465b244e` |
| `include/may13/` count | 12 (Step 2 v2 baseline) |
| Clean rebuild rc | 0 |
| libcipher_rt.so md5 | `56291439c7fd941ca37cf86c482948d7` ✓ baseline reproduces |
| Warnings | 18 (baseline) |

---

## 2. Sub-step A — ROOT-variant re-validation — PASS

### A.1 — ROOT fresh compile matches prebuilt (byte-identical)

```
022c5176943dad86fbcf4dc8daeb10e5  /tmp/v3_d1/cipher_dispatch_root.o
022c5176943dad86fbcf4dc8daeb10e5  build/cipher_dispatch.o            ← byte-identical

542d892d66b432a9367a6a2dcd21c043  /tmp/v3_d1/cipher_oracle_root.o
542d892d66b432a9367a6a2dcd21c043  build/cipher_oracle.o              ← byte-identical
```

ROOT-CANONICAL conclusion from D3 reproduces cleanly.

### A.2 — A4 closure walker re-confirmation

`/tmp/symbol_closure_a4.py` re-ran with same result as A4 closure rerun (`be484e63`):

```
final closure size: 11 files
orphaned symbols:   1   (cipher_intercept_stats → H1 stub)
external (system): 4
.cu count:          3   (green_ctx, l2_persist, liquid_state)
```

Authoritative for ROOT. Proceeding with confidence.

---

## 3. Sub-step B — Header port + library invariant — PASS

| signal | value |
| --- | ---:|
| 6 new headers copied (md5-verified to evidence include/) | ✓ |
| `include/may13/` count | 12 → 18 |
| 15 bare-name rewrites applied | ✓ |
| Library invariant build rc | 0 |
| libcipher_rt.so md5 unchanged | `56291439...` ✓ HELD (no existing TU reaches new may13 surface) |
| Warnings | 18 (unchanged) |
| Synthetic 18-header compile smoke | rc=0, errors=0, warnings=0 |

---

## 4. Sub-step C — Port 11 ROOT-variant sources — PASS (the load-bearing change)

### C.1 ROOT-variant copies (2 paired files)

Source `cipher-may13-evidence/cipher_dispatch.cpp` (ROOT, **NOT** `src/`) → `cipher_rt_phase4/src/may13/cipher_dispatch.cpp`:

```
ROOT md5: 59271a09f88ec5a15f6d7bfb12238a44
dest md5: 59271a09f88ec5a15f6d7bfb12238a44  ✓
SRC md5:  59d805306298aa873e0f73cdbf0a9321  ≠ dest (correctly NOT copied)
```

Same byte-identical match for `cipher_oracle.cpp` (md5 `3b2c0d68...`).

### C.2 Line-count + functional-signature verification (the v3-vs-v3-D1 gate)

| check | value | expected |
| --- | ---:| --- |
| `wc -l src/may13/cipher_dispatch.cpp` | **543** | 543 (ROOT) |
| `wc -l src/may13/cipher_oracle.cpp` | **539** | 539 (ROOT) |
| `grep -c 'cipher_edmd_live_collect' .../cipher_dispatch.cpp` | **3** | ≥1 (ROOT uses live-EDMD) |
| `grep -c 'g_block_sub\|cipher_block_sub_collect' .../cipher_dispatch.cpp` | **0** | 0 (NOT the SRC substitute-cache pattern) |

All four signature checks pass. The v3 PARTIAL failure mode (SRC variant with block_sub refs) is structurally excluded.

### C.3 9 unpaired sources copied byte-identical from evidence src/

`cipher_recipes.cpp`, `cipher_sense.cpp`, `cipher_structural_lookup.cpp`, `cipher_kernel_table.cpp`, `cipher_runtime.cpp`, `cipher_telemetry.cpp`, `cipher_green_ctx.cu`, `cipher_l2_persist.cu`, `cipher_liquid_state.cu` — all md5-verified. `src/may13/` count = 11. `grep -rn CipherKernelEntry src/may13/` = 0 (Step 1 rename propagated).

### C.4 Bare-name include rewrites in destination sources

```
cipher_dispatch.cpp: 7 rewrites
cipher_oracle.cpp:   2 rewrites
cipher_recipes.cpp:  1
cipher_sense.cpp:    1
cipher_structural_lookup.cpp: 1
cipher_kernel_table.cpp: 1
cipher_runtime.cpp:  2
cipher_telemetry.cpp: 2
cipher_green_ctx.cu: 2
cipher_l2_persist.cu: 2
cipher_liquid_state.cu: 2
---
Total: 23
```

`grep -rnE '^\s*#\s*include\s+"cipher_' src/may13/` returns **0** matches — every in-closure include uses the `may13/` prefix.

---

## 5. Sub-step D — H1 stub created — PASS

`src/cipher_may13_stubs.cpp` (25 lines):

```cpp
#include "may13/cipher_intercept.h"
namespace { const CipherInterceptStats g_zero_intercept_stats = {0}; }
extern "C" const CipherInterceptStats* cipher_intercept_stats(void) {
    return &g_zero_intercept_stats;
}
```

Header `may13/cipher_intercept.h` is in closure (Step 2 v3 sub-step B); `CipherInterceptStats` type resolves; H1 stub matches the consumer signature at `cipher_runtime.cpp:123`.

---

## 6. Sub-step E — Makefile extension — PASS

Three edits applied:

1. **Line 6-11**: Added `NVCC ?= nvcc` and `NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_90 --compiler-options -fPIC,-Wall,-Wextra,-Wno-unused-parameter` (mirrored from `cipher-may13-evidence/Makefile`). Added `-lcudart` to `LIBS`.
2. **OBJS extension**: Added 12 new objects (`may13_cipher_*.o` × 11 + `cipher_may13_stubs.o`).
3. **Rules block**: Added 11 per-source rules (8 `.cpp` with `-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1`, 3 `.cu` with `nvcc $(NVCCFLAGS)`) + 1 stub rule.

No flag conflicts with cipher_rt_phase4's existing nvcc-less build (Marlin is `.cpp`, not `.cu`).

---

## 7. Sub-step F — Build + symbol audit — PASS

### Build

```
make clean && make
rc=0
libcipher_rt.so md5: 56291439 → bc249c9dfac9a1d5cb777d9d8a4d7964 (changed, expected)
warning count: 18 → 78  (delta +60, all cosmetic — same categorization as v3 attempt)
```

Warning breakdown (delta +60):
```
  58  [-Wmissing-field-initializers]
  14  [-Wunused-parameter]
   1  [-Wunused-variable]
   1  [-Wunused-function]
   1  [-Wstringop-truncation]   ← in /usr/include (not may13)
   1  [-Wparentheses]            ← cipher_rt_tenant.cpp pre-existing
```

No latent-bug indicators.

### Undefined-symbol audit (the v3 gate that previously failed)

```
$ nm -D --undefined-only libcipher_rt.so | filter system/libc/cuda/torch
                 U _Unwind_Resume@GCC_3.0
                 U __popcountdi2@GCC_3.4
                 U cuCtxPopCurrent_v2
                 U cuCtxPushCurrent_v2
```

**Zero non-system CIPHER undefs.** Specifically:
- No `g_block_sub` ✓ (would have indicated SRC variant)
- No `cipher_block_sub_collect` ✓ (same)
- No `cipher_intercept_stats` ✓ (H1 stub satisfies it)
- No `cipher_liquid_record_*`, no `cipher_oracle_*`, no `g_cipher` undefs

The remaining 4 undefs are routine system symbols resolved by `-lgcc_s` (`_Unwind_Resume`), libgcc intrinsics (`__popcountdi2`), and libcuda (`cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`).

---

## 8. Sub-step G — Runtime invariant verification — PASS

### G.1 Loader smoke (LD_PRELOAD /bin/true)

```
$ LD_PRELOAD=$(realpath libcipher_rt.so) /bin/true
rc=0  (no symbol-lookup-error)
```

### G.2 Shim injection smoke (SDPA cuDNN backend)

```
$ CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 -c "<sdpa test>"
[cipher_v2] GREEN/CP54: ALLOCATE ok — qos=1 sm_count=0 -> grp_mask=0x0000 grp_count=0
[cipher_v2] SMP: sm_packer initialized (...)
[cipher_v2] PR: cipher_rt_pr_init entered
[cipher_v2] PR: partition router initialized (cache=256 slots, ...)
[cipher_v2] CUPTI subscribed: kernel launch callbacks active (...)
[cipher_v2] MATMUL: substrate initialized (max 16 actuators; first registration awaited)
[cipher_v2] MARLIN: actuator DISABLED (CIPHER_MARLIN not set)
[cipher-attn] substrate active (torch verified 2.11.0+cu130); symbol resolution is lazy per-call
[cipher_v2] GOT: patch applied — 8 slot(s) across 71 module(s); 4 target(s) registered
[cipher_v2] GREEN: CP 5.4 qos=shared — tenant owns no SM groups; ...

SDPA mean: -0.00408172607421875
finite: True

[cipher_v2] MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=0)
[cipher-attn] exit totals - tramp_calls=1 tramp_fake=0 observed=1
              handled=0 passthrough=1 redirected=0 (flash=0 eff=0 cudnn=1)

Python rc=0
```

Banner count: **16** cipher banners. Trampoline accounting confirms LP-2 invariant holds at runtime: `tramp_calls=1` (cuDNN backend used), `handled=0` (no actuator returned HANDLED — defensive abort() guard never fired), `passthrough=1` (orig() called as designed).

### G.3 Static-init verification

```
$ nm libcipher_rt.so | grep -iE 'cipher_so_init|cipher_hook_init|cipher_intercept_init$'
(none)
```

The hostile static-init markers from the v3 pre-flight FLAG files (`cipher_intercept.cpp::cipher_so_init`, `cipher_intercept_cudart.cpp::cipher_hook_init`) are absent — confirms the A4 exclusion landed structurally, not just in source-file choice.

---

## 9. Sub-step H — CP 5.4 regression — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
rc=0
```

All 15 kmod-substrate tests pass: legacy nr-9 deactivated, ALLOCATE / FREE / QUERY, pool resize, do_exit reaper, disjointness, concurrent stress. Substrate behavior unchanged vs Week 1 closeout baseline.

---

## 10. Sub-step I — Commit + tag — PASS

```
git diff --cached --stat:
 19 files changed, 4857 insertions(+), 2 deletions(-)

git rev-parse HEAD:                                23014c1e4673a3a73974bbf7173153050f8bb0c8
Tag week-2-step-2-may13-ports-a4-h1-root:          23014c1e ✓
Tag week-2-step-1-lp2-sdpa-refactor (Step 1):      465b244e ✓
Working tree pending: 0
```

### File diff summary

```
 Makefile                               |   59 +-
 include/may13/cipher.h                 |   92 +
 include/may13/cipher_edmd.h            |  228 +
 include/may13/cipher_green_ctx.h       |  132 +
 include/may13/cipher_intercept.h       |   92 +
 include/may13/cipher_l2_persist.h      |   99 +
 include/may13/cipher_telemetry.h       |  115 +
 src/cipher_may13_stubs.cpp             |   25 +
 src/may13/cipher_dispatch.cpp          |  543 +
 src/may13/cipher_green_ctx.cu          |  333 +
 src/may13/cipher_kernel_table.cpp      |  317 +
 src/may13/cipher_l2_persist.cu         |  238 +
 src/may13/cipher_liquid_state.cu       |  330 +
 src/may13/cipher_oracle.cpp            |  539 +
 src/may13/cipher_recipes.cpp           |  522 +
 src/may13/cipher_runtime.cpp           |  145 +
 src/may13/cipher_sense.cpp             |  340 +
 src/may13/cipher_structural_lookup.cpp |  300 +
 src/may13/cipher_telemetry.cpp         |  410 +
                                     ────────
                                       4857 lines added
```

The 543-line `cipher_dispatch.cpp` and 539-line `cipher_oracle.cpp` are the ROOT variants (confirmed by md5 in §4).

---

## 11. Discipline notes

- Single commit. Tag chain: `week-2-step-1-lp2-sdpa-refactor` → `week-2-step-2-may13-ports-a4-h1-root`.
- Rollback: `git reset --hard week-2-step-1-lp2-sdpa-refactor`
- Substrate behavior unchanged. No may13 code is reached by existing cipher_rt_phase4 dispatch paths. Step 6 (CLASSIFY+SENSE+ORACLE wiring) will route through them via the classifier substrate.
- The H1 stub (5 actual lines + comments) is the only boundary between cipher_rt_phase4 and may13. No silent runtime behaviors masked.
- The v3-PARTIAL audit-foundation hole is closed: the symbol-closure tool's input (prebuilt .o = ROOT) and the port source (now ROOT) are now consistent.

---

## 12. What this unlocks

- **Week 2 Step 3** — `cipher_rt_classify_substrate.cpp` scaffolding (priority-ordered classifier-actuator registry, twin of matmul + attn substrates). Now has a working may13 surface to call into.
- **Week 2 Step 4** — `cipher_rt_classify_observer.c` skeleton.
- **Week 2 Step 5** — `/proc/cipher/classify_stats` kmod proc node.
- **Week 2 Step 6** — Hot-path `CLASSIFY+SENSE+ORACLE` wiring (the structural payoff: actually call the ported may13 code from cipher_rt_phase4's dispatch paths).
- **Week 2 Step 7** — closeout.

---

## 13. Anchors at close

```
cipher-fusion-evidence: (to be set after this doc commits)
cipher_rt_phase4:        23014c1e4673a3a73974bbf7173153050f8bb0c8  (tag: week-2-step-2-may13-ports-a4-h1-root)
cipher_kmod:             f8572ecf422050a488f1fb45f74534ecd1bde678  (Week 1 close, unchanged)
cipher-may13-evidence:   fc8a9ae6025cff345e86d054b1df02d07a9fb1fe  (Week 1 close, unchanged)
```

Week 2 progress: 2/7 steps complete (LP-2 SDPA refactor + may13 ports). Awaiting Step 3 brief.
