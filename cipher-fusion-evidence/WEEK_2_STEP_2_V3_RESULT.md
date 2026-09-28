# Week 2 Step 2 v3 — A4+H1 Port — RESULT

**Status: PARTIAL — STOPPED at sub-step F. Root-cause structural finding surfaces an unresolved input-selection ambiguity that invalidates the prior 3-round audit chain. Awaiting adjudication.**

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 2 v3 (third attempt at brief's Scope-6 port)
**Tree state at stop:**
  - `cipher_rt_phase4` HEAD: `465b244e` (`week-2-step-1-lp2-sdpa-refactor`, unchanged)
  - Working tree: dirty (12 new files in src/may13/ + Makefile changes); md5 `8cbd2f71800b897bc64d9c0502224ed2` on the broken .so
  - No commit, no tag

---

## 1. Headline

Sub-steps A.0 → E executed clean. Sub-step F surfaced **2 strong undefined symbols** in the freshly-built libcipher_rt.so:

```
U cipher_block_sub_collect
U g_block_sub
```

Both are referenced by the freshly-ported `cipher_dispatch.cpp` (L241, L246, L258, L259, L260 — 15 occurrences total). Both are defined in `cipher_edmd.cpp`, which is **not in the A4 11-file closure**.

The brief's stop discipline is unambiguous: "If any CIPHER symbol undefined: STOP. The A4 closure missed something. Surface."

Tracing the root cause uncovered a structural ambiguity in cipher-may13-evidence that invalidates the symbol-closure foundation of the past three rounds of audit:

**There are TWO distinct `cipher_dispatch.cpp` files in cipher-may13-evidence.**

| location | size | block_sub refs | canonical? |
| --- | ---:| ---:| --- |
| `cipher-may13-evidence/cipher_dispatch.cpp` (root) | 543 lines | **0** | **YES — used by evidence Makefile** |
| `cipher-may13-evidence/src/cipher_dispatch.cpp` (subdir) | 616 lines | 15 | NO — explicitly filtered by Makefile |

The evidence Makefile lines 65–66 do this exclusion explicitly:

```make
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp \
                            src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp \
                            src/cipher_oracle.cpp, $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp
```

The same dual-file pattern exists for `cipher_oracle.cpp` (root 539 lines, src/ 451 lines — same direction, root is canonical).

**The shipped `cipher-may13-evidence/libcipher_rt.so` is built from the root versions.** The prebuilt `cipher-may13-evidence/build/cipher_dispatch.o` (which all my symbol_closure / closure-walker analyses consumed) was compiled from the **root** canonical version. But my Step 2 v3 *port* copied the **src/** fork (616 lines, with block_sub refs).

The discrepancy explains the build failure: the closure walker's view of cipher_dispatch.cpp's external surface (from the root-built .o, no block_sub refs) does not match the source code I ported (src/ version, 15 block_sub refs).

Per discipline: STOP, document, surface. No mitigation proposed.

---

## 2. Sub-step A.0 — Pre-flight baseline reproduction — PASS

```
HEAD:                465b244e (week-2-step-1-lp2-sdpa-refactor)
include/may13/ count: 12 (post v2 residue cleanup)
make clean && make:   rc=0
libcipher_rt.so md5:  56291439c7fd941ca37cf86c482948d7  ✓ baseline reproduces
warning count:        18 (baseline)
```

v2 residue (6 new headers + src/may13/ + Makefile changes) cleanly removed; baseline reproduces byte-identical.

---

## 3. Sub-step A — Header closure walk — PASS

Closure walker (`/tmp/v3_header_closure.py`) walked transitive `#include` graph from 11 port sources + the H1 stub's `cipher_intercept.h` requirement.

Result: **18-header closure** (12 existing + 6 new). Identical to v2's set. A4 did not drop any headers — the dropped .cpp files (`cipher_intercept*.cpp`) still required their headers to be visible for type declarations.

New headers added in this step: `cipher.h`, `cipher_edmd.h`, `cipher_intercept.h`, `cipher_green_ctx.h`, `cipher_l2_persist.h`, `cipher_telemetry.h`. (Walker initially listed 5; `cipher_intercept.h` was the stub-driven seed and required manual surfacing.)

UNRESOLVED: NONE.

---

## 4. Sub-step B — Header port + library invariant — PASS

| signal | value |
| --- | ---:|
| 6 new headers ported md5-verified | ✓ |
| 18-header closure verified | ✓ |
| 15 bare-name `#include` rewrites applied (cipher.h: 9, cipher_edmd.h: 1, cipher_intercept.h: 1, cipher_green_ctx.h: 1, cipher_l2_persist.h: 1, cipher_telemetry.h: 2) | ✓ |
| Zero bare-name `cipher_*` includes remaining | ✓ |
| Library invariant build | `rc=0, md5=56291439, warnings=18` ✓ HELD |
| Synthetic 18-header compile smoke | `rc=0, errors=0, warnings=0` ✓ |

---

## 5. Sub-step C — Port 11 sources — PASS (but from wrong source variant)

11 files copied byte-identical (`cp -p`) from `cipher-may13-evidence/src/` to `cipher_rt_phase4/src/may13/`. md5s match source. 23 bare-name include rewrites applied. Zero `CipherKernelEntry` refs (Step 1 rename propagated).

**This is the step that introduced the structural ambiguity.** I copied from `cipher-may13-evidence/src/cipher_dispatch.cpp` (616 lines, 15 block_sub refs), but the canonical evidence build uses `cipher-may13-evidence/cipher_dispatch.cpp` (543 lines, 0 block_sub refs).

I did not realize there were two versions until the build failure in §6 forced the investigation.

---

## 6. Sub-step D — H1 stub — PASS

`src/cipher_may13_stubs.cpp` created (24 lines, `cipher_intercept_stats()` returns pointer to a static zero-init `CipherInterceptStats`). Verified include of `may13/cipher_intercept.h` resolves; struct type matches the declaration at L88 of the ported header.

---

## 7. Sub-step E — Makefile extension — PASS

Three edits applied:

1. Added `NVCC ?= nvcc` and `NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_90 --compiler-options -fPIC,-Wall,-Wextra,-Wno-unused-parameter` (mirrored from `cipher-may13-evidence/Makefile` lines 10, 15). Also added `-lcudart` to `LIBS`.
2. Added 12 new objects to `OBJS`: 11 `may13_cipher_*.o` + 1 `cipher_may13_stubs.o`.
3. Added 12 build rules: 8 `.cpp` (g++ with `-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1`), 3 `.cu` (nvcc with may13 flags), 1 stub.

No flag conflicts surfaced — cipher_rt_phase4 had no prior nvcc rules (Marlin is `.cpp`), so the new nvcc invocation is additive.

---

## 8. Sub-step F — Build + symbol verification — **FAIL**

### Build

```
rc=0
libcipher_rt.so md5: 56291439 → 8cbd2f71800b897bc64d9c0502224ed2
warning count: 18 → 78  (delta +60)
```

Build succeeded (rc=0) and produced a binary, but the `-shared` linker did NOT enforce strong-undef resolution.

### Warning categorization (delta +60)

```
  58  [-Wmissing-field-initializers]   — research-grade brain code style
  14  [-Wunused-parameter]
   1  [-Wunused-variable]
   1  [-Wunused-function]
   1  [-Wstringop-truncation]
   1  [-Wparentheses]
```

All cosmetic. No latent-bug indicators.

### .o sizes

```
22880  may13_cipher_dispatch.o
18976  may13_cipher_kernel_table.o
13488  may13_cipher_green_ctx.o
13280  may13_cipher_recipes.o
12008  may13_cipher_telemetry.o
11848  may13_cipher_liquid_state.o
10616  may13_cipher_l2_persist.o
10120  may13_cipher_oracle.o
 7960  may13_cipher_sense.o
 7144  may13_cipher_runtime.o
 7112  may13_cipher_structural_lookup.o
 1576  cipher_may13_stubs.o
```

### Undefined-symbol audit — FAIL gate

```
$ nm -D --undefined-only libcipher_rt.so | filter system/libc/cuda
                 U cipher_block_sub_collect        ← STRONG CIPHER undef
                 U g_block_sub                     ← STRONG CIPHER undef
                 w cipher_get_edmd_pipeline        (weak, OK)
                 w cipher_tls_get_gemm_ptrs        (weak, OK)
                 w cipher_tls_get_gemm_shape       (weak, OK)
                 w cipher_tls_relaunch             (weak, OK)
```

Two strong CIPHER undefs. Gate failure per brief: "If any CIPHER symbol undefined: STOP. The A4 closure missed something. Surface."

### Cause trace

- `g_block_sub` is referenced at L241, L258, L259, L260 of the ported `cipher_dispatch.cpp` (from `cipher-may13-evidence/src/cipher_dispatch.cpp`, 616-line variant)
- `cipher_block_sub_collect()` is referenced at L246 of the same file
- Both are defined in `cipher_edmd.cpp`, which is NOT in the A4 11-file closure
- The symbol-closure walker (`/tmp/symbol_closure.py`) did NOT report these as undefs in `cipher_dispatch.cpp` because the walker reads the **prebuilt .o file** at `cipher-may13-evidence/build/cipher_dispatch.o`, which was compiled from the OTHER `cipher_dispatch.cpp` (the canonical root variant, 543 lines, **zero** block_sub refs)

---

## 9. Root cause — Two variants of cipher_dispatch.cpp + cipher_oracle.cpp in evidence

### File inventory

```
cipher-may13-evidence/cipher_dispatch.cpp        543 lines  md5 59271a09...  0 block_sub refs
cipher-may13-evidence/src/cipher_dispatch.cpp    616 lines  md5 59d80530...  15 block_sub refs

cipher-may13-evidence/cipher_oracle.cpp          539 lines  md5 3b2c0d68...
cipher-may13-evidence/src/cipher_oracle.cpp      451 lines  md5 1b5a175c...
```

Both pairs are at the same mtime (`May 13 04:27`) but differ in size and content.

### What the evidence Makefile does (lines 65–66, verified)

```make
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp \
                            src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp \
                            src/cipher_oracle.cpp, $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp
```

Result: the canonical evidence build excludes both `src/cipher_dispatch.cpp` and `src/cipher_oracle.cpp`, and uses the ROOT-level variants instead. The prebuilt `build/cipher_dispatch.o` (no `src__` prefix in name — visible signal!) was compiled from the root version.

### What the shipped `cipher-may13-evidence/libcipher_rt.so` actually has

```
$ nm cipher-may13-evidence/libcipher_rt.so | grep -E 'block_sub_collect|cipher_dispatch '
00000000000018e90 T cipher_block_sub_collect    ← defined (from cipher_edmd.cpp)
0000000000045490 T cipher_dispatch              ← defined (from ROOT cipher_dispatch.cpp)
```

Both are defined and exported. The shipped .so includes both variants' contributions because the closure-builder pulled in `cipher_edmd.cpp` transitively from some OTHER caller (not the root cipher_dispatch.cpp, which doesn't call block_sub). The 73-source full-tree build resolves everything.

### Cascading impact on prior audits

The structural ambiguity invalidates the foundation of three rounds of audit:

| audit | what it claimed | reality |
| --- | --- | --- |
| Symbol-closure (`60078d99`) | Scope-6 has 40-file closure | walker used **prebuilt .o = root variant**, but planned port copied **src/ variant** → undef set differs by `g_block_sub` + `cipher_block_sub_collect` |
| Static-init pre-flight (`e6afb4b2`) | 2 FLAG files (cipher_intercept*) | the variant choice has no impact on static-init analysis (intercept files are not duplicated) — **this finding is still valid** |
| A4 closure rerun (`be484e63`) | 11-file closure + 1 orphan | uses prebuilt .o = root variant; the 11-file scope is correct IF we port the root variant; with the src/ variant, the actual closure is larger (must include cipher_edmd.cpp + cipher_block_sub_kernel.cu + their transitive surface) |
| Orphan inspection (`67f68ad5`) | H1 PURE-LOG verdict | cipher_runtime.cpp:123 inspection used `cipher-may13-evidence/src/cipher_runtime.cpp` — but `cipher_runtime.cpp` does NOT have a root variant; this finding is **still valid** |

### The unresolved question

The question that needs adjudication: which variant is canonical for the Week 2 port?

- The **root variant** is what the evidence Makefile builds. It is the canonical source of the shipped `libcipher_rt.so`. It has zero block_sub refs, so the 11-file A4 closure works as planned.
- The **src/ variant** is longer (616 vs 543), contains additional functionality (block_sub references → EDMD Koopman online learning), and is what I picked because the existing `WEEK_2_SCOPE_LOCK.md` and the v1 closure walker said "the files are in src/".

Both files appear to have the same mtime, suggesting they were checked into the evidence tree at the same time as deliberate variants, not a stale-fork accident.

---

## 10. Sub-steps G, H, I — NOT EXECUTED

Per brief STOP discipline at the sub-step F undef-symbol gate. Sub-steps G (runtime smoke + static-init verification), H (CP 5.4 regression), I (commit + tag) not executed.

`week-2-step-2-may13-ports-a4-h1` tag **not created**. cipher_rt_phase4 HEAD unchanged at `465b244e`.

---

## 11. Rollback path

Working tree is currently dirty with the v3-from-src/ attempt:

```
cd /home/ubuntu/cipher_rt_phase4
rm -rf include/may13/cipher.h include/may13/cipher_edmd.h \
       include/may13/cipher_intercept.h include/may13/cipher_green_ctx.h \
       include/may13/cipher_l2_persist.h include/may13/cipher_telemetry.h
rm -rf src/may13/
rm -f src/cipher_may13_stubs.cpp
rm -f may13_cipher_*.o cipher_may13_stubs.o
git checkout -- Makefile
make clean && make
# verify md5 returns to 56291439
```

The dirty state is retained on-disk so the user can inspect the v3 artifacts during adjudication.

---

## 12. Telemetry on disk

- `/tmp/week2_step2_v3/pre_baseline_build.log` — baseline reproduced 56291439
- `/tmp/week2_step2_v3/post_b_build.log` — library invariant after header port (md5 unchanged)
- `/tmp/week2_step2_v3/header_smoke.log` — synthetic 18-header smoke (rc=0)
- `/tmp/week2_step2_v3/post_f_build.log` — broken build with src/ variant (rc=0, but 2 strong undefs)
- `/tmp/week2_step2_v3/undefined.txt` — full nm undef dump from broken .so
- `/tmp/week2_step2_v3/header_closure.txt` — header closure walk output
- `/tmp/v3_header_closure.py`, `/tmp/v3_apply_rewrites.py`, `/tmp/v3_rewrite_sources.py` — tooling
- `/tmp/test_root_dispatch.o`, `/tmp/test_root_oracle.o` — fresh compiles of ROOT variants, used to verify the variant discrepancy

The freshly-built ROOT-variant .o files (`test_root_*.o`) confirm: their undef sets do NOT contain `g_block_sub` or `cipher_block_sub_collect`. If we had ported from the root variant, the build would have succeeded at sub-step F.

---

## 13. Discipline notes

- STOP discipline honored. No commit, no tag, no anchor rotation.
- The Step 1 LP-2 anchor (`week-2-step-1-lp2-sdpa-refactor` at `465b244e`) is the in-place baseline.
- No mitigation proposed in this document. The structural finding (two-variant ambiguity) requires user adjudication before any direction can be taken.
- A4 closure rerun and v3 pre-flight findings about the GOT-patcher conflict (`cipher_intercept.cpp` + `cipher_intercept_cudart.cpp` FLAGs) remain valid independent of the variant choice — those files are not duplicated.
- The audit-foundation invalidation does NOT apply to Step 1 LP-2 work, which was substrate-internal to cipher_rt_phase4 and does not depend on cipher-may13-evidence variant choice.

---

## 14. Awaiting

User adjudication. Possible structural directions (no recommendation):

- **D1 — Port from ROOT variants** (`cipher-may13-evidence/cipher_dispatch.cpp` 543L + `cipher-may13-evidence/cipher_oracle.cpp` 539L); re-run the prior 4-round audit chain against the root variants to re-confirm the 11-file A4 closure; if confirmed, retry sub-step C with the right sources.
- **D2 — Port from SRC variants** (current attempt); accept the larger closure (must add cipher_edmd.cpp + cipher_block_sub_kernel.cu + their transitive deps); re-walk the symbol closure against the src/ variants' actual symbol surface.
- **D3 — Investigate why two variants exist** before deciding: were they deliberately forked (e.g., one is a research-grade extension), or is one of them obsolete? May need an external review of the cipher-may13-evidence tree's history.
- **D4 — Other.**

The Week 2 Step 2 v3 port cannot proceed without this adjudication. All four prior audit-document SHAs (`60078d99`, `e6afb4b2`, `be484e63`, `67f68ad5`) reference data derived from the prebuilt .o files (root-variant compilation) but my port sourced the src/ variant — so the audit chain is internally inconsistent. Re-confirming or re-running it with the chosen variant is the natural unblock.
