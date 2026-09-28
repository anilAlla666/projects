# Week 2 Step 2 — Six may13 .cpp source ports — RESULT

**Status: PARTIAL — STOPPED at Step B per brief discipline. Surfaced for adjudication.**

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 2 of 7
**Tree touched (read-only, no commit yet):** `cipher_rt_phase4` (Tree B1)
**Anchors at stop point:**
  - `cipher_rt_phase4` HEAD: `465b244e` (`week-2-step-1-lp2-sdpa-refactor`, unchanged)
  - libcipher_rt.so md5: `56291439c7fd941ca37cf86c482948d7` (unchanged)
  - `cipher-may13-evidence` HEAD: `fc8a9ae6` (unchanged)
  - `cipher-fusion-evidence` HEAD: pending this doc

---

## 1. Headline

Step B (transitive closure walk) **surfaced 6 new may13 headers** required to compile the source set. Per brief: *"If any new may13 header surfaces: STOP, do not port .cpp files. Surface for adjudication."* — STOPPED before Steps C-G.

**Critical observation:** only **1 of 6** .cpp files pulls in the new headers (`cipher_dispatch.cpp` includes `cipher.h` and `cipher_edmd.h`). The other 5 .cpp files resolve cleanly through the existing 12-header `include/may13/` closure from Step 2 v2.

This gives the user three structural options (presented at the end, no recommendation per discipline).

---

## 2. Step A — Pre-edit verification — PASS

| signal | value | expected | match |
| --- | --- | --- | --- |
| `cipher_rt_phase4` HEAD | `465b244e` | `465b244e` | ✓ |
| libcipher_rt.so md5 | `56291439c7fd941ca37cf86c482948d7` | same | ✓ |
| working tree | clean | clean | ✓ |
| `cipher-may13-evidence` HEAD | `fc8a9ae6` | `fc8a9ae6` | ✓ |
| `include/may13/` count | 12 headers | 12 (Step 2 v2 baseline) | ✓ |
| 6 source .cpp files present | yes (md5s recorded) | yes | ✓ |
| Pre-edit `make clean && make` | rc=0, md5 matches, 18 warnings | rc=0, md5, ≤18 | ✓ |

Baseline preserved at `/tmp/week2_step2/libcipher_rt.so.prestep2` (md5 `56291439`), nm dump at `/tmp/week2_step2/nm_prestep2.txt` (186 dyn symbols).

### Source set md5s (record)

| file | lines | md5 |
| --- | --- | --- |
| `cipher_dispatch.cpp` | 616 | `59d805306298aa873e0f73cdbf0a9321` |
| `cipher_oracle.cpp` | 451 | `1b5a175c02ae7d8de16f740253d9ed92` |
| `cipher_recipes.cpp` | 522 | `0dd1e9744aab4596885427909362a0e3` |
| `cipher_sense.cpp` | 340 | `656780604038845feed6bc10ac7d3c4e` |
| `cipher_structural_lookup.cpp` | 300 | `b57aee5c02a0c4919e96469f860622ae` |
| `cipher_kernel_table.cpp` | 317 | `bacbdc432f5babe3c01423d31cf087f4` |

---

## 3. Step B — Transitive closure walk — STOPPED (surface)

### 3.1 Per-file summary (direct `#include` classification)

```
file                              tot  clo   p4  new  src  sys  unr
cipher_dispatch.cpp                11    5    0    2    0    4    0
cipher_oracle.cpp                   6    2    0    0    0    4    0
cipher_recipes.cpp                  5    1    0    0    0    4    0
cipher_sense.cpp                    8    1    0    0    0    7    0
cipher_structural_lookup.cpp        4    1    0    0    0    3    0
cipher_kernel_table.cpp             7    1    0    0    0    6    0
```

Legend: `tot`=total `#include`s · `clo`=resolves through may13 closure · `p4`=resolves through cipher_rt_phase4 native · `new`=new may13 header (in evidence/include/ not in closure) · `src`=header in evidence/src/ · `sys`=system/angle-bracketed · `unr`=UNRESOLVED.

**5 of 6 .cpp files clean.** Only `cipher_dispatch.cpp` introduces 2 new may13 headers directly.

### 3.2 Full transitive BFS (closure walk through all reachable headers)

The BFS recurses into each new header; here is the full set of headers **NOT in the current 12-header closure** but reachable from the 6 .cpp files:

| new header | lines | pulled by | role (per header comment) |
| --- | --- | --- | --- |
| `cipher.h` | 92 | `cipher_dispatch.cpp` | master include — pulls F1..F5 component headers |
| `cipher_edmd.h` | 228 | `cipher_dispatch.cpp` | L3.5 EDMD pipeline (Koopman derivation for novel ops) |
| `cipher_intercept.h` | 92 | `cipher.h` (transitive) | F1: cuLaunchKernel intercept layer |
| `cipher_green_ctx.h` | 132 | `cipher.h` (transitive) | F2: green-context allocation (8-SM carve) |
| `cipher_l2_persist.h` | 99 | `cipher.h` (transitive) | F3: L2 persistent weight loading |
| `cipher_telemetry.h` | 115 | `cipher.h` (transitive) | F5: hardware telemetry pipeline (CUPTI) |

Total new lines: **758** (would go into `include/may13/` if extended).

### 3.3 Resolution check

- UNRESOLVED: **NONE** — every directly- and transitively-included quoted header either is already in the may13 closure (12), is a cipher_rt_phase4 native file, or is in the new-headers set above.
- Headers in `cipher-may13-evidence/src/` (non-public): **none** — every `#include` resolves through `evidence/include/` or system.
- System headers: **28 distinct** angle-bracketed includes across the 6 .cpp files (standard library, CUDA, no Torch).

### 3.4 File-level name collisions vs `cipher_rt_phase4`

Checked each new header against `cipher_rt_phase4` root and `include/` (excluding `include/may13/`):

```
clean : cipher.h               (no name collision at file level)
clean : cipher_edmd.h          (no name collision at file level)
clean : cipher_green_ctx.h     (no name collision at file level)
clean : cipher_intercept.h     (no name collision at file level)
clean : cipher_l2_persist.h    (no name collision at file level)
clean : cipher_telemetry.h     (no name collision at file level)
```

**Zero file-level collisions** — all `cipher_rt_phase4` native files in this concept-space are `cipher_rt_*`-prefixed. The new headers would land cleanly as `include/may13/<name>`.

### 3.5 Conceptual-layer overlap (informational, not a collision)

Several new headers cover concept-space that `cipher_rt_phase4` has its own production implementation for. Documenting for adjudicator visibility — these are not file-level collisions, but they are *role* overlaps:

| may13 header (thinking layer) | cipher_rt_phase4 production analog |
| --- | --- |
| `cipher_intercept.h` (F1 hook) | `cipher_rt_got_patch.c` (GOT/PLT patcher) |
| `cipher_green_ctx.h` (F2 carve) | `cipher_rt_green_ctx.c` (CP 5.4 group ledger) |
| `cipher_l2_persist.h` (F3 pin) | CP 4.4 closed at memo via L2 budget bound (118-524×); not built |
| `cipher_telemetry.h` (F5 CUPTI) | `cipher_cupti.c` (CUPTI launch counter) |

If the may13 closure is extended, both sets co-exist in the library (the may13 versions arrive as compile-only declarations, no implementations linked unless their companion .cpp is also ported — and there is no `cipher_intercept.cpp` / `cipher_green_ctx.cpp` in the 6-file set, only the 6 listed). Whether this dual-presence is fine or surprising is the adjudication question.

---

## 4. Why this is a Step B STOP per brief

Direct quote from brief §STEP B:

> *If X exists in cipher-may13-evidence/include/ but is NOT in existing closure: this is a NEW transitive header. STOP and surface. Do not auto-add.*

> *If any new may13 header surfaces: STOP, do not port .cpp files. Surface for adjudication (would require Step 2 v2-style closure walk extension first).*

Two new headers surfaced directly (`cipher.h`, `cipher_edmd.h`); BFS expanded the set to six. STOP-and-surface invoked. No `.cpp` was copied; no Makefile change; no commit on `cipher_rt_phase4`.

---

## 5. Steps C-G — NOT EXECUTED

| step | status | reason |
| --- | --- | --- |
| C — copy 6 .cpp | not executed | gated on Step B clean |
| D — bare-name rewrites | not executed | gated on Step C |
| E — Makefile + build | not executed | gated on Step D |
| F — CP 5.4 regression | not executed | gated on Step E |
| G — commit + tag + result doc | partial (this doc only; no git changes) | gated on F |

Tag `week-2-step-2-may13-cpp-ports` **not created**. Tree anchors unchanged.

---

## 6. Adjudication options (for user — no recommendation per discipline)

The closure walk gives a clean factual basis for three structural directions. The brief said "would require Step 2 v2-style closure walk extension first" which hints at (a); user may prefer another.

### Option (a) — Extend the closure (Step 2 v2 redux)

Port the 6 new headers into `include/may13/`, applying bare-name → `may13/` rewrites the same way Step 2 v2 did for the original 12. Then resume Step 2 normally — port all 6 .cpp.

- closure size: 12 → 18 headers
- new lines under may13/: +758 (headers only)
- rewrites required: each new header's quoted `#include`s scanned and rewritten (Step 2 v2 pattern — 9 rewrites in cipher.h, 1 in cipher_edmd.h, others TBD)
- after: all 6 .cpp files compile as inert TUs

### Option (b) — Port only the 5 clean .cpp files; defer `cipher_dispatch.cpp`

The 5 "clean" .cpp files (`cipher_oracle.cpp`, `cipher_recipes.cpp`, `cipher_sense.cpp`, `cipher_structural_lookup.cpp`, `cipher_kernel_table.cpp`) compile through the existing 12-header closure. `cipher_dispatch.cpp` is deferred until the closure is extended (later step or later week).

- closure size: 12 (unchanged)
- new lines under src/may13/: +1930 (5 .cpp totaling 1930 lines; cipher_dispatch.cpp's 616 deferred)
- after: 5 of 6 .cpp linked as inert TUs; dispatch path stays as cipher_rt_phase4 native until a follow-on step

  - **risk to flag:** the 5 .cpp files may declare symbols whose definitions live in cipher_dispatch.cpp. Closure walker did not run the linker; would surface at Step E build. May produce "undefined reference" requiring weak-stub workaround or partial revert. Pre-flight estimate would need a forward-declaration scan, which is itself a STOP-worthy artifact.

### Option (c) — Different scope-lock

Whatever the user wants — could be: extend WEEK_2_SCOPE_LOCK.md to include the 6 header port as a new Step 2.5; or move dispatch port to Week 3; or fold the new headers into a separate "thinking-layer" subdirectory rather than `include/may13/`; or anything else.

---

## 7. Telemetry / artifacts on disk

- `/tmp/week2_step2/libcipher_rt.so.prestep2` — baseline .so (md5 56291439)
- `/tmp/week2_step2/nm_prestep2.txt` — baseline dynamic-symbol dump
- `/tmp/week2_step2/closure_walk.txt` — direct-include walk output (Step B.1)
- `/tmp/week2_step2/closure_walk_full.txt` — full transitive BFS output (Step B.2)
- `/tmp/closure_walker_cpp.py` — direct-walk script
- `/tmp/closure_walker_full.py` — BFS script

All artifacts ephemeral; do not block rollback.

---

## 8. Discipline notes

- No mitigation in this document — only data + structural options for adjudicator.
- Working tree clean. No git changes. No tag. No anchor rotation.
- Brief's STOP discipline honored: surfacing happens at Step B, before any code is copied or any Makefile is touched.
- Step 1 of Week 2 (`week-2-step-1-lp2-sdpa-refactor`) remains the in-place anchor.

---

## 9. Awaiting

User adjudication on which option to take. After adjudication, Step 2 will resume from Step B's terminal point with the chosen scope.
