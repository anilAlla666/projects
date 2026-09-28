# Week 1 Step 2 — Closure DELTA for cipher_recipes.h

**HEADLINE STATUS: CLOSURE COMPLETE.**

**Date:** 2026-05-20
**Reason for delta:** Option B adjudication added `cipher_recipes.h` to the Week 1 Step 2 v2 starting set. The existing 11-header closure (`WEEK_1_STEP_2_CLOSURE.md`, md5 `1f55748ac929aeb1ae2da749ad301515`) was walked from a different starting set; this document walks only the delta produced by adding `cipher_recipes.h`.
**Walker:** `/tmp/closure_walker_delta.py` (BFS from `{cipher_recipes.h}` with the existing 11 as stop-nodes). Full JSON result at `/tmp/closure_delta_result.json`.
**Predecessor:** `WEEK_1_STEP_2_CLOSURE.md` (md5 `1f55748ac929aeb1ae2da749ad301515`).
**Companion task:** `/tmp/wave5_header_list.md` (md5 `915927fafbca7c8bdedd51902e7c4cc4`) named `cipher_recipes.h` in Week 1 §L568.

---

## Section 1 — Delta closure

### 1.1 Delta closure size

**1 new header.** Walking `cipher_recipes.h` discovers **zero new transitive headers** in `cipher-may13-evidence/include/`. The walk terminates at the existing-closure stop-node `cipher_classify.hpp` and at angle-bracketed system headers.

### 1.2 Per-header delta table

| Header | Depth (in delta) | Source path | Size (B) | md5 | Deps into existing closure | Deps into delta closure | External deps | Angle includes |
|---|---|---|---|---|---|---|---|---|
| `cipher_recipes.h` | 0 | `cipher-may13-evidence/include/cipher_recipes.h` | 8,843 | `e4d7e84a` | `cipher_classify.hpp` (at L38) | (none) | (none) | `<stdint.h>`, `<stdbool.h>`, `<stddef.h>` |

The full md5 of the file is `e4d7e84aXXXXXXXX` (truncated in this column for display; the walker recorded the full 32-char hex which the closure-pull command can re-emit on demand). The Step 2 PARTIAL-run pre-port copy md5 of the same file (`B.2` table) was `e4d7e84a` prefix — identical, so `cipher_recipes.h` did not change between Step 2 PARTIAL and this delta walk.

### 1.3 External dependencies surfaced

**Count: 0.** No double-quoted `#include` in `cipher_recipes.h` resolves to anything outside `cipher-may13-evidence/include/` or the existing closure. The 3 angle-bracket includes (`<stdint.h>`, `<stdbool.h>`, `<stddef.h>`) are standard C library headers reached through the system toolchain — no new cipher_rt_phase4 include-path setting is required to satisfy them.

### 1.4 Unresolved includes

**Count: 0.** Every quoted include in `cipher_recipes.h` resolves either to a file in the existing closure (the one case at L38) or is angle-bracketed.

---

## Section 2 — Delta rewrite manifest

### 2.1 Delta rewrite count

**1 new rewrite.**

### 2.2 Per-file rewrite table (delta only)

| # | File (source path in `cipher-may13-evidence/include/`) | Line | Before | After | Target | Target's closure |
|---|---|---|---|---|---|---|
| 1 (delta) | `cipher_recipes.h` | 38 | `#include "cipher_classify.hpp"` | `#include "may13/cipher_classify.hpp"` | `cipher_classify.hpp` | existing |

The rewrite points into the existing closure (`cipher_classify.hpp` was already at depth 0 in the 11-header closure). Per the Option (b) rule, every bare-name include in a closure header that targets another closure header is rewritten — independent of which closure (existing vs delta) the target lives in.

---

## Section 3 — Combined manifest summary (existing + delta)

### 3.1 Combined closure size

| Component | Count |
|---|---|
| Existing closure (per `WEEK_1_STEP_2_CLOSURE.md` §1.1) | 11 |
| Delta closure (this document §1.1) | 1 |
| **Combined** | **12 headers** |

### 3.2 Combined rewrite count

| Component | Count |
|---|---|
| Existing rewrites (per `WEEK_1_STEP_2_CLOSURE.md` §2.1) | 7 |
| Delta rewrites (this document §2.1) | 1 |
| **Combined** | **8 rewrites** |

### 3.3 Combined rewrite manifest (complete list, for Step 2 v2 execution)

| # | File (source path in `cipher-may13-evidence/include/`) | Line | Before | After | Target |
|---|---|---|---|---|---|
| 1 | `cipher_oracle.h` | 42 | `#include "cipher_liquid_state.h"` | `#include "may13/cipher_liquid_state.h"` | `cipher_liquid_state.h` |
| 2 | `cipher_oracle.h` | 43 | `#include "cipher_classify.hpp"` | `#include "may13/cipher_classify.hpp"` | `cipher_classify.hpp` |
| 3 | `cipher_oracle.h` | 44 | `#include "cipher_structural_lookup.h"` | `#include "may13/cipher_structural_lookup.h"` | `cipher_structural_lookup.h` |
| 4 | `cipher_sense.h` | 12 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 5 | `cipher_predict.h` | 14 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 6 | `cipher_determinism.h` | 10 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 7 | `cipher_liquid_state.h` | 30 | `#  include "cipher_stubs.h"` | `#  include "may13/cipher_stubs.h"` | `cipher_stubs.h` |
| 8 (delta) | `cipher_recipes.h` | 38 | `#include "cipher_classify.hpp"` | `#include "may13/cipher_classify.hpp"` | `cipher_classify.hpp` |

### 3.4 Combined per-header table (complete list, for Step 2 v2 copy)

| # | Header | Origin in this audit | Depth in combined closure | Source path |
|---|---|---|---|---|
| 1 | `cipher_classify.hpp` | existing | 0 | `cipher-may13-evidence/include/cipher_classify.hpp` |
| 2 | `cipher_oracle.h` | existing | 0 | `cipher-may13-evidence/include/cipher_oracle.h` |
| 3 | `cipher_sense.h` | existing | 0 | `cipher-may13-evidence/include/cipher_sense.h` |
| 4 | `cipher_predict.h` | existing | 0 | `cipher-may13-evidence/include/cipher_predict.h` |
| 5 | `cipher_determinism.h` | existing | 0 | `cipher-may13-evidence/include/cipher_determinism.h` |
| 6 | `cipher_kernel_table.h` | existing | 0 | `cipher-may13-evidence/include/cipher_kernel_table.h` |
| 7 | `cipher_param_recovery.h` | existing | 0 | `cipher-may13-evidence/include/cipher_param_recovery.h` |
| 8 | `cipher_10ops.h` | existing | 1 | `cipher-may13-evidence/include/cipher_10ops.h` |
| 9 | `cipher_liquid_state.h` | existing | 1 | `cipher-may13-evidence/include/cipher_liquid_state.h` |
| 10 | `cipher_structural_lookup.h` | existing | 1 | `cipher-may13-evidence/include/cipher_structural_lookup.h` |
| 11 | `cipher_stubs.h` | existing | 2 | `cipher-may13-evidence/include/cipher_stubs.h` |
| 12 | `cipher_recipes.h` | **delta** | 0 (in delta; pulls from existing only) | `cipher-may13-evidence/include/cipher_recipes.h` |

---

## Section 4 — Sanity checks

### 4.1 LP-7 rename propagation across delta closure

`grep CipherKernelEntry cipher_recipes.h` — **zero hits**. The Step 1 rename (commit `fc8a9ae6...` in `cipher-may13-evidence`, tag `week-1-step-1-lp7-rename`) did not need to touch `cipher_recipes.h` because the file does not reference `CipherKernelEntry` at all. Combined-closure LP-7 propagation total: **0** (existing 11 + delta 1).

### 4.2 Out-of-`include/` dependencies

**Count: 0.** No double-quoted `#include` in `cipher_recipes.h` targets a file in `cipher-may13-evidence/src/` or any other directory outside `cipher-may13-evidence/include/`.

### 4.3 Circular dependencies across full 12-header graph

DFS run across the union graph (existing-closure adjacency from `WEEK_1_STEP_2_CLOSURE.md` §3.3 + delta adjacency from this document §1.2). Cycles found: **0**.

Updated topological summary including delta:

```
cipher_stubs.h
  └ cipher_liquid_state.h
      └ cipher_oracle.h
cipher_10ops.h
  ├ cipher_determinism.h
  ├ cipher_predict.h
  └ cipher_sense.h
cipher_classify.hpp                (depth 0; pulled by cipher_oracle.h AND cipher_recipes.h)
  ├ cipher_oracle.h (already shown above)
  └ cipher_recipes.h               (NEW edge: cipher_recipes.h → cipher_classify.hpp)
cipher_kernel_table.h              (independent)
cipher_param_recovery.h            (independent)
cipher_structural_lookup.h         (depth 1; pulled by cipher_oracle.h only)
cipher_recipes.h                   (NEW; depends only on cipher_classify.hpp)
cipher_oracle.h                    (closes the dep tree)
```

The new edge `cipher_recipes.h → cipher_classify.hpp` does not introduce a cycle. `cipher_classify.hpp` is a leaf (zero in-closure dependencies) so no back-edge is possible.

---

## Headline summary

| Field | Value |
|---|---|
| Status | **CLOSURE COMPLETE** |
| Delta closure size | 1 header |
| Delta rewrites | 1 |
| Combined closure size (existing + delta) | 12 headers |
| Combined rewrites (existing + delta) | 8 |
| External dependencies surfaced (delta) | 0 |
| Unresolved includes (delta) | 0 |
| LP-7 propagation check (delta) | PASS (0 hits) |
| Out-of-include/ check (delta) | PASS (0 violations) |
| Circular-dependency check (full 12-header graph) | PASS (0 cycles) |

### What this means for Step 2 v2 execution

The Step 2 v2 retry per the brief should:

1. Copy **12 headers** (the existing 11 + `cipher_recipes.h`) from `cipher-may13-evidence/include/` to `cipher_rt_phase4/include/may13/`.
2. Apply **8 sed rewrites** to the destination copies (§3.3 above is the complete authoritative list).
3. Re-run the synthetic compile smoke (with `-I/home/ubuntu/cipher_rt_phase4/include`) — expectation: rc=0 because all transitive deps are present and `may13/`-prefixed.
4. Re-verify the library invariant — expectation: `libcipher_rt.so` md5 stays at anchor `83afd1ca`.
5. Commit + tag `week-1-step-2-header-port`.

Step 2 v2 is **unblocked**. The delta walk surfaced no new external dependencies, no new transitive headers, no LP-7 leak, no out-of-include deps, no cycles. The combined 12-header / 8-rewrite manifest is the authoritative input.

---

**End of WEEK_1_STEP_2_CLOSURE_DELTA.md.**
