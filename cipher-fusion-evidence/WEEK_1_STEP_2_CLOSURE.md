# Week 1 Step 2 — Pre-Step-2-v2 Closure Enumeration

**HEADLINE STATUS: CLOSURE COMPLETE.**

**Date:** 2026-05-20
**Predecessor:** WEEK_1_STEP_2_RESULT.md (md5 `d4ff2fc5408f84edc26b5baa3bf8d678`) — Step 2 returned PARTIAL on 3 transitive `#include` deps + 1 bare-name resolution gap.
**Adjudication input:** Option (b) — rewrite bare-name may13 includes inside the ported headers to use the `may13/` prefix. Applied across the entire transitive closure, not only the 7 originally scoped headers.
**Scope:** Pure enumeration. No code changes. No source modification. The walker output is the input to Step 2 v2.
**Walker:** `/tmp/closure_walker.py` (BFS over `#include` graph, scoped to `cipher-may13-evidence/include/`). Full JSON output at `/tmp/closure_result.json`.

---

## Section 1 — Transitive closure

### 1.1 Closure size

**11 headers in total** (7 starting set + 3 named transitive + 1 fourth-layer discovery).

| Tier | Count | Headers |
|---|---|---|
| Starting set (audit-named) | 7 | `cipher_classify.hpp`, `cipher_oracle.h`, `cipher_sense.h`, `cipher_predict.h`, `cipher_determinism.h`, `cipher_kernel_table.h`, `cipher_param_recovery.h` |
| Depth 1 (direct transitive of starting set) | 3 | `cipher_10ops.h`, `cipher_liquid_state.h`, `cipher_structural_lookup.h` |
| Depth 2 (transitive-of-transitive) | 1 | `cipher_stubs.h` (pulled by `cipher_liquid_state.h:30`) |
| **Total closure size** | **11** | — |

### 1.2 Per-header dependency table

| Header | Depth | Source path | Size (B) | md5 | Direct in-closure deps | External deps | Angle includes |
|---|---|---|---|---|---|---|---|
| `cipher_classify.hpp` | 0 | `cipher-may13-evidence/include/cipher_classify.hpp` | 9,952 | `4c615fa12e6a4f2465d57edabc7ac893` | (none) | (none) | `<array>`, `<atomic>`, `<cstdint>`, `<cstring>` |
| `cipher_determinism.h` | 0 | `cipher-may13-evidence/include/cipher_determinism.h` | 724 | `c94191c1b399c06076ba05e62a89dedd` | `cipher_10ops.h` | (none) | `<stdint.h>` |
| `cipher_kernel_table.h` | 0 | `cipher-may13-evidence/include/cipher_kernel_table.h` | 3,492 | `349d327f4f68c0309f6c1200576b3e63` | (none) | (none) | `<stddef.h>`, `<stdint.h>` |
| `cipher_oracle.h` | 0 | `cipher-may13-evidence/include/cipher_oracle.h` | 9,238 | `a871785beea709d8b99171ee2ec57ac0` | `cipher_classify.hpp`, `cipher_liquid_state.h`, `cipher_structural_lookup.h` | (none) | `<stdbool.h>`, `<stdint.h>` |
| `cipher_param_recovery.h` | 0 | `cipher-may13-evidence/include/cipher_param_recovery.h` | 4,190 | `b6d3bd74b3f2a5a784ca26afe53ce0e6` | (none) | (none) | `<stddef.h>`, `<stdint.h>` |
| `cipher_predict.h` | 0 | `cipher-may13-evidence/include/cipher_predict.h` | 1,366 | `acb714b5c5734abd026a38718d400d27` | `cipher_10ops.h` | (none) | `<stdint.h>` |
| `cipher_sense.h` | 0 | `cipher-may13-evidence/include/cipher_sense.h` | 1,527 | `c7f66410679bfdbbe2181e8dff3f3b3c` | `cipher_10ops.h` | (none) | `<stdint.h>` |
| `cipher_10ops.h` | 1 | `cipher-may13-evidence/include/cipher_10ops.h` | 4,349 | `202a4d97bfe709f08da762a38f07d905` | (none) | (none) | `<atomic>`, `<stdatomic.h>`, `<stdint.h>` |
| `cipher_liquid_state.h` | 1 | `cipher-may13-evidence/include/cipher_liquid_state.h` | 10,297 | `e3639184a86f9f7df0b4555b24b5821f` | `cipher_stubs.h` | (none) | `<atomic>`, `<cuda.h>`, `<cuda_runtime.h>`, `<stdbool.h>`, `<stdint.h>` |
| `cipher_structural_lookup.h` | 1 | `cipher-may13-evidence/include/cipher_structural_lookup.h` | 4,273 | `d2bf16c4abe5300298a53a5db6db032a` | (none) | (none) | `<stdbool.h>`, `<stdint.h>` |
| `cipher_stubs.h` | 2 | `cipher-may13-evidence/include/cipher_stubs.h` | 6,993 | `b66ed66f0d487a3ed994984adb130899` | (none) | (none) | `<atomic>`, `<stdbool.h>`, `<stdint.h>`, `<stdlib.h>`, `<string.h>` |

**Total closure footprint:** 56,401 bytes across 11 files.

### 1.3 External dependencies surfaced

**Count: 0.** Every double-quoted `#include` directive inside the 11 closure headers resolves to a file inside `cipher-may13-evidence/include/`. There are no double-quoted includes that point to files outside the closure (i.e., no header in this closure depends on a header that lives in `cipher-may13-evidence/src/` or anywhere else outside `include/`).

### 1.4 Angle-bracketed includes (cipher_rt_phase4 must resolve these via its existing include path)

Aggregated across the closure, the angle-bracket include surface is:

```
<array>           <atomic>         <cstdint>        <cstring>
<cuda.h>          <cuda_runtime.h> <stdatomic.h>    <stdbool.h>
<stddef.h>        <stdint.h>       <stdlib.h>       <string.h>
```

All of these resolve through cipher_rt_phase4's existing include path: `-I$(CUDA_INCLUDE)` covers `<cuda.h>` and `<cuda_runtime.h>`; the rest are C/C++ standard library headers resolved by the system toolchain. No new `-I` flag is required for the angle-bracket includes.

### 1.5 Unresolved includes

**Count: 0.** Every quoted include reached during BFS either lands inside the closure (i.e., the target file exists in `cipher-may13-evidence/include/` and is added to the closure) or is angle-bracketed (handled in §1.4). The walker found zero unresolved double-quoted includes.

### 1.6 Closure walker invocation record

```
$ python3 /tmp/closure_walker.py > /tmp/closure_result.json
$ echo $?
0
```

The closure walker is `/tmp/closure_walker.py` (created in this turn for the enumeration). Algorithm: BFS from `START_SET` using `re` for `#include` regex parsing, with `find_in_include_dir(name)` checking `cipher-may13-evidence/include/<name>` for existence. Per-header bookkeeping: `md5`, `size`, `depth` (BFS distance from starting set), `direct_deps` (in-closure quoted includes), `external_deps` (quoted includes not in closure — empty here), `angle_includes`.

---

## Section 2 — Bare-name rewrite manifest

### 2.1 Rewrite count

**Total rewrites: 7** lines across 4 source files.

### 2.2 Per-file rewrite table

| # | File (source path in `cipher-may13-evidence/include/`) | Line | Before | After | Target in closure |
|---|---|---|---|---|---|
| 1 | `cipher_oracle.h` | 42 | `#include "cipher_liquid_state.h"` | `#include "may13/cipher_liquid_state.h"` | `cipher_liquid_state.h` |
| 2 | `cipher_oracle.h` | 43 | `#include "cipher_classify.hpp"` | `#include "may13/cipher_classify.hpp"` | `cipher_classify.hpp` |
| 3 | `cipher_oracle.h` | 44 | `#include "cipher_structural_lookup.h"` | `#include "may13/cipher_structural_lookup.h"` | `cipher_structural_lookup.h` |
| 4 | `cipher_sense.h` | 12 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 5 | `cipher_predict.h` | 14 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 6 | `cipher_determinism.h` | 10 | `#include "cipher_10ops.h"` | `#include "may13/cipher_10ops.h"` | `cipher_10ops.h` |
| 7 | `cipher_liquid_state.h` | 30 | `#  include "cipher_stubs.h"` | `#  include "may13/cipher_stubs.h"` | `cipher_stubs.h` |

Note on rewrite #7: the source line `cipher_liquid_state.h:30` uses indented `#  include` (two spaces between `#` and `include`) because it sits inside an `#ifdef CIPHER_CPU_STUB` block. The rewrite preserves the indentation; the manifest column "Before/After" shows the directive verbatim with the leading whitespace.

Note on rewrite scope: these 7 rewrites are the **only** quoted-include modifications required for the closure. The remaining 4 closure headers (`cipher_classify.hpp`, `cipher_kernel_table.h`, `cipher_param_recovery.h`, `cipher_10ops.h`, `cipher_structural_lookup.h`, `cipher_stubs.h`) have zero in-closure quoted includes (their only includes are either angle-bracketed system headers or empty).

### 2.3 Rewrite strategy for Step 2 v2

Per the brief's Option (b) framing, the rewrites are applied to the **destination copies** under `cipher_rt_phase4/include/may13/`, not to the source headers in `cipher-may13-evidence/include/`. The source tree (cipher-may13-evidence) stays at its `week-1-step-1-lp7-rename` tag — the bare-name includes there are legacy-correct for that tree's internal build.

A single sed pass per rewrite (or a single multi-pattern sed) suffices. Suggested Step 2 v2 execution order:

1. Copy all 11 closure headers from `cipher-may13-evidence/include/` to `cipher_rt_phase4/include/may13/` (replaces the 7-header copy from Step 2; the same destination directory).
2. Apply the 7 rewrites listed in §2.2 to the destination files.
3. Re-run the synthetic compile smoke. Expectation: rc=0 because all transitive deps are now both present AND addressable via `-Iinclude` resolution.
4. Re-verify the library invariant (libcipher_rt.so md5 unchanged at anchor `83afd1ca`).
5. Commit + tag `week-1-step-2-header-port`.

---

## Section 3 — Sanity checks

### 3.1 LP-7 rename propagation across the closure

`grep CipherKernelEntry` across all 11 closure headers — **zero hits**.

The Step 1 rename (commit `fc8a9ae6025cff345e86d054b1df02d07a9fb1fe` in `cipher-may13-evidence`, tag `week-1-step-1-lp7-rename`) propagated correctly: the closure headers either contain the rename target (`CipherKtEntry` in `cipher_kernel_table.h`, `CipherParamEntry` in `cipher_param_recovery.h`) or contain no reference to either name. No closure header refers to the pre-rename `CipherKernelEntry`.

### 3.2 Out-of-`include/` dependencies

**Count: 0.** No closure header contains a double-quoted `#include` directive whose target exists somewhere in the `cipher-may13-evidence/` tree **outside** `include/`. The walker's `find_anywhere_in_may13` check returned zero hits in `src/` or root-level paths for every transitive include.

This rules out the scenario where a Step 2 v2 port would need to reclassify a "header" that actually lives in `src/` (which would not be portable as a header). All 11 closure entries live in `include/` and are header-shaped.

### 3.3 Circular dependencies within the closure

**Count: 0.** DFS from every closure node (with WHITE/GRAY/BLACK coloring) found zero back-edges and zero cycles. The 11-header closure forms a DAG.

Topological order (one valid linearization, depth-first):

```
cipher_stubs.h
  └ cipher_liquid_state.h
      └ cipher_oracle.h
cipher_10ops.h
  ├ cipher_determinism.h
  ├ cipher_predict.h
  └ cipher_sense.h
cipher_classify.hpp        (independent — no closure-internal deps)
cipher_kernel_table.h      (independent)
cipher_param_recovery.h    (independent)
cipher_structural_lookup.h (independent; also pulled by cipher_oracle.h)
cipher_oracle.h            (closes the dependency tree from depth 1 deps)
```

No header in the closure is reachable from itself via the in-closure adjacency graph.

---

## Headline summary

| Field | Value |
|---|---|
| Status | **CLOSURE COMPLETE** |
| Closure size | 11 headers |
| Rewrites required | 7 (in 4 source files) |
| External dependencies | 0 |
| Unresolved includes | 0 |
| LP-7 propagation check | PASS (0 hits) |
| Out-of-include/ check | PASS (0 violations) |
| Circular-dependency check | PASS (0 cycles) |

### What changed vs the 7-header port set from Week 1 Step 2 (PARTIAL)

| Aspect | Step 2 (PARTIAL) | Step 2 v2 closure |
|---|---|---|
| Header count | 7 | 11 (+4 transitive) |
| Headers added | — | `cipher_10ops.h`, `cipher_liquid_state.h`, `cipher_structural_lookup.h`, `cipher_stubs.h` |
| Bare-name rewrites | 0 (Step 2 used Option 2 strict) | 7 (Option (b) per adjudication) |
| External deps surfaced | 3 (missing transitive headers) | 0 |
| Unresolved includes | 1 (`cipher_classify.hpp` bare-name from `cipher_oracle.h:43`) | 0 (covered by rewrite #2 in §2.2) |
| Synthetic smoke expectation | failed | should pass once §2.2 rewrites applied + `-Iinclude` flag on synthetic test |

### Step 2 v2 is unblocked

All preconditions for Step 2 v2 retry are met:

- 11-header closure enumerated with full md5 + size + dep table (§1.2).
- 7-line rewrite manifest produced (§2.2).
- All sanity checks pass (§3.1, §3.2, §3.3).
- No further adjudication required.

Step 2 v2 can proceed when the user issues the retry prompt. The brief's "no mitigation proposed in this document" rule applies: this closure document enumerates the gap; the actual port + rewrite + build + commit is the next prompt's work.

---

**End of WEEK_1_STEP_2_CLOSURE.md.**
