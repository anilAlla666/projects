# Week 1 Step 2 v2 — Result

**HEADLINE STATUS: PASS.**

**Date:** 2026-05-20
**Scope:** Week 1 Step 2 retry under Option B closure (12-header port + 8 bare-name rewrites). Step 2 PARTIAL surfaced 3 missing transitive deps + 1 bare-name resolution gap; the closure docs (`WEEK_1_STEP_2_CLOSURE.md` md5 `1f55748ac929aeb1ae2da749ad301515`, `WEEK_1_STEP_2_CLOSURE_DELTA.md` md5 `38423a8c65d8b1eae9a3ef430e9f5734`) corrected the scope. This retry executed against the corrected manifest.
**Outcome:** All six steps PASS. Library invariant held (`libcipher_rt.so` md5 stays at anchor `83afd1ca4118dc651854ef751e4ef82d`). 12-header synthetic compile-only smoke succeeds. CP 5.4 isolation 15/15 PASS. Commit `50a6f2283032420d7bd664f29ba1d81a786f25c0` landed and tagged `week-1-step-2-v2-header-port`.

---

## Section A — Pre-port verification + clean prior attempt

### A.1 Git state at entry

| Field | Value |
|---|---|
| HEAD | `b621453c4602f78d4751502a6873032ca1999e7a` |
| `pre-week-1-baseline` tag | `b621453c4602f78d4751502a6873032ca1999e7a` (matches HEAD) |
| Pre-cleanup working tree | `?? include/` (Step 2 PARTIAL's 7 untracked headers) |

### A.2 Step 2 residue cleaned

```
$ rm -rf include/
$ git status --short
(clean — no untracked, no modified)
```

### A.3 Pre-port libcipher_rt.so md5

`83afd1ca4118dc651854ef751e4ef82d` — matches anchor `83afd1ca` documented in WEEK_1_PRE_FLIGHT.md §2.1.

### A.4 Pre-port baseline build

`make clean && make` in cipher_rt_phase4/:

- rc = 0
- Warning count: **18**
- Post-build libcipher_rt.so md5: `83afd1ca4118dc651854ef751e4ef82d` (anchor REPRODUCED byte-identical)
- Full log: `/tmp/week1_step2_v2/pre_port_v2_build.log`
- Pre-port .so preserved at `/tmp/week1_step2_v2/libcipher_rt.so.preport_v2`

**Section A verdict: PASS.** Clean baseline; anchor confirmed; ready to port.

---

## Section B — Copy 12 closure headers

### B.1 Destination created

`mkdir -p /home/ubuntu/cipher_rt_phase4/include/may13/` — directory created fresh after Step 2 residue removal.

### B.2 Source-vs-destination md5 match for all 12

| Destination file | md5 | source size (B) |
|---|---|---|
| `cipher_classify.hpp` | `4c615fa12e6a4f2465d57edabc7ac893` | 9,952 |
| `cipher_oracle.h` | `a871785beea709d8b99171ee2ec57ac0` | 9,238 |
| `cipher_sense.h` | `c7f66410679bfdbbe2181e8dff3f3b3c` | 1,527 |
| `cipher_predict.h` | `acb714b5c5734abd026a38718d400d27` | 1,366 |
| `cipher_determinism.h` | `c94191c1b399c06076ba05e62a89dedd` | 724 |
| `cipher_kernel_table.h` | `349d327f4f68c0309f6c1200576b3e63` | 3,492 |
| `cipher_param_recovery.h` | `b6d3bd74b3f2a5a784ca26afe53ce0e6` | 4,190 |
| `cipher_10ops.h` | `202a4d97bfe709f08da762a38f07d905` | 4,349 |
| `cipher_liquid_state.h` | `e3639184a86f9f7df0b4555b24b5821f` | 10,297 |
| `cipher_structural_lookup.h` | `d2bf16c4abe5300298a53a5db6db032a` | 4,273 |
| `cipher_stubs.h` | `b66ed66f0d487a3ed994984adb130899` | 6,993 |
| `cipher_recipes.h` (delta) | `e4d7e84a04380c062d2d8a0a6b66575c` | 8,843 |

12/12 md5 source-vs-destination MATCH. Total footprint at destination: 65,244 bytes.

### B.3 Verification

- `ls include/may13/ | wc -l` → **12** (matches expected).
- `grep -rn "CipherKernelEntry" include/may13/` → **0** hits (Step 1 rename propagated; the 2 rename-target headers contain only `CipherKtEntry` and `CipherParamEntry`).

**Section B verdict: PASS.** All 12 closure headers byte-identical at destination; LP-7 propagation intact.

---

## Section C — Apply 8 bare-name rewrites

### C.1 Pre-rewrite verification — every "before" line matches the closure-doc-recorded text

All 8 expected `before` lines matched on first read; **zero drift** between the closure manifest and the destination copies. Per-line verification:

```
MATCH  cipher_oracle.h:42         #include "cipher_liquid_state.h"
MATCH  cipher_oracle.h:43         #include "cipher_classify.hpp"
MATCH  cipher_oracle.h:44         #include "cipher_structural_lookup.h"
MATCH  cipher_sense.h:12          #include "cipher_10ops.h"
MATCH  cipher_predict.h:14        #include "cipher_10ops.h"
MATCH  cipher_determinism.h:10    #include "cipher_10ops.h"
MATCH  cipher_liquid_state.h:30   #  include "cipher_stubs.h"    [indented inside #ifdef CIPHER_CPU_STUB]
MATCH  cipher_recipes.h:38        #include "cipher_classify.hpp"
```

### C.2 Rewrites applied (per closure-doc §3.3 manifest)

Per-line exact-text rewrite (Python one-shot, not blind sed-replace, so each `before → after` pair is enforced against the exact line and the indented `#  include` form on `cipher_liquid_state.h:30` is preserved):

| # | File | Line | After |
|---|---|---|---|
| 1 | `cipher_oracle.h` | 42 | `#include "may13/cipher_liquid_state.h"` |
| 2 | `cipher_oracle.h` | 43 | `#include "may13/cipher_classify.hpp"` |
| 3 | `cipher_oracle.h` | 44 | `#include "may13/cipher_structural_lookup.h"` |
| 4 | `cipher_sense.h` | 12 | `#include "may13/cipher_10ops.h"` |
| 5 | `cipher_predict.h` | 14 | `#include "may13/cipher_10ops.h"` |
| 6 | `cipher_determinism.h` | 10 | `#include "may13/cipher_10ops.h"` |
| 7 | `cipher_liquid_state.h` | 30 | `#  include "may13/cipher_stubs.h"` (indentation preserved) |
| 8 | `cipher_recipes.h` | 38 | `#include "may13/cipher_classify.hpp"` |

### C.3 Post-rewrite verification

- `grep -rnE '^\s*#\s*include\s+"may13/' include/may13/ | wc -l` → **8** (matches expected).
- `grep -rnE '^\s*#\s*include\s+"cipher_' include/may13/` → **zero hits** (no residual bare-name include of any closure header).

### C.4 Verbatim cross-check on the indented include

`sed -n '28,33p' include/may13/cipher_liquid_state.h`:

```
#pragma once
#ifdef CIPHER_CPU_STUB
#  include "may13/cipher_stubs.h"
#else
#  include <cuda.h>
#  include <cuda_runtime.h>
```

Indentation preserved (`#  include` with two-space gap) inside the `#ifdef CIPHER_CPU_STUB` block.

**Section C verdict: PASS.** All 8 rewrites applied to expected lines; indentation preserved; zero residual bare-name closure includes.

---

## Section D — Compile-only verification

### D.1 Post-port library clean build

`make clean && make`:

- rc = 0
- Warning count: **18** (no change from pre-port baseline)
- Full log: `/tmp/week1_step2_v2/post_port_v2_build.log`

### D.2 Library invariant — INVARIANT HELD

| Metric | Pre-port | Post-port | Status |
|---|---|---|---|
| `libcipher_rt.so` md5 | `83afd1ca4118dc651854ef751e4ef82d` | `83afd1ca4118dc651854ef751e4ef82d` | **HELD ✓ byte-identical** |
| Build rc | 0 | 0 | unchanged |
| Warning count | 18 | 18 | delta zero |

The Step 2 INVARIANT is the load-bearing correctness signal: the headers exist on disk at `include/may13/` but no library TU includes them, no Makefile change was made, and no `-I` flag references the new directory. The library build sees zero observable change. Verified.

### D.3 Synthetic 12-header compile-only smoke

`/tmp/may13_compile_check.cpp` updated to `#include` all 12 may13/-prefixed headers in one TU:

```
#include "may13/cipher_classify.hpp"
#include "may13/cipher_oracle.h"
#include "may13/cipher_sense.h"
#include "may13/cipher_predict.h"
#include "may13/cipher_determinism.h"
#include "may13/cipher_kernel_table.h"
#include "may13/cipher_param_recovery.h"
#include "may13/cipher_10ops.h"
#include "may13/cipher_liquid_state.h"
#include "may13/cipher_structural_lookup.h"
#include "may13/cipher_stubs.h"
#include "may13/cipher_recipes.h"
int main() { return 0; }
```

Compile command:

```
g++ -O2 -Wall -Wextra -fPIC \
    -I/home/ubuntu/cipher_rt_phase4/include \
    -I/home/ubuntu/cipher_kmod -I/usr/include -I/usr/include \
    -c -o /tmp/week1_step2_v2/may13_compile_check.o \
    /tmp/may13_compile_check.cpp
```

**Result: rc = 0.** All 12 closure headers compile coexisting in one TU. Output object 1,304 bytes (substantively empty `main()`; the result confirms the header set is self-consistent and the `may13/` resolution works under `-Iinclude`).

### D.4 What this proves

1. The LP-7 collision (Step 1) is fully resolved — both renamed headers can be `#include`d in the same TU without struct-redefinition errors.
2. The Option B closure (12 headers) is **complete** — no further transitive headers need to be ported. The synthetic TU walks the full quoted-include graph and finds every target resolvable.
3. The bare-name rewrite manifest (8 rewrites) is **complete** — every in-closure quoted include resolves via the `may13/` prefix under `-Iinclude` resolution.
4. The Step 2 library invariant holds: the headers are on disk but inert to the existing library build because no library TU includes them and no Makefile change was made.

**Section D verdict: PASS.** Library invariant held; 12-header synthetic smoke compiles cleanly.

---

## Section E — Regression smoke

### E.1 CP 5.4 isolation 15/15

Re-ran `cipher-fusion-evidence/cp_5_4/step1_3/cp54_isolation_test`:

| Test | Sub-assertion | Result |
|---|---|---|
| Test 1 — legacy nr-9 deactivated | nr-9 REQUEST_SM_PARTITION returns -ENOSYS | PASS |
| Test 2 — ALLOCATE / FREE / QUERY | (4 sub-assertions) | 4× PASS |
| Test 3 — pool resize | (4 sub-assertions) | 4× PASS |
| Test 4 — do_exit reaper | child exited → reaper reclaimed 3 groups | PASS |
| Test 5 — disjointness | (3 sub-assertions) | 3× PASS |
| Test 6 — concurrent stress | (2 sub-assertions) | 2× PASS |
| **Total** | **15 sub-assertions** | **15 PASS, 0 FAIL** |

Output line-for-line identical to the Step 1 baseline (`/tmp/week1_step2_v2/cp54_isolation.log`).

### E.2 Other tests in scope

`cipher_rt_phase4/tests/` does not exist as a directory; CP 5.4 isolation is the primary substrate regression for this Step. No additional pod-runnable test infrastructure was found. The library md5 invariant from Section D is the load-bearing correctness signal in the absence of a richer regression suite.

**Section E verdict: PASS.** No regression introduced; CP 5.4 isolation 15/15 byte-identical to baseline.

---

## Section F — Commit and tag

### F.1 Diff stat

```
 include/may13/cipher_10ops.h             | 122 +++++++++++++++
 include/may13/cipher_classify.hpp        | 250 +++++++++++++++++++++++++++++++
 include/may13/cipher_determinism.h       |  24 +++
 include/may13/cipher_kernel_table.h      |  90 +++++++++++
 include/may13/cipher_liquid_state.h      | 244 ++++++++++++++++++++++++++++++
 include/may13/cipher_oracle.h            | 218 +++++++++++++++++++++++++++
 include/may13/cipher_param_recovery.h    | 101 +++++++++++++
 include/may13/cipher_predict.h           |  36 +++++
 include/may13/cipher_recipes.h           | 199 ++++++++++++++++++++++++
 include/may13/cipher_sense.h             |  45 ++++++
 include/may13/cipher_structural_lookup.h |  99 ++++++++++++
 include/may13/cipher_stubs.h             | 139 +++++++++++++++++
 12 files changed, 1567 insertions(+)
```

All 12 files are new additions; zero existing files modified. The `Makefile` is untouched (Option 2 strict from Step 2 carried forward).

### F.2 Commit and tag

| Field | Value |
|---|---|
| Commit SHA | `50a6f2283032420d7bd664f29ba1d81a786f25c0` |
| Tag | `week-1-step-2-v2-header-port` (lightweight) at the same SHA |
| `pre-week-1-baseline` tag (preserved) | `b621453c4602f78d4751502a6873032ca1999e7a` |
| Working tree post-commit | clean (zero pending) |
| Author | `Anil <anil.0666369@gmail.com>` |
| Co-Authored-By | `Claude Opus 4.7 (1M context) <noreply@anthropic.com>` |

### F.3 Commit message (verbatim)

```
Week 1 Step 2 v2: may13 header port (compile-only, Option B closure)

Step 2 PARTIAL surfaced 3 missing transitive deps + bare-name include
resolution gap. Option B adjudication: Wave 5 Weeks 1-5 starting set,
rewrite includes to may13/ prefix in destination copies.

- 12 closure headers copied to include/may13/ (existing 11-header
  closure + cipher_recipes.h delta)
- 8 bare-name include rewrites applied to destination copies
  (source tree cipher-may13-evidence unchanged)
- Closure docs: WEEK_1_STEP_2_CLOSURE.md (md5 1f55748ac929aeb1ae2da749ad301515),
  WEEK_1_STEP_2_CLOSURE_DELTA.md (md5 38423a8c65d8b1eae9a3ef430e9f5734)

Pre-port DSO build md5s (rc=0; 18 warnings):
  libcipher_rt.so:           83afd1ca4118dc651854ef751e4ef82d

Post-port DSO build md5s (rc=0; 18 warnings; delta zero):
  libcipher_rt.so:           83afd1ca4118dc651854ef751e4ef82d (INVARIANT HELD ✓)

12-header synthetic compile-only smoke:
  /tmp/may13_compile_check.cpp pulls all 12 may13/-prefixed includes
  in one TU. g++ -O2 -Wall -Wextra -fPIC -Iinclude rc=0.
  Confirms LP-7 collision resolved AND transitive closure complete
  AND bare-name resolution works under -Iinclude.

CP 5.4 isolation regression: 15/15 PASS (no regression vs Step 1 baseline).

Resolves Week 1 Step 2 of integration sequence per
CIPHER_REENGINEERING_PLAN.md v1.2.2 §7. Closure scope expansion
documented in WEEK_1_STEP_2_CLOSURE.md + WEEK_1_STEP_2_CLOSURE_DELTA.md;
no plan v1.2.3 bump (per adjudication).

Substrate behavior unchanged. No may13 code reached by existing
dispatch paths. Headers ready for Week 2 wiring.

Rollback: git reset --hard pre-week-1-baseline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Section F verdict: PASS.** Commit landed, tag placed, working tree clean.

---

## Headline summary

| Field | Value |
|---|---|
| Headline status | **PASS** |
| Step A — clean residue + baseline | PASS |
| Step B — copy 12 closure headers | PASS (12/12 md5-matched) |
| Step C — apply 8 bare-name rewrites | PASS (8/8 applied; zero residual bare-name includes) |
| Step D — library invariant + smoke | PASS (`libcipher_rt.so` md5 unchanged at anchor `83afd1ca`; 12-header synthetic smoke rc=0) |
| Step E — CP 5.4 isolation regression | PASS (15/15) |
| Step F — commit + tag | PASS (SHA `50a6f228...`, tag `week-1-step-2-v2-header-port`) |
| Files changed | 12 (all in `cipher_rt_phase4/include/may13/`) |
| Lines added | 1,567 insertions |
| Rollback path | `git reset --hard pre-week-1-baseline` (tag preserved at `b621453c...`) |

### Surfaced (no mitigation proposed; informational)

1. **The `Makefile` is unchanged.** No `-I` flag for `include/may13/` was added (Option 2 strict carried from Step 2). The synthetic compile smoke used a test-local `-Iinclude` flag. Week 2 will add the `-I` flag when the first consumer TU lands; until then, no library TU includes any may13 header.

2. **`cipher-may13-evidence` source tree is untouched.** All rewrites apply to the destination copies in `cipher_rt_phase4/include/may13/` only. The source tree stays at the `week-1-step-1-lp7-rename` tag (commit `fc8a9ae6...`). The bare-name includes in `cipher-may13-evidence/include/` remain legacy-correct for that tree's internal build.

### Week 1 Step 3 readiness

Week 1 Step 3 per plan v1.2.2 §7 Week 1 is the cross-tree compile harness — build a synthetic TU in `cipher_rt_phase4` that actually exercises one may13 header by declaring a variable of a may13 type, to prove the integration TU can hold both worlds. Still compile-only, no behavior change.

Pre-conditions for Step 3 are met:

- 12 may13 closure headers in place at `cipher_rt_phase4/include/may13/`, all `may13/`-prefixed includes resolvable.
- Library invariant verified holdable (anchor `83afd1ca` byte-identical to baseline).
- Step 2 v2 commit + tag landed; rollback path documented.
- Source tree (`cipher-may13-evidence`) at `week-1-step-1-lp7-rename` tag, unchanged.

**Week 1 Step 3 prompt can be sent.**

---

**End of WEEK_1_STEP_2_V2_RESULT.md.**
