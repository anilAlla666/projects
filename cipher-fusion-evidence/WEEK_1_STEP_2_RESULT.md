# Week 1 Step 2 — Result

**HEADLINE STATUS: PARTIAL.**

**Date:** 2026-05-20
**Scope:** may13 header port into `cipher_rt_phase4/include/may13/` as compile-only artifacts, per `CIPHER_REENGINEERING_PLAN.md` v1.2.2 §7 Week 1 + brief protocol.
**Outcome:** Steps A, B, C, and Step D's library-build invariant **PASS**. Step D's **synthetic smoke FAILED** on three transitive `#include` dependencies the port set did not satisfy (`cipher_10ops.h`, `cipher_liquid_state.h`, `cipher_structural_lookup.h`). Per the brief's discipline rule ("STOP and surface; document the missing dependency"), Steps E and F were not run. Working tree contains the 7 headers on disk at `include/may13/` as **uncommitted untracked files**; nothing has been committed in `cipher_rt_phase4`.
**Library invariant held:** `libcipher_rt.so` md5 = `83afd1ca4118dc651854ef751e4ef82d` (pre-port and post-port byte-identical to the documented anchor). Production runtime behavior unchanged.

---

## Section A — Pre-port verification

### A.1 Git state

| Field | Value |
|---|---|
| HEAD | `b621453c4602f78d4751502a6873032ca1999e7a` |
| `pre-week-1-baseline` tag | `b621453c4602f78d4751502a6873032ca1999e7a` (matches HEAD) |
| Working-tree status pre-step | clean |
| Other tags | only `pre-week-1-baseline` (cipher_rt_phase4 unchanged since v1.2.1 baseline) |

### A.2 Pre-port libcipher_rt.so md5

- On-disk pre-port: `83afd1ca4118dc651854ef751e4ef82d`, size 150,408 bytes — **matches anchor `83afd1ca`** documented in `WEEK_1_PRE_FLIGHT.md` §2.1.
- Pre-port `may13` references in `cipher_rt_phase4`: **zero** (no source file or Makefile mentions `may13`).
- Pre-existing `include/` subdir in `cipher_rt_phase4`: **none** (it will be created in Step B).

### A.3 Pre-port baseline build

`make clean && make` in `cipher_rt_phase4/`:

- **rc = 0**
- Baseline warning count: **18**
- Full log: `/tmp/week1_step2/pre_port_build.log`
- Post-baseline-build libcipher_rt.so md5: `83afd1ca4118dc651854ef751e4ef82d` (anchor REPRODUCED byte-identical)

Pre-port DSO copy preserved at `/tmp/week1_step2/libcipher_rt.so.preport` for Step D comparison.

**Section A verdict: PASS.** Baseline reproducible, anchor confirmed, zero `may13` baseline state.

---

## Section B — Header copy

### B.0 Filename discrepancy surfaced

The brief listed `cipher_classify.h` but the canonical file in `cipher-may13-evidence/include/` is `cipher_classify.hpp` (C++ header). Verified by `ls` — only the `.hpp` variant exists. The destination used the same `.hpp` extension to preserve source-of-truth identity.

### B.1 Destination created

`mkdir -p /home/ubuntu/cipher_rt_phase4/include/may13/` — `include/` and `include/may13/` both created (neither pre-existed in `cipher_rt_phase4`).

### B.2 Copy + md5 match verification

All 7 headers copied byte-identical (source md5 = destination md5):

| Destination file | md5 | Source path | Size (B) |
|---|---|---|---|
| `include/may13/cipher_classify.hpp` | `4c615fa12e6a4f2465d57edabc7ac893` | `cipher-may13-evidence/include/cipher_classify.hpp` | 9,952 |
| `include/may13/cipher_oracle.h` | `a871785beea709d8b99171ee2ec57ac0` | `cipher-may13-evidence/include/cipher_oracle.h` | 9,238 |
| `include/may13/cipher_sense.h` | `c7f66410679bfdbbe2181e8dff3f3b3c` | `cipher-may13-evidence/include/cipher_sense.h` | 1,527 |
| `include/may13/cipher_predict.h` | `acb714b5c5734abd026a38718d400d27` | `cipher-may13-evidence/include/cipher_predict.h` | 1,366 |
| `include/may13/cipher_determinism.h` | `c94191c1b399c06076ba05e62a89dedd` | `cipher-may13-evidence/include/cipher_determinism.h` | 724 |
| `include/may13/cipher_kernel_table.h` | `349d327f4f68c0309f6c1200576b3e63` | `cipher-may13-evidence/include/cipher_kernel_table.h` (post-Step-1 rename) | 3,492 |
| `include/may13/cipher_param_recovery.h` | `b6d3bd74b3f2a5a784ca26afe53ce0e6` | `cipher-may13-evidence/include/cipher_param_recovery.h` (post-Step-1 rename) | 4,190 |

### B.3 Step-1 rename propagation verification

- `grep -rn CipherKernelEntry include/may13/`: **0 hits** (Step 1 rename propagated to the copies as expected).
- `grep -rn CipherKtEntry include/may13/`: **3 hits** (in `cipher_kernel_table.h`).
- `grep -rn CipherParamEntry include/may13/`: **5 hits** (in `cipher_param_recovery.h`).
- Total of the two new names: **8** (matches the ≥8 expected per the brief's verification rule).

**Section B verdict: PASS.** 7 headers byte-identical at destination; Step-1 rename intact; no `CipherKernelEntry` residue.

---

## Section C — Include-path strategy

### C.1 Basename collision check

For each of the 7 ported header basenames, searched the entire `cipher_rt_phase4` tree (excluding the newly-created `include/may13/`):

| Basename | Existing file in cipher_rt_phase4? |
|---|---|
| `cipher_classify.*` | none |
| `cipher_oracle.*` | none |
| `cipher_sense.*` | none |
| `cipher_predict.*` | none |
| `cipher_determinism.*` | none |
| `cipher_kernel_table.*` | none |
| `cipher_param_recovery.*` | none |

Zero collisions. Existing `cipher_rt_phase4` headers all carry the `cipher_rt_` prefix (`cipher_rt_arbitrate.h`, `cipher_rt_attn_dispatch.h`, `cipher_rt_audit.h`, etc.).

### C.2 Existing `#include` directives for the 7 basenames in cipher_rt_phase4 sources

Searched all `.c` / `.cpp` / `.h` files: **zero existing `#include` directives** for any of the 7 ported basenames anywhere in the cipher_rt_phase4 source tree.

### C.3 Strategy chosen: **Option 2 strict — no Makefile change**

Rationale:

- Zero basename collisions (C.1).
- Zero existing `#include` directives for these names (C.2).
- The Step 2 invariant is "library build state unchanged" — the strictest way to honor that is to leave the build flags untouched.
- The synthetic compile smoke (Step D.4) uses a test-local `-Iinclude` flag that does NOT enter the Makefile.
- Future Weeks (Week 2 and later) add the include path at the point a consumer TU lands — at that time, the library build naturally needs to compile the new TU and any `-I` change is local to that change.

Current `cipher_rt_phase4/Makefile` (md5 `4191557fac002230af0f7f879dda1c05`) unchanged from baseline; `INCLUDES := -I$(CIPHER_KMOD_DIR) -I$(CUDA_INCLUDE) -I$(CUPTI_INCLUDE)` remains the only include-path setting.

**Section C verdict: PASS.** Option 2 strict adopted; Makefile untouched.

---

## Section D — Compile-only verification build

### D.1 Post-port library build

`make clean && make` in `cipher_rt_phase4/`:

- **rc = 0**
- Post-port warning count: **18**
- Warning delta vs pre-port baseline: **0**
- Full log: `/tmp/week1_step2/post_port_build.log`

### D.2 Library invariant — LIBCIPHER_RT.SO MD5 UNCHANGED

| Metric | Pre-port | Post-port | Invariant |
|---|---|---|---|
| libcipher_rt.so md5 | `83afd1ca4118dc651854ef751e4ef82d` | `83afd1ca4118dc651854ef751e4ef82d` | **HELD ✓ byte-identical** |
| Size (bytes) | 150,408 | 150,408 | identical |

The library compile-only invariant holds. The port has zero observable effect on `libcipher_rt.so` because no library TU includes any of the 7 ported headers and no Makefile change was made.

### D.3 Warning delta

Pre-port: 18 warnings. Post-port: 18 warnings. **Delta zero**.

### D.4 Synthetic smoke — **FAILED on transitive #include dependencies**

The synthetic smoke compile of `/tmp/may13_compile_check.cpp` (a TU that explicitly `#include`s all 7 may13 headers in one compilation unit, with `-I/home/ubuntu/cipher_rt_phase4/include` so subdir-prefixed includes resolve) failed at:

```
In file included from /tmp/may13_compile_check.cpp:12:
include/may13/cipher_oracle.h:42:10: fatal error: cipher_liquid_state.h: No such file or directory
   42 | #include "cipher_liquid_state.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```

This is the first missing transitive dependency. Full enumeration of all transitive `#include` deps in the 7 ported headers (D.5 below) shows the port set is short by 3 headers and also has an include-path resolution issue for one already-ported header.

### D.5 Full transitive-include enumeration

Quoted (`#include "..."`) directives in the 7 ported headers and their status:

| Transitive include | Cited from | Status |
|---|---|---|
| `cipher_10ops.h` | `cipher_determinism.h:10`, `cipher_predict.h:14`, `cipher_sense.h:12` | **NOT in port set; available in `cipher-may13-evidence/include/cipher_10ops.h`** |
| `cipher_liquid_state.h` | `cipher_oracle.h:42` | **NOT in port set; available in `cipher-may13-evidence/include/cipher_liquid_state.h`** |
| `cipher_structural_lookup.h` | `cipher_oracle.h:44` | **NOT in port set; available in `cipher-may13-evidence/include/cipher_structural_lookup.h`** |
| `cipher_classify.hpp` | `cipher_oracle.h:43` | Already ported to `include/may13/`, but cipher_oracle.h includes it via bare relative name `"cipher_classify.hpp"`; with `-Iinclude` alone, the path resolves to `include/cipher_classify.hpp` (not present). Two ways forward, surfaced for adjudication: (a) add `-Iinclude/may13` so bare-name resolution works, or (b) rewrite cipher_oracle.h's include directive to `"may13/cipher_classify.hpp"` for consistency. |

System / angle-bracket includes (no path concerns, all standard library):

```
<array>      <atomic>     <cstdint>    <cstring>
<stdbool.h>  <stddef.h>   <stdint.h>
```

### D.6 Implication

The headers DO compile under the existing cipher_rt_phase4 library build because no library TU includes them — the transitive dependency tree is never walked by the production build. The library invariant holds (D.2). However, **any future consumer TU that #includes a may13 header WILL hit the missing transitive deps** unless the additional 3 headers are ported and the `cipher_classify.hpp` bare-name resolution is fixed.

The port set defined by the brief is therefore **necessary but not sufficient** for Week 2 wiring. Surfaced for user adjudication. Per the brief discipline rule, no mitigation is proposed in this document.

### D.7 Step D outcome

| Component | Verdict |
|---|---|
| D.1 library build rc=0 | PASS |
| D.2 libcipher_rt.so md5 unchanged | PASS (invariant held) |
| D.3 warning delta zero | PASS |
| D.4 synthetic smoke compile | **FAIL on transitive deps** |
| Overall Step D | **PARTIAL** |

**Section D verdict: PARTIAL.** Library invariant held; synthetic smoke failed on documented transitive dependency gap.

---

## Section E — Regression smoke

**NOT RUN per brief discipline.** The brief instructs "STOP and surface" if Step D's synthetic compile fails. Step E (CP 5.4 isolation regression) and Step F (commit + tag) are deferred to the next round, once the transitive-dep adjudication lands.

The implicit regression signal from the library build (D.2) is that `libcipher_rt.so` is byte-identical to the anchor — production runtime behavior is unchanged. CP 5.4 isolation 15/15 PASS was independently verified in `WEEK_1_PRE_FLIGHT.md` §3.1 against this same anchor and would re-pass if rerun.

---

## Section F — Commit and tag

**NOT EXECUTED** per the brief discipline rule. No commit, no tag.

Current `cipher_rt_phase4` working-tree state:

- HEAD: `b621453c4602f78d4751502a6873032ca1999e7a` (unchanged, at `pre-week-1-baseline` tag).
- Working-tree pending: `include/` subdir is untracked (contains the 7 headers from Step B). Makefile unchanged. No tracked file modified.
- Rollback path: `rm -rf /home/ubuntu/cipher_rt_phase4/include` removes all artifacts of this step. `pre-week-1-baseline` tag is unchanged and remains the rollback anchor.

The untracked headers are left on disk in their current location so the next round can fold the additional transitive headers and commit everything at once. If the user prefers a fully clean working tree, the headers can be removed with the rollback command above without prejudice to the next round.

---

## Headline summary

| Field | Value |
|---|---|
| Headline status | **PARTIAL** |
| Step A (pre-verification) | PASS |
| Step B (header copy) | PASS |
| Step C (include strategy) | PASS |
| Step D library invariant | PASS (libcipher_rt.so md5 unchanged at anchor `83afd1ca`) |
| Step D synthetic smoke | **FAIL** (3 transitive deps missing + 1 bare-name resolution gap) |
| Step E (regression) | NOT RUN per discipline |
| Step F (commit/tag) | NOT EXECUTED per discipline |
| Working-tree diff | `include/may13/` (7 untracked files) |
| Rollback path | `rm -rf cipher_rt_phase4/include` (or leave for next round) |

### Surfaced for user adjudication (no mitigation proposed in this document)

1. **Three transitive `#include` headers not in the port set:**
   - `cipher_10ops.h` (consumed by `cipher_determinism.h`, `cipher_predict.h`, `cipher_sense.h`)
   - `cipher_liquid_state.h` (consumed by `cipher_oracle.h:42`)
   - `cipher_structural_lookup.h` (consumed by `cipher_oracle.h:44`)
   - All three available verbatim in `cipher-may13-evidence/include/`. Each may carry its own further transitive dependencies that would surface when ported (chain not walked in this step).

2. **`cipher_classify.hpp` bare-name resolution issue:** `cipher_oracle.h:43` includes it via bare relative name `#include "cipher_classify.hpp"`. With Option 2 strict (`-Iinclude` only), the bare name doesn't resolve to `include/may13/cipher_classify.hpp`. Two forward-paths surfaced (require user pick): (a) add `-Iinclude/may13` so bare-name resolution works (returns to Option 1 territory); (b) rewrite cipher_oracle.h's directive to `"may13/cipher_classify.hpp"` (touches a ported file's content).

3. **Brief filename discrepancy on `cipher_classify`:** brief listed `.h`; canonical file is `.hpp`. Surfaced in §B.0. The port preserved `.hpp` for source-of-truth fidelity.

### Week 1 Step 2.1 (or equivalent) readiness

The next round can proceed once the user adjudicates the 3 missing transitive headers and the `cipher_classify.hpp` resolution. The pre-conditions for that round are:

- `cipher_rt_phase4/include/may13/` already contains 7 headers (no need to re-copy).
- Library invariant proven holdable (anchor `83afd1ca` byte-identical to baseline).
- The dependency chain beyond the 3 named transitives is unmeasured; the round that ports them may surface further-order deps and should walk the chain explicitly.

**Week 1 Step 2 is not done. Adjudication required before retry.**

---

**End of WEEK_1_STEP_2_RESULT.md.**
