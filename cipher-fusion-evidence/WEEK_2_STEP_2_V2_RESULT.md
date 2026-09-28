# Week 2 Step 2 v2 — may13 .cpp ports with extended closure — RESULT

**Status: PARTIAL — runtime regression surfaced at sub-step D. STOPPED. Awaiting adjudication.**

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 2 (retry) of 7
**Tree state at stop point (uncommitted):**
  - `cipher_rt_phase4` HEAD: `465b244e` (`week-2-step-1-lp2-sdpa-refactor`, unchanged)
  - Working tree: dirty (18 new files in `include/may13/` + `src/may13/`, Makefile modified)
  - libcipher_rt.so on disk: md5 `e8981f5aa5891ee3924d9b475130d1b2` (links cleanly, fails at runtime)

---

## 1. Headline

Sub-steps A-C built clean (closure walk, library invariant hold, .cpp link with 6 new TUs). Sub-step D (regression smoke) surfaced a runtime regression:

**The post-step .so contains 18 new undefined symbols. 7 are strong cross-TU references that the linker (`-shared` mode) accepted but the dynamic loader rejects.** Result: under `LD_PRELOAD`, python aborts with `symbol lookup error: undefined symbol: g_cipher`. Under `CUDA_INJECTION64_PATH` the CUDA driver silently swallows the shim-load failure and the process continues without the substrate — making the "finite SDPA output" smoke a false positive (the shim wasn't actually active).

Per brief discipline: STOP, do not propose mitigation, surface. Commit + tag (sub-step E) **NOT executed**. No anchor rotation.

---

## 2. Sub-step A — Extended closure walk — PASS

### Closure delta verification

| signal | result |
| --- | --- |
| FURTHER-ORDER new headers beyond the 6 | **NONE** |
| UNRESOLVED includes | **NONE** |
| Header rewrites enumerated | 15 |
| .cpp rewrites enumerated | 13 |
| Cumulative rewrites (Step 2 v2 + this) | 8 + 28 = 36 |

### Per-new-header `#include` classification

```
file                              tot  exi  new   p4  sys  fur  src  unr
cipher.h                            9    5    4    0    0    0    0    0
cipher_edmd.h                       3    1    0    0    2    0    0    0
cipher_intercept.h                  6    1    0    0    5    0    0    0
cipher_green_ctx.h                  5    1    0    0    4    0    0    0
cipher_l2_persist.h                 6    1    0    0    5    0    0    0
cipher_telemetry.h                  7    2    0    0    5    0    0    0
```

### Per-cpp `#include` classification (rewrite manifest)

```
file                              tot  exi  new   p4  sys  fur  src  unr
cipher_dispatch.cpp                11    5    2    0    4    0    0    0
cipher_oracle.cpp                   6    2    0    0    4    0    0    0
cipher_recipes.cpp                  5    1    0    0    4    0    0    0
cipher_sense.cpp                    8    1    0    0    7    0    0    0
cipher_structural_lookup.cpp        4    1    0    0    3    0    0    0
cipher_kernel_table.cpp             7    1    0    0    6    0    0    0
```

**HEADER closure clean** at 18. Walker only inspects `#include` directives; cannot infer cross-TU symbol references.

---

## 3. Sub-step B — Port 6 new headers + library invariant — PASS

### Header port

6 byte-identical copies into `include/may13/` (md5 matches verified for each):

| header | md5 |
| --- | --- |
| `cipher.h` | `0d3d7a3b433bdc35fbb12f33302bb5e5` |
| `cipher_edmd.h` | `8577a8ffe3d071b49d3fe519c43a1cce` |
| `cipher_intercept.h` | `162b576a4e3f767f5a6abb916c770a85` |
| `cipher_green_ctx.h` | `6dfa21012aab5a7adb954cd872eeb67c` |
| `cipher_l2_persist.h` | `d41e4b0a7f22d20a5c0e2861e526f35f` |
| `cipher_telemetry.h` | `6565fd4c7bf2bd26ce9971ec739b2397` |

`include/may13/` count: 12 → 18.

### Bare-name rewrites in destination copies

15 rewrites landed (`cipher.h: 9`, `cipher_edmd.h: 1`, `cipher_intercept.h: 1`, `cipher_green_ctx.h: 1`, `cipher_l2_persist.h: 1`, `cipher_telemetry.h: 2`). Verified: zero bare-name `cipher_*` includes remain across all 18 closure headers; 23 `may13/`-prefixed includes total.

Initial sed-multi-statement attempt mis-quoted; reapplied with `Edit` cleanly.

### Library invariant build (no .cpp added yet)

```
rc=0
md5: 56291439c7fd941ca37cf86c482948d7  (UNCHANGED — invariant held)
warning count: 18  (unchanged from baseline)
```

Invariant confirmation: no existing TU reaches into the new may13 surface. The 6 new headers are inert until a TU includes them.

### Synthetic 18-header compile smoke

```c
// /tmp/may13_compile_check.cpp
#include "may13/cipher.h"          // + 5 other new
#include "may13/cipher_10ops.h"    // + 11 other existing
int main() { return 0; }

$ g++ -O2 -Wall -Wextra -fPIC -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
      -Iinclude -c /tmp/may13_compile_check.cpp ...
rc=0
warnings: 0
errors: 0
.o size: 1304 bytes
```

All 18 headers compose cleanly with `cipher_rt_phase4` CXXFLAGS.

---

## 4. Sub-step C — Port 6 .cpp files + Makefile + build — BUILD PASS, LOADER FAIL

### .cpp port

6 byte-identical copies into `src/may13/` (md5 matches verified). All 13 bare-name `#include` rewrites applied. `grep` confirms zero bare-name `cipher_*` includes remain in the 6 .cpp.

### Makefile diff

Added 6 OBJS targets and 6 pattern rules (each `may13_*.o: src/may13/*.cpp` with the same CXXFLAGS + C++17 + `_GLIBCXX_USE_CXX11_ABI=1` as the attn substrate).

### Build result

```
rc=0
md5: 56291439 → e8981f5aa5891ee3924d9b475130d1b2  (CHANGED, expected)
warning count: 18 → 51  (delta +33 from 6 new TUs)
errors: 0
```

### Warning categorization (delta +33)

```
  31  [-Wmissing-field-initializers]   — research-grade may13 code style
  14  [-Wunused-parameter]             — function stubs awaiting wire-up
   1  [-Wunused-variable]
   1  [-Wunused-function]
   1  [-Wstringop-truncation]          — in /usr/include/.../string_fortified.h (not may13)
   1  [-Wparentheses]                  — in cipher_rt_tenant.cpp (pre-existing, surfaced
                                          by recompile, not introduced by may13 ports)
```

No latent-bug indicators. All cosmetic / unused / pre-existing.

### nm-D symbol diff (build-time)

| metric | value |
| --- | --- |
| Pre-step exported symbols | 186 |
| Post-step exported symbols | 242 |
| Symbols REMOVED | **0** |
| Symbols ADDED | 56 |

Added symbols include the expected may13 API surface: `cipher_dispatch`, `cipher_oracle_init`, `cipher_oracle_decide`, `cipher_kt_observe`, `cipher_kt_dump_json`, `cipher_get_oracle`, `cipher_layer3_init`, `cipher_chebyshev_eval`, etc.

### Build-pass-but-runtime-fail surfaced

After build, two injection paths produce divergent behavior:

**Under `LD_PRELOAD=libcipher_rt.so`** (strict loader resolution):

```
$ LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so timeout 60 python3 -c "...SDPA..."
timeout: symbol lookup error: /home/ubuntu/cipher_rt_phase4/libcipher_rt.so: undefined symbol: g_cipher
```

Loader rejects the .so on first unresolved strong-undefined.

**Under `CUDA_INJECTION64_PATH=libcipher_rt.so`** (CUDA driver swallows shim-load failures):

```
$ CUDA_INJECTION64_PATH=...libcipher_rt.so python3 -c "...SDPA..."
mean: -0.0095...
(no cipher banners; no [cipher_v2] MATMUL: substrate initialized line; shim was never active)
```

SDPA still produced finite output because torch ran on its own. The CIPHER substrate was NOT active. This makes the original "SDPA smoke PASS" in this run a **false positive**.

---

## 5. Sub-step D — Regression smoke — FAILURE MODE DETAILS

### Undefined-symbol audit on the post-step .so

```
Pre-step  undef count: 95
Post-step undef count: 113   (+18)
```

**18 new undefined symbols (delta):**

| category | count | symbols |
| --- | --- | --- |
| Strong cross-TU `U` references (loader-fatal) | 7 | `g_cipher`, `g_block_sub`, `cipher_block_sub_collect`, `cipher_liquid_record_op`, `cipher_liquid_record_passthrough`, `cipher_liquid_record_substitution`, `cipher_liquid_update_grad_ema` |
| System / libc / glibc | 7 | `cudaFree`, `cudaMalloc`, `cudaMemcpy`, `dladdr@GLIBC_2.34`, `logf@GLIBC_2.27`, `sqrtf@GLIBC_2.2.5`, `strncmp@GLIBC_2.2.5` |
| Weak (`w`) — resolve at link to defined-elsewhere | 4 | `cipher_get_edmd_pipeline`, `cipher_tls_get_gemm_ptrs`, `cipher_tls_get_gemm_shape`, `cipher_tls_relaunch` |

Only the 7 strong `U` references cause loader-fatal regression. The 7 system symbols already resolve through existing `-ldl`, `-lcuda`, `-lpthread`, libc — they are silent (just listed as undefined-in-this-.so because they live in linked-shared deps).

### Provider mapping for the 7 strong undefined symbols

| symbol | defined in (NOT in 6-file port set) | file kind |
| --- | --- | --- |
| `g_cipher` | `cipher_runtime.cpp` (L18: `CipherRuntime g_cipher = {0};`) | C++ |
| `g_block_sub` | `cipher_edmd.cpp` (L436: `CipherBlockSub g_block_sub;`) | C++ |
| `cipher_block_sub_collect` | `cipher_edmd.cpp` (L484: function def) | C++ |
| `cipher_liquid_record_op` | `cipher_liquid_state.cu` (L250) | **.cu (CUDA, nvcc required)** |
| `cipher_liquid_record_passthrough` | `cipher_liquid_state.cu` (L149) | **.cu** |
| `cipher_liquid_record_substitution` | `cipher_liquid_state.cu` (L112) | **.cu** |
| `cipher_liquid_update_grad_ema` | `cipher_liquid_state.cu` (L164) | **.cu** |

### Per-cpp reference matrix (which of the 6 ported .cpp pulls each symbol)

```
                                     dispatch  oracle  recipes  sense  s_lookup  kernel_tab
g_cipher                                  4       —       —       —        —          —
g_block_sub                              13       —       —       —        —          —
cipher_block_sub_collect                  2       —       —       —        —          —
cipher_liquid_record_op                   1       —       —       —        —          —
cipher_liquid_record_passthrough          —       2       —       —        —          —
cipher_liquid_record_substitution         —       1       —       —        —          —
cipher_liquid_update_grad_ema             —       1       —       —        —          —
```

**Concentrated:** all 7 strong undefineds come from exactly 2 of 6 ported .cpp files (`cipher_dispatch.cpp` brings 4 symbols × 20 refs; `cipher_oracle.cpp` brings 3 symbols × 4 refs). The other 4 (`recipes`, `sense`, `structural_lookup`, `kernel_table`) introduce zero cross-TU undefineds.

### Why the closure walker missed these

The walker is a `#include`-graph walker, not a symbol-graph walker. It can only see what a TU textually includes; it cannot see which extern declarations a TU actually *references*. The brief's risk note in §6 of the PARTIAL doc anticipated this:

> *"the 5 .cpp files may declare symbols whose definitions live in cipher_dispatch.cpp. Closure walker did not run the linker; would surface at Step E build."*

The actual failure was the converse: ported .cpp reference symbols defined in *unported* sources. Same class of risk.

### Other regression signals

**CP 5.4 isolation suite:** 15/15 PASS. Substrate-via-`/dev/cipher` path unaffected (the kmod tests don't load libcipher_rt.so).

**SDPA trampoline accounting:** Cannot be verified at this stop point because the shim isn't loading. The Week 2 Step 1 LP-2 refactor invariant (`handled=0 passthrough=1`) is not exercised by the current broken .so.

---

## 6. Sub-step E — Commit + tag — NOT EXECUTED

Per brief STOP discipline. Working tree state:

```
$ git -C /home/ubuntu/cipher_rt_phase4 status --short
?? include/may13/cipher.h
?? include/may13/cipher_edmd.h
?? include/may13/cipher_intercept.h
?? include/may13/cipher_green_ctx.h
?? include/may13/cipher_l2_persist.h
?? include/may13/cipher_telemetry.h
?? src/may13/                       (6 .cpp files)
 M Makefile
... (object files in repo root: may13_*.o)
```

No commit. No tag `week-2-step-2-may13-ports-extended-closure` created. `week-2-step-1-lp2-sdpa-refactor` remains the in-place anchor.

---

## 7. Rollback path

To restore working-baseline runtime (libcipher_rt.so loadable):

```
cd /home/ubuntu/cipher_rt_phase4
git restore Makefile
rm -rf include/may13/{cipher.h,cipher_edmd.h,cipher_intercept.h,cipher_green_ctx.h,cipher_l2_persist.h,cipher_telemetry.h}
rm -rf src/may13/
rm -f may13_*.o
make clean && make
# verify libcipher_rt.so md5 returns to 56291439...
```

Working tree is untracked (no commit), so `git status` will show clean after the rm + restore above.

---

## 8. Telemetry on disk

- `/tmp/week2_step2_v2/closure_walk_extended.txt` — sub-step A walk output
- `/tmp/week2_step2_v2/post_substep_b_build.log` — sub-step B build (md5 56291439 unchanged)
- `/tmp/week2_step2_v2/may13_compile_check.log` — sub-step B synthetic include-only smoke (rc=0)
- `/tmp/week2_step2_v2/post_substep_c_build.log` — sub-step C build (md5 e8981f5a, +33 warnings, rc=0)
- `/tmp/week2_step2_v2/nm_post_step2_v2.txt` — post-step nm dump
- `/tmp/week2_step2_v2/cp54.out` — CP 5.4 isolation (15/15 PASS)
- `/tmp/week2_step2_v2/sdpa_smoke.log` — false-positive SDPA smoke (no cipher banners)
- `/tmp/week2_step2_v2/sdpa_full.log` — second SDPA smoke (same false positive)
- `/tmp/u_pre.txt`, `/tmp/u_post.txt`, `/tmp/new_undef_syms.txt` — undefined-symbol audit
- `/tmp/may13_compile_check.cpp`, `/tmp/may13_compile_check.o` — synthetic check artifacts
- `/tmp/closure_walker_extended.py` — closure walker
- `/tmp/apply_cpp_rewrites.py` — .cpp rewrite applier

---

## 9. Discipline notes

- No commit. No tag. No anchor rotation.
- No mitigation in this document — only failure-mode data + the surface for adjudication.
- The brief's "closure walker can't predict cross-TU symbol references" caveat anticipated exactly this failure class; this is the predicted-class regression.
- Rollback path documented in §7 to restore loadable baseline if the user chooses to abandon and revisit scope.

---

## 10. Awaiting

User adjudication on the structural path. Three classes of resolution exist, surfaced without recommendation:

- **Extend further** — port `cipher_runtime.cpp` + `cipher_edmd.cpp` (provide g_cipher, g_block_sub, cipher_block_sub_collect) + `cipher_liquid_state.cu` (provide 4 record-fns; requires nvcc build infrastructure). Each will have its own closure walk and may surface further-order deps.

- **Stub-and-defer** — provide local empty definitions of the 7 symbols inside `cipher_rt_phase4`; the 6 .cpp link as inert TUs whose static-init does nothing real until Step 6 wires them. Risks: may13 .cpp expects real behavior from these stubs; subtle silent runtime bugs.

- **Narrow scope** — port only the 4 .cpp files that introduce zero cross-TU undefineds (`recipes`, `sense`, `structural_lookup`, `kernel_table`); defer `cipher_dispatch.cpp` and `cipher_oracle.cpp` until their dep closure is built. The load-bearing matmul dispatch entry stays unported.

- **Abandon retry** — rollback (§7) and revisit Week 2 scope-lock; possibly fold may13 source ports into a later week with different design constraints.

The decision is structural and out of scope for me.
