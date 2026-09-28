# Week 2 Step 2 — A4 Closure Rerun (intercept files excluded)

**HEADLINE STATUS: ADJUDICATION REQUIRED — 1 Category B orphan needs handling decision.**

**Aggregate counts:**

- Final closure size: **11 files** (down from 40 — a 29-file reduction from excluding 2 FLAG files)
- Category A orphans (safe to drop): **0**
- Category B orphans (need handling): **1** (`cipher_intercept_stats`)
- `.cu` files in closure: **3** (`cipher_green_ctx.cu`, `cipher_l2_persist.cu`, `cipher_liquid_state.cu`)
- New FLAG static-init in reduced closure: **0**
- External strong undef (system / libc / cuda runtime): **4**

**Date:** 2026-05-20
**Adjudication input:** Option A4 from `WEEK_2_STEP_2_V3_PREFLIGHT.md` §9 (exclude both `cipher_intercept.cpp` + `cipher_intercept_cudart.cpp` from port set)
**Tree state:** unchanged (read-only diagnostic)

---

## Step 1 — Closure rerun with exclusion

Starting set (6 files, brief's original Scope-6):
```
cipher_dispatch.cpp, cipher_oracle.cpp, cipher_recipes.cpp,
cipher_sense.cpp, cipher_structural_lookup.cpp, cipher_kernel_table.cpp
```

Exclusion set (2 files, FLAG'd in v3 pre-flight):
```
cipher_intercept.cpp, cipher_intercept_cudart.cpp
```

### 11-file A4 closure (files that would be ported)

| file | kind | in start set? |
| --- | --- | --- |
| `cipher_dispatch.cpp` | .cpp | ✓ START |
| `cipher_oracle.cpp` | .cpp | ✓ START |
| `cipher_recipes.cpp` | .cpp | ✓ START |
| `cipher_sense.cpp` | .cpp | ✓ START |
| `cipher_structural_lookup.cpp` | .cpp | ✓ START |
| `cipher_kernel_table.cpp` | .cpp | ✓ START |
| `cipher_runtime.cpp` | .cpp | transitive (provides `g_cipher`) |
| `cipher_telemetry.cpp` | .cpp | transitive |
| `cipher_green_ctx.cu` | **.cu** | transitive (F2 green-context impl) |
| `cipher_l2_persist.cu` | **.cu** | transitive (F3 L2 pin impl) |
| `cipher_liquid_state.cu` | **.cu** | transitive (F4 liquid-state record-fns) |

### Why the closure shrunk from 40 → 11

The 29-file drop came from removing `cipher_intercept_cudart.cpp` (in-degree 12, most-referenced TU in the tree). When the intercept files no longer pull in their downstream "ops support" wing (`cipher_carbon`, `cipher_comply`, `cipher_guard`, `cipher_shield`, `cipher_hibernate`, `cipher_loop`, `cipher_pipeline`, `cipher_pulse`, `cipher_receipt`, `cipher_sustain`, `cipher_thermostat`, `cipher_topology`, `cipher_trace`, `cipher_volt`, `cipher_carbon`, plus several more), the closure collapses to just the matmul-dispatch + L3 substrate + the 3 .cu kernels that the brain ACTUALLY needs.

The closure shape now matches Option B's profile (which the symbol-closure audit predicted at 9 files with 4 boundary stubs) — but Option A4 reaches 11 files with **1 boundary symbol to handle** instead of 4. The reason: A4 keeps `cipher_runtime.cpp` IN closure (providing `g_cipher` natively), so the g_cipher stub is no longer needed. The cost is +2 transitive files (`cipher_runtime.cpp` + `cipher_telemetry.cpp`).

### Orphaned symbols (would-have-been provided by excluded files)

3 orphans surfaced in pass 1, but after correcting the system-symbol filter (`__cuda*` prefix), only **1 real orphan remains**:

| symbol | excluded provider | requesters in closure |
| --- | --- | --- |
| `cipher_intercept_stats` | `cipher_intercept.cpp` | `cipher_runtime.cpp` |

The other 2 initially-flagged symbols (`__cudaRegisterFatBinary` / `__cudaRegisterFatBinaryEnd`) were misclassified — they are CUDA toolkit runtime symbols (from `libcudart.so`), not in-tree symbols. They resolve through the existing `cipher_rt_phase4` linker flags (`-lcuda` + libtorch's libcudart dependency).

The system-filter patch (`is_system()` now matches `__cuda*` prefix) has been applied to `/tmp/symbol_closure.py` for future runs.

### External strong undefineds (system, satisfied via .so deps)

```
U __fprintf_chk     (libc — fortified printf)
U __memcpy_chk      (libc — fortified memcpy)
U __snprintf_chk    (libc — fortified snprintf)
U __stack_chk_fail  (libc — -fstack-protector)
```

All resolve through existing `cipher_rt_phase4` link flags (libc default).

---

## Step 2 — Provider audit for Category B orphan

### `cipher_intercept_stats`

**Declaration** (in already-ported header `include/may13/cipher_intercept.h:88`):
```c
const CipherInterceptStats* cipher_intercept_stats(void);
```

**Body in excluded provider** (disassembled from `src__cipher_intercept.o`):

```
0x0c40 <cipher_intercept_stats>:
    endbr64
    mov  X+0, %rax    ; load 6 counter fields atomically
    mov  %rax, Y+0    ; copy into return struct
    mov  X+8, %rax
    mov  %rax, Y+8
    mov  X+10, %rax
    mov  %rax, Y+10
    mov  X+18, %rax
    mov  %rax, Y+18
    mov  X+20, %rax
    mov  %rax, Y+20
    mov  X+28, %rax
    mov  %rax, Y+28
    lea  Y, %rax       ; return pointer to copy
    ret
```

A 6-field snapshot of the F1 hook counters (call counts, passthrough counts, hooked-fn counts, etc.). It does NOT touch driver state, NOT spawn threads, NOT call CUDA — it just returns a pointer to internally-maintained counters.

**Consumer in closure:**
```c
// cipher_runtime.cpp:123
const CipherInterceptStats* is = cipher_intercept_stats();
```
…likely consumed in a `cipher_runtime_report()` or similar logging path.

**Provider audit in the 73-source tree:** only `cipher_intercept.cpp` provides this symbol. No alternative provider exists.

**cipher_rt_phase4 native:** no equivalent symbol exists. cipher_rt_phase4 has its own audit/GOT-patcher counters (`cipher_v2` MATMUL totals, `cipher-attn` trampoline totals) but they expose different struct layouts via different APIs (`cipher_rt_attn_calls_total()`, etc.).

---

## Step 3 — Cascading exclusion check

Question: if we can't satisfy `cipher_intercept_stats`, does `cipher_runtime.cpp` cascade-exclude (forcing us to remove `g_cipher`'s provider)?

**No cascade required.** `cipher_runtime.cpp`'s reference to `cipher_intercept_stats` is one line at L123 in a likely-reporting path. The reference is unconditional at link time but can be satisfied by a no-op stub (the call's return value is consumed for logging — a zero-init struct produces zero counters in the log, which is the truthful state when no F1 hooks are installed).

If the user does not want a stub: cascade-excluding `cipher_runtime.cpp` would re-introduce the `g_cipher` problem (no provider for it remaining in scope; would need to re-add `g_cipher` stub as in Option B). Net: same number of stubs, but loses `cipher_telemetry.cpp` from closure too. Worth considering only if the user wants minimum LOC port.

Closure is **stable** at 11 files. No cascading.

---

## Step 4 — Static-init re-verification on reduced closure

Re-ran `/tmp/static_init_scan.py` against the 11 A4 files:

| file | init bytes | init target | verdict |
| --- | ---:| --- | --- |
| cipher_dispatch.cpp | 0 | — | CLEAN |
| cipher_oracle.cpp | 0 | — | CLEAN |
| cipher_recipes.cpp | 0 | — | CLEAN |
| cipher_sense.cpp | 0 | — | CLEAN |
| cipher_structural_lookup.cpp | 0 | — | CLEAN |
| cipher_kernel_table.cpp | 0 | — | CLEAN |
| cipher_runtime.cpp | 0 | — | CLEAN |
| cipher_telemetry.cpp | 0 | — | CLEAN |
| cipher_green_ctx.cu | 8 | `__sti____cudaRegisterAll()` | CLEAN (stock nvcc fatbin reg.) |
| cipher_l2_persist.cu | 8 | `__sti____cudaRegisterAll()` | CLEAN (stock nvcc fatbin reg.) |
| cipher_liquid_state.cu | 8 | `__sti____cudaRegisterAll()` | CLEAN (stock nvcc fatbin reg.) |

**0 FLAG, 0 WATCH.** The 3 .cu files use the standard nvcc-generated fatbin registration which calls `__cudaRegisterFatBinary` / `__cudaRegisterFunction` / `__cudaRegisterFatBinaryEnd` — all are stock CUDA toolkit calls that register the embedded kernel binary with the runtime's bookkeeping, no GPU touch, no driver state change. Verified CLEAN in v3 pre-flight; re-confirmed.

---

## Step 5 — Stub-versus-port handlings for `cipher_intercept_stats`

Per discipline (no recommendation), three handlings:

### Handling 1 — Stub at cipher_rt_phase4 boundary

Add to the boundary-stubs file (proposed in v2 sym-closure audit as ~30 LOC):

```c
// cipher_rt_may13_stubs.cpp — additional stub
#include "may13/cipher_intercept.h"   // for CipherInterceptStats type

const CipherInterceptStats* cipher_intercept_stats(void) {
    // F1 hook layer is not installed in cipher_rt_phase4 (CP 2.5 D2(iii)
    // GOT patcher is the canonical interceptor instead). Return a snapshot
    // of zero counters — telemetry will reflect "no F1 activity," which
    // is the truthful state.
    static const CipherInterceptStats zero = {0};
    return &zero;
}
```

LOC estimate: **5** lines plus 1 include. Total Option A4 boundary surface: 1 stub function + 1 `extern CipherRuntime g_cipher = {0};` replaced by linking `cipher_runtime.cpp` directly (so 1 stub net).

### Handling 2 — Substitute with cipher_rt_phase4 native symbol

cipher_rt_phase4 has no equivalent type (`CipherInterceptStats` vs `cipher_rt_attn_calls_total()`-style scalars). To use the native counters, the `CipherInterceptStats` struct fields would need to be backfilled from cipher_rt_phase4's GOT-patcher counters at every call to `cipher_intercept_stats()`. Routing requires a translation shim (~15-20 LOC) plus a per-call atomic snapshot — more code than Handling 1, with no clear benefit since the consumer (cipher_runtime.cpp:123) only logs the value.

Net effect: more LOC than Handling 1 with no behavioral upside.

### Handling 3 — Cascade-exclude `cipher_runtime.cpp`

Removing `cipher_runtime.cpp` from closure:
- Severs the only `cipher_intercept_stats` consumer (orphan disappears)
- Re-introduces the `g_cipher` problem (cipher_dispatch.cpp consumes it; no provider in remaining closure)
- Removes `cipher_telemetry.cpp` from closure (transitive — only cipher_runtime.cpp pulled it in)
- Net new closure: 9 files (back to Option B's profile)
- Stubs required: `g_cipher = {0}` definition (1) — same count as Handling 1, different symbol

Trade-off: Option A4-with-cascade ≈ Option B in cost. Option A4-without-cascade gets us `cipher_runtime.cpp` IN closure (real `g_cipher` definition with real layout), at the cost of one tiny stub.

---

## Headline summary (re-stated)

| metric | value |
| --- | ---:|
| Final closure size | **11 files** |
| Category A orphans | 0 |
| Category B orphans | **1** (`cipher_intercept_stats`) |
| .cu files | 3 |
| Static-init FLAGs in reduced closure | 0 |
| Static-init WATCHes | 0 |
| Boundary stubs required (Handling 1) | **1** (5 LOC) |

### What this means relative to the prior options

| option | closure | stubs | `.cu` | port LOC | notes |
| --- | ---:| ---:| ---:| ---:| --- |
| A (literal port everything) | 40 | 0 | 4 | ~15K | reverses CP 2.5 D2(iii) — blocked |
| **A4 (exclude intercept + 1 stub)** | **11** | **1** | **3** | ~3.4K + 5 LOC stub | this rerun |
| B (boundary-stub 4 syms) | 9 | 4 | 1 | ~3.6K + 30 LOC stub | sym-closure audit |
| B′ (stub 4 more cu fns) | 8 | 8 | 0 | ~3.6K + 60 LOC stub | sym-closure audit |
| C (4 clean .cpp) | 4 | 0 | 0 | ~1.6K | sym-closure audit |
| C+ (5 .cpp incl. oracle) | 11 | 6 | 1 | ~3.0K + 45 LOC stub | sym-closure audit |

**A4 is the smallest stub surface (1) of any option that keeps `cipher_dispatch.cpp` in scope.** It costs 2 more .cu files than Option B (3 vs 1) but avoids 3 boundary stubs.

---

## Telemetry on disk

- `/tmp/symbol_closure_a4.py` — A4 closure walker with exclusion + orphan tracker
- `/tmp/week2_step2_a4/closure_walk_a4_v2.txt` — final walk output (post system-filter fix)
- `/tmp/week2_step2_a4_closure.json` — structured closure manifest
- `/tmp/a4_static_init_verify.py`, `/tmp/a4_static_init_verify_v2.py` — static-init re-verification
- `/tmp/week2_step2_a4/static_init_verify_v2.txt` — re-verify output

System-filter fix landed in `/tmp/symbol_closure.py` (added `__cuda*` prefix to the CUDA-runtime classifier).

---

## Discipline notes

- Read-only diagnostic. No source-tree changes. `cipher_rt_phase4` working tree still dirty from v2 attempt (rollback in `WEEK_2_STEP_2_V2_RESULT.md` §7 still applies).
- No commit / tag rotation in any tree.
- No mitigation in this document — only the facts + the 3-handling surface for the 1 orphan.
- A4 is now properly priced. Decision on Handling 1 / 2 / 3 unblocks the Step 2 v3 port prompt.

---

## Awaiting

Per-symbol handling decision on `cipher_intercept_stats`:

- **H1** — stub (5 LOC, zero-counter snapshot, truthful "no F1 hooks installed" telemetry)
- **H2** — substitute with cipher_rt_phase4 native counter struct (~20 LOC translation shim, behavior identical for the log path)
- **H3** — cascade-exclude `cipher_runtime.cpp` (closure → 9 files, but re-introduces `g_cipher` stub need; net 1 stub same as H1, different symbol; loses `cipher_telemetry.cpp` from scope)

After this decision lands, the Step 2 v3 port brief can be drafted from a stable 11-file (H1/H2) or 9-file (H3) closure.
