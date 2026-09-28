# Week 2 Step 2 — Symbol-Closure Audit

**Status:** read-only diagnostic; no source-tree changes
**Date:** 2026-05-20
**Trigger:** user question "how come you didn't expose this while we were doing initial audit"
**Purpose:** symbol-level analysis the prior `#include`-only closure walker could not do; produces the audit artifact that should have existed before `WEEK_2_SCOPE_LOCK.md` was written

---

## 0. Method

For each of the 73 sources in `cipher-may13-evidence/src/` (67 `.cpp` + 6 `.cu`), the evidence tree ships a corresponding prebuilt `.o` at `cipher-may13-evidence/build/`. Reading these directly with `nm -C` is faster and more reliable than rebuilding — it gives the authoritative defined/undefined symbol surface each TU produces.

The analyzer (`/tmp/symbol_closure.py` + `/tmp/symbol_closure_v3.py`) builds:

- **Provider map** `symbol → [TUs defining it]` (3,383 strong defs, 44 weak defs)
- **Consumer map** `TU → [symbols it references]` (1,121 strong U refs total)
- **System-extern filter** (libc/libm, CUDA runtime, libstdc++, libssl, dlfcn, c10/ATen — everything that resolves through `.so` deps the cipher_rt_phase4 linker already pulls in)
- **Transitive closure walker** (start from any subset; follow U → provider → repeat to fixed point)
- **Greedy stub picker** (at each step pick the symbol whose stubbing maximally shrinks closure)

---

## 1. Why the prior audits missed it

The pre-flight machinery I had on hand:

| audit | granularity | catches | missed |
| --- | --- | --- | --- |
| Wave 5 §5.5 verification sweep | intent-vs-impl | semantic mismatch (LP-2, LP-7) | linkage |
| WEEK_2_SCOPE_LOCK.md | file list | "do these 6 exist?" | linkage |
| Closure walker (Step 2 PARTIAL + v2 sub-step A) | `#include` graph | new transitive headers | symbol refs to definitions outside `#include` graph |

None of those audits operated at the symbol level. The closure walker is a `cpp -M` style tool — it sees what files a TU `#include`s, not what extern symbols a TU *references*. A `.cpp` can use `extern Foo g_thing;` from a header and then call `g_thing.x` — the walker resolves the header (good), but cannot see that `g_thing`'s definition lives in some *other* `.cpp` outside the chosen port set.

The brief's own §6 risk note in `WEEK_2_STEP_2_RESULT.md` (Step 2 PARTIAL) acknowledged this exact class — "Closure walker did not run the linker; would surface at Step E build" — but treated it as something we'd discover at link time. The link itself passed (`-shared` accepts undefined-strong refs); the **loader** is what rejected the .so. So even "let the linker tell us" wouldn't have caught it without an explicit `-Wl,--no-undefined`.

This was a missing audit, not a bug. The audit is `symbol_closure.py`. It now exists.

---

## 2. Headline numbers

| start set | closure size | added files | external strong undef | `.cu` files in closure |
| --- | ---:| ---:| ---:| ---:|
| `cipher_marlin_src.cpp` (leaf check) | 1 | 0 | 0 | 0 |
| **Scope-4** (4 already-clean .cpp) | **4** | **0** | 3 (sys only) | 0 |
| `cipher_oracle.cpp` alone | 9 | 8 | 11 (sys) | 1 |
| **Scope-5** (4 clean + oracle) | **11** | **6** | 11 (sys) | 1 |
| **Scope-6** (brief's original) | **40** | **34** | 24 (sys) | 4 |
| `cipher_dispatch.cpp` alone | 40 | 39 | 24 (sys) | 4 |
| Entire evidence/src/ (sanity) | 73 | 0 | 33 (sys) | 6 |

**The 40-file closure for the brief's Scope-6 was driven almost entirely by one file** — `cipher_dispatch.cpp`. Its 1-file standalone closure equals the full Scope-6 closure (40), meaning the other 5 .cpp files add nothing new transitively.

`cipher_oracle.cpp` is the second culprit (small but real — 8-file standalone closure including the .cu).

The other 4 .cpp (`recipes`, `sense`, `structural_lookup`, `kernel_table`) form a self-contained subgraph with **zero in-tree cross-TU references**.

### What "39 transitive files for dispatch.cpp" pulls in

The 39 cover most of the runtime: `cipher_runtime.cpp`, `cipher_edmd.cpp`, `cipher_lnn.cpp`, `cipher_intercept.cpp`, `cipher_intercept_cudart.cpp`, plus operational support (`cipher_carbon.cpp`, `cipher_comply.cpp`, `cipher_guard.cpp`, `cipher_shield.cpp`, `cipher_hibernate.cpp`, `cipher_loop.cpp`, `cipher_pipeline.cpp`, `cipher_pulse.cpp`, `cipher_receipt.cpp`, `cipher_sustain.cpp`, `cipher_telemetry.cpp`, `cipher_thermostat.cpp`, `cipher_topology.cpp`, `cipher_trace.cpp`, `cipher_volt.cpp`), plus 4 `.cu` files requiring nvcc.

This is the entire CIPHER "brain". Porting Scope-6 the literal way means porting the entire brain.

---

## 3. The high-leverage discovery: single-stub closure shrinkage

The greedy stub picker — at each step, find the symbol whose stubbing maximally shrinks closure — produces an enormous lever:

```
Step 1: stub 'g_cipher'                          → closure 40 → 12 files (-28)
Step 2: stub 'cipher_flow_recorder_observe'      → closure 12 → 11 files  (-1)
Step 3: stub 'cipher_flow_substitute_consider'   → closure 11 → 10 files  (-1)
Step 4: stub 'cipher_flow_patterns_check'        → closure 10 →  9 files  (-1)
```

**4 stubs at the `cipher_rt_phase4` boundary collapse the closure from 40 files to 9.**

The 9-file closure with 4 boundary stubs:

| file | role |
| --- | --- |
| `cipher_dispatch.cpp` | (start) load-bearing matmul dispatch entry |
| `cipher_oracle.cpp` | (start) |
| `cipher_recipes.cpp` | (start) |
| `cipher_sense.cpp` | (start) |
| `cipher_structural_lookup.cpp` | (start) |
| `cipher_kernel_table.cpp` | (start) |
| `cipher_intercept_cudart.cpp` | transitive — referenced by dispatch.cpp |
| `cipher_liquid_state.cu` | transitive — provides 4 `liquid_record_*` |
| `cipher_persist.cpp` | transitive — referenced by oracle.cpp's recovery path |

vs. the alternative (port-everything) closure of 40.

### Why `g_cipher` is the dominant lever

`g_cipher` is the global `CipherRuntime` singleton (defined in `cipher_runtime.cpp`). cipher_runtime.cpp depends on (and pulls in transitively): cipher_runtime → cipher_carbon, cipher_comply, cipher_guard, cipher_loop, cipher_pipeline, cipher_pulse, cipher_receipt, cipher_shield, cipher_sustain, cipher_thermostat, cipher_topology, cipher_trace, cipher_volt, ... — 28 files in the "ops support" wing of the runtime.

cipher_dispatch.cpp only uses `g_cipher` for `.initialized` check + `.liquid` field. If we provide a local definition `CipherRuntime g_cipher = {0};` inside cipher_rt_phase4 (plus matching struct layout from cipher.h), the bridge to cipher_runtime.cpp is severed and 28 files vanish from closure.

The other 3 stubs (flow_recorder_observe / flow_substitute_consider / flow_patterns_check) sever the dependency on `cipher_flow_*.cpp` files — these are research-grade telemetry hooks that cipher_dispatch.cpp calls but that have no Phase-5/6 wiring yet (they would no-op or return null in our boundary).

---

## 4. ODR / linkage caveats surfaced during the audit

Two facts to flag:

- **125 ODR conflicts in evidence/src/** — same strong symbol defined in multiple TUs. All 125 are `.LC*` (compiler-generated string-literal labels with local linkage); the symbols are not actually exported, so this is benign. Not the same class as the cross-TU strong-undef.

- **44 weak definitions across the tree** — `inline` functions and template specializations. These are fine: the loader merges them. None of the 7 missing-strong symbols our v2 attempt hit are in this set.

- **External strong undef count grows monotonically with closure size.** The Scope-4 closure needs only 3 system externs (libc stack-check/printf); cipher_dispatch.cpp's closure needs 24 (adds: cuda runtime, openssl HMAC/EVP, gmtime_r, sincosf, etc.). cipher_rt_phase4's existing link flags already cover most of these via `-lcuda`, `-lcupti`, `-lcrypto`, `-lpthread`, `-ldl`, `-lc10`. Two probable adds: `-lm` (for `fmaxf`/`fminf`/`sincosf`/`tanhf`/`fmod`) and `_FORTIFY_SOURCE=0` or `-fno-stack-protector` (for `__*_chk` and `__stack_chk_fail`, if minimal port). These are routine.

---

## 5. The four adjudication options, re-priced with real numbers

Updating the v2 PARTIAL doc's option table with concrete closure sizes (`.cpp` + `.cu` to be ported):

| option | start set | closure | new `.cu` (nvcc cost) | boundary stubs needed in `cipher_rt_phase4` | total LOC port |
| --- | --- | ---:| ---:| ---:| ---:|
| **A — Extend further (literal)** | brief's 6 | **40 files** | 4 | 0 | very large (~15K+ LOC) |
| **B — Stub-and-defer** *(previously vague)* | brief's 6 + 4 boundary stubs | **9 files** | 1 (cipher_liquid_state.cu) | **4** (`g_cipher`, 3 flow hooks) | ~3.6K LOC + 4 small stubs |
| **C — Narrow (port only the 4 clean .cpp)** | 4 already-clean | **4 files** | 0 | 0 | ~1.6K LOC |
| **C+ — Narrow + oracle** | 5 (drop only dispatch) | **11 files** | 1 (cipher_liquid_state.cu) | **6** (oracle's deps) | ~3.0K LOC + 6 small stubs |
| **D — Abandon retry** | rollback | 0 | 0 | 0 | 0 |

Option B was previously surfaced as "stub-and-defer" but without numbers — the audit shows it costs only **4 stub symbols** to land the brief's full Scope-6 with 9-file closure. That is dramatically more tractable than Option A's "port the entire brain" path.

### Per-stub specifics (Option B)

The 4 stub symbols and what they would need to look like in cipher_rt_phase4:

```c
// cipher_rt_may13_stubs.cpp  (new file, ~30 lines)
#include "may13/cipher.h"       // for CipherRuntime layout
extern "C" CipherRuntime g_cipher = {0};   // zero-init; .initialized stays false

// No-op flow hooks — these are research telemetry; cipher_rt_phase4 has its
// own audit pipeline. Returning passthrough/default until Phase 6+ wiring.
extern "C" int cipher_flow_recorder_observe(const void* call) { (void)call; return 0; }
extern "C" int cipher_flow_substitute_consider(const void* call) { (void)call; return 0; }
extern "C" int cipher_flow_patterns_check(const void* call) { (void)call; return 0; }
```

(Exact signatures need to be lifted from cipher.h / cipher_classify.hpp / cipher_flow_*.h — the closure walker has those headers available since they're in the existing 18-header may13/.)

### One real cost remaining: cipher_liquid_state.cu

Even with the 4 stubs, `cipher_liquid_state.cu` stays in closure (it provides 4 `cipher_liquid_record_*` functions used by `cipher_oracle.cpp` and `cipher_dispatch.cpp`). This means nvcc setup IS required for Option B and C+, but for ONE .cu file rather than 4.

Possible fifth stub:

```
Step 5 (hypothetical): stub 4 cipher_liquid_record_* → closure 9 → 8 files (-1)
```

That would drop cipher_liquid_state.cu and eliminate the nvcc requirement entirely. The 4 functions are statistics-collection (record_op / record_passthrough / record_substitution / update_grad_ema). They could be stubbed as no-ops if the substrate just needs the function to exist and accept calls.

If the user is willing to defer the liquid-state stats collection, Option B becomes pure-`.cpp` (no nvcc) with 8 source files + 8 stubs (4 prior + 4 new).

---

## 6. What this means for prior decisions

- The Step 2 PARTIAL doc's "extend closure" option (presented as the favored Wave-5-style continuation) is now visible as the **40-file-port** path. Worth costing before committing to it.

- The Step 2 v2 PARTIAL doc's "stub-and-defer" option was correctly identified but I undersold its tractability (presented as vague "risks: subtle silent runtime bugs"). The greedy stub picker shows **4 stubs is the entire ask** — not dozens. That makes Option B much more competitive than the v2 doc implied.

- The Step 2 PARTIAL doc's "narrow scope" option (Option C, 4 .cpp) is **fully validated** by the audit — 0 transitive files, 0 in-tree externs, only routine libc externs. It's the safest path if the constraint is "ship something this week."

- The Step 2 PARTIAL doc's "abandon retry" option (Option D) is unaffected by the audit.

---

## 7. Artifacts (read-only; not in any commit)

- `/tmp/symbol_closure.py` — symbol-graph builder + closure walker (v1)
- `/tmp/symbol_closure_v2.py` — stub-impact analysis per start set
- `/tmp/symbol_closure_v3.py` — greedy stub picker
- `/tmp/week2_step2_v2/symbol_closure_report.txt` — v1 output
- `/tmp/week2_step2_v2/symbol_closure_v2_report.txt` — v2 output
- `/tmp/week2_step2_v2/symbol_closure_v3_report.txt` — v3 output

All tools are reusable for future weeks — any time the brief asks to "port these N source files", running symbol_closure.py against the proposed start set produces the full closure picture in seconds. Recommend including this as a step in the standard scope-lock workflow going forward.

---

## 8. Discipline notes

- No source-tree edits. No commit on `cipher_rt_phase4` (working tree still dirty from v2 attempt — rollback path in `WEEK_2_STEP_2_V2_RESULT.md` §7 still applies).
- No recommendation on which option to take. The audit just re-prices the options the v2 PARTIAL surfaced.
- The new diagnostic tool (`symbol_closure.py`) is the artifact that should have been built before `WEEK_2_SCOPE_LOCK.md`. It now exists and can be re-run at any point.

---

## 9. Awaiting

User adjudication on which of the (now properly-priced) options to take:

- Option **A** (extend, 40-file port + 4 nvcc files)
- Option **B** (stub at boundary, 9-file port + 4 stubs, 1 nvcc file)
- Option **B′** (same as B but stub 4 more liquid_record_* → 8-file pure-`.cpp`, no nvcc)
- Option **C** (narrow to 4 clean .cpp, 0 stubs, 0 nvcc)
- Option **C+** (narrow to 5 .cpp incl. oracle, 11-file + 6 stubs, 1 nvcc)
- Option **D** (abandon retry, rollback)
