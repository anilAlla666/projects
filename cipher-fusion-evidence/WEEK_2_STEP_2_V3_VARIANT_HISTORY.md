# Week 2 Step 2 v3 — Variant History Audit (D3)

**HEADLINE STATUS: ROOT-CANONICAL → D1 RECOMMENDED.**

The Makefile uses ROOT; the prebuilt `build/cipher_dispatch.o` is **byte-identical** to a fresh compile of the ROOT variant (md5 `022c5176943dad86fbcf4dc8daeb10e5` exact match); SRC fresh compile produces a different binary. No ambiguity remains.

**Aggregate counts:**

| metric | value |
| --- | ---:|
| Total variant pairs found | **2** (`cipher_dispatch.cpp`, `cipher_oracle.cpp`) |
| Files with last commit > 30 days ago for one variant only | **0** (git history is non-informative — see §3) |
| Files used by `libcipher_rt.so` build path | **ROOT** (both pairs); src/ versions explicitly filtered |
| Prebuilt `.o` byte-identical to fresh-compile of ROOT | **YES, both pairs** |
| Prebuilt `.o` byte-identical to fresh-compile of SRC | **NO, both pairs** |

**Date:** 2026-05-20
**Read-only diagnostic.** No source modifications.

---

## Step 1 — Variant pair survey

Survey at evidence root + `src/`:

```bash
$ cd /home/ubuntu/cipher-may13-evidence && ls *.cpp *.cu *.hpp *.h
cipher_dispatch.cpp
cipher_oracle.cpp

$ for f in cipher_dispatch.cpp cipher_oracle.cpp; do
    [ -f "src/$f" ] && echo "PAIR: $f"
  done
PAIR: cipher_dispatch.cpp
PAIR: cipher_oracle.cpp
```

**Only 2 variant pairs in the entire tree.** No `.h` / `.hpp` / `.cu` duplicates at root level. The `src/` directory has 67 `.cpp` + 6 `.cu` = 73 files; root has 2 `.cpp`. The 2 root files are deliberately placed there to be discovered as separate from `src/*` (the Makefile relies on this).

---

## Step 2 — Per-pair diff metrics

### `cipher_dispatch.cpp`

| | ROOT (`cipher_dispatch.cpp`) | SRC (`src/cipher_dispatch.cpp`) |
| --- | ---:| ---:|
| lines | 543 | 616 |
| md5 | `59271a09f88ec5a15f6d7bfb12238a44` | `59d805306298aa873e0f73cdbf0a9321` |
| mtime | 2026-05-13 04:27:19 | 2026-05-13 04:27:19 |
| diff line count | — | 408 (substantial divergence) |
| g++ compile rc | 0 | 0 |
| `.o` symbol count | 56 | 64 |
| common symbols | 47 | 47 |

**Genuinely divergent** — each defines symbols the other doesn't:

- **Only in ROOT:** `cipher_edmd_live_collect`, `cipher_tls_get_gemm_types`, plus calls to `cipher_oracle_bill_gemm` / `cipher_oracle_bill_nongemm` / `cipher_oracle_billing_report` (consumed from ROOT cipher_oracle.cpp)
- **Only in SRC:** `cipher_block_sub_collect`, `g_block_sub`, plus references to `cudaMalloc`/`cudaFree`/`cudaMemcpy` (CUDA-touching), `memset`, `sqrtf`, `s_cache_hit_count`, `s_cached_output`, `s_relaunch_count`

The two variants represent **parallel implementations of the same EDMD/block_sub responsibility area** with different mechanisms:

- ROOT path uses `cipher_edmd_live_collect` (called from ROOT cipher_dispatch.cpp) + billing functions in ROOT cipher_oracle.cpp.
- SRC path uses `cipher_block_sub_collect` (defined in `cipher_edmd.cpp`) + a global `g_block_sub` struct (also in `cipher_edmd.cpp`) + relaunch / cache-hit instrumentation.

### `cipher_oracle.cpp`

| | ROOT (`cipher_oracle.cpp`) | SRC (`src/cipher_oracle.cpp`) |
| --- | ---:| ---:|
| lines | 539 | 451 |
| md5 | `3b2c0d6830d54cf26e436c13676895e8` | `1b5a175c02ae7d8de16f740253d9ed92` |
| mtime | 2026-05-13 04:27:19 | 2026-05-13 04:27:19 |
| diff line count | — | 97 |
| g++ compile rc | 0 | 0 |
| `.o` symbol count | 63 | 52 |
| common symbols | 52 | 52 |

**ROOT is a strict superset of SRC.** Only-in-ROOT additions:

- `cipher_oracle_bill_gemm`
- `cipher_oracle_bill_nongemm`
- `cipher_oracle_billing_report`
- `cipher_oracle_update_mfu`

These 4 billing/MFU-tracking functions are present in ROOT, absent in SRC. ROOT cipher_dispatch.cpp's symbol table shows undefined refs to all 4 — i.e., ROOT cipher_dispatch.cpp calls into the billing functions that ROOT cipher_oracle.cpp defines. The two ROOT files are paired by design.

---

## Step 3 — Git history walk

```
$ git rev-parse HEAD
fc8a9ae6  Week 1 Step 1: LP-7 struct rename

$ git log --oneline --all -- cipher_dispatch.cpp
83b76da  Baseline snapshot pre-Week-1 integration.

$ git log --oneline --all -- src/cipher_dispatch.cpp
83b76da  Baseline snapshot pre-Week-1 integration.

(same for cipher_oracle.cpp pair)
```

**Both variants were ingested in a single commit (`83b76da`, 2026-05-20 10:19) as a baseline snapshot.** Total commit count = 1 for each variant. Git history offers no fork-point info — the cipher-fusion campaign's git repo was initialized fresh from a pre-existing evidence directory; whatever variant state the source tree had at the moment of ingest, both variants came in together.

The mtime of all 4 files is identical (`May 13 04:27:19`), suggesting they all moved through the same backup/snapshot process at the same time on the prior system. They are not active branches in this repo; they're a frozen import.

**Implication for adjudication:** git history cannot tell us which variant is canonical. The signal must come from build-system consumption (Step 4) and prebuilt-artifact attribution (Step 6).

---

## Step 4 — Makefile consumption

The evidence Makefile, lines 29-31:

```make
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp src/cipher_persist.cpp \
                            src/cipher_nccl_tuner.cpp src/cipher_dispatch.cpp \
                            src/cipher_oracle.cpp, $(wildcard src/*.cpp))
RT_CPP_SRC += cipher_dispatch.cpp cipher_oracle.cpp
```

Lines 43-44 derive object paths (note the path prefix discriminator):

```make
RT_CPP_OBJ := $(patsubst src/%.cpp,$(BUILDDIR)/src__%.o,$(filter src/%,$(RT_CPP_SRC))) \
              $(patsubst %.cpp,$(BUILDDIR)/%.o,$(filter-out src/%,$(RT_CPP_SRC)))
```

The pattern produces two object-name conventions:
- `src/foo.cpp → build/src__foo.o`   (src/-prefixed convention)
- `foo.cpp     → build/foo.o`         (no-prefix convention — the ROOT variants)

This explains why `build/cipher_dispatch.o` (no `src__` prefix) exists and `build/src__cipher_dispatch.o` does NOT — the prebuilt path name itself is a signal that ROOT was the source.

### Other build targets verified

```make
HOOK_SRC := src/cipher_intercept_cudart.cpp src/cipher_persist.cpp \
            src/cipher_graph_inspect.cpp src/cipher_kernel_table.cpp \
            src/cipher_flow_recorder.cpp src/cipher_flow_patterns.cpp \
            src/cipher_flow_substitute.cpp
```

`libcipher_hook.so` (the F1 LD_PRELOAD hook DSO) uses only `src/*` files — none of them have ROOT-level variants. **The two-variant-pair pattern is unique to `cipher_dispatch.cpp` and `cipher_oracle.cpp`, and only for `libcipher_rt.so`.**

```make
tuner: libcipher_nccl_tuner.so libnccl-tuner-cipher.so
```

The NCCL tuner DSO uses `src/cipher_nccl_tuner.cpp` only — single variant, no pair.

### Documentation / test references

```
$ grep -rln 'cipher_dispatch.cpp\|src/cipher_dispatch' /home/ubuntu/cipher-may13-evidence/*.md
(no matches)
```

No README / docs file mentions either variant by path. Tests reference oracle/dispatch by **API symbol** (`cipher_oracle_init`, `cipher_oracle_decide`, `cipher_dispatch`) — both variants expose those symbols, so tests don't disambiguate. Test signal is null.

---

## Step 5 — Functional intent inspection

### File-header banners

Both `cipher_dispatch.cpp` variants begin with the *identical* header:

```
// CPU stub redirect
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
// ============================================================
// CIPHER — Layer 3 Dispatch Engine
// cipher_dispatch.cpp
//
// Replaces the Phase 0 passthrough stub in cipher_runtime.cpp.
// This IS the hot path. Every GPU kernel launch hits this function.
//
// DECISION FLOW (per kernel launch):
//   cuLaunchKernel intercept (F1) → classify_launch → struct_lookup
//   → oracle_decide → registry_lookup → apply recipe → record_substitution
// ============================================================
```

(Identical first 40 lines in both. Both variants describe themselves as "Layer 3 Dispatch Engine" — same canonical role.)

Same for `cipher_oracle.cpp` — identical first 30 lines: "CIPHER — L3.6 + L3.7 + L3.9: Accuracy Oracle Implementation".

**No `deprecated` / `experimental` / `legacy` / `v2` markers in either variant.** Both look equally production-grade. The headers do not disambiguate intent.

### Functional pattern

The behavioral diff cluster (from §2):

- ROOT cipher_dispatch.cpp's classifier path: calls **`cipher_edmd_live_collect`** (live-EDMD batched collection from cipher_edmd_live.cpp)
- SRC cipher_dispatch.cpp's classifier path: calls **`cipher_block_sub_collect`** (block-substitute pattern with relaunch cache in cipher_edmd.cpp)

These are two different EDMD harvesting strategies — both legitimate research-grade implementations of the L3.5 "novel-op derivation" component. The ROOT variant uses the lighter-weight live-collect; the SRC variant uses the heavier substitute-and-cache pattern with cudaMalloc-managed side buffers (visible in the SRC-only undef set).

This pattern is consistent with **the SRC variant being a research/extension branch that did not merge to the production build path**. The Makefile actively excludes it; the test/runtime path doesn't reach it; and the prebuilt build/.o reflects the ROOT path.

---

## Step 6 — Prebuilt .o attribution (the decisive test)

Compile both variants with the may13 Makefile's exact flags (`CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter` with `INCLUDES := -I./include -I/usr/include`) and compare to the prebuilt `.o`:

```
$ md5sum /tmp/v3_root_dispatch.o /tmp/v3_src_dispatch.o build/cipher_dispatch.o
022c5176943dad86fbcf4dc8daeb10e5  /tmp/v3_root_dispatch.o      ← ROOT fresh
d09ebcdf3d4832f8cb00d304b04aa294  /tmp/v3_src_dispatch.o       ← SRC fresh
022c5176943dad86fbcf4dc8daeb10e5  build/cipher_dispatch.o      ← PREBUILT
```

**The prebuilt is byte-identical to a fresh-compile of ROOT.** Not just symbol-set-equivalent — the same binary down to every byte. This is the strongest possible attribution signal.

### Symbol-set intersection (corroborates byte-identical)

| metric | value |
| --- | ---:|
| ROOT fresh ∩ Prebuilt | 56 / 56 (100% match) |
| SRC fresh ∩ Prebuilt | 47 / 64 (74%) |
| Symbols in Prebuilt but not in SRC | `cipher_edmd_live_collect`, `cipher_oracle_bill_gemm`, `cipher_oracle_bill_nongemm`, `cipher_oracle_billing_report`, `cipher_tls_get_gemm_types` |
| Symbols in Prebuilt but not in ROOT | (none) |

The exact symbol set of the prebuilt matches the exact symbol set of ROOT. SRC is missing 9 symbols and has 17 extra symbols vs the prebuilt.

(Same byte-identical result confirmed independently for cipher_oracle.cpp: fresh-compile of ROOT cipher_oracle.cpp = build/cipher_oracle.o.)

---

## Step 7 — Verdict

### Evidence summary

| signal | direction |
| --- | --- |
| Makefile lines 29-31 explicit filter | **ROOT** (src/ excluded by name) |
| Object-name convention `build/foo.o` vs `build/src__foo.o` | **ROOT** (no `src__` prefix → root variant) |
| Prebuilt `build/cipher_dispatch.o` byte-identical md5 | **ROOT** (`022c5176` exact) |
| Symbol-set match | **ROOT** (56/56, 100%) |
| Functional pattern (lighter live-collect vs heavier substitute-cache) | **ROOT** path is the simpler, production-build-consumed one |
| File-header banners | inconclusive (identical) |
| Git history | inconclusive (single baseline ingest) |
| Tests / docs references | inconclusive (API-only) |

All disambiguating signals point in the same direction. No signal points to SRC.

### Scenario classification

**ROOT-CANONICAL.** The Makefile uses ROOT, the prebuilt build artifact is bit-for-bit derived from ROOT, and the SRC variants are file-system orphans that the build system explicitly filters out. They appear to be a research/experimental branch that never merged into the canonical build path.

### Recommendation: D1

Per the v3 PARTIAL doc's adjudication framing:

> **D1 — Port from ROOT variants** (`cipher-may13-evidence/cipher_dispatch.cpp` 543L + `cipher-may13-evidence/cipher_oracle.cpp` 539L); re-run the prior 4-round audit chain against the root variants to re-confirm the 11-file A4 closure; if confirmed, retry sub-step C with the right sources.

D1 is the right path because:

1. **The prior 4 audits already used the ROOT-variant prebuilt .o files** (verified byte-identical above). Re-validation against ROOT is therefore mostly a *confirmation* exercise — the symbol-closure, A4 closure rerun, and orphan inspection are already authoritative for the ROOT variant.
2. **The static-init pre-flight finding (FLAG on `cipher_intercept.cpp` + `cipher_intercept_cudart.cpp`) is unaffected** — those files have no ROOT/SRC duplicates.
3. The single change required is to re-port `cipher_dispatch.cpp` and `cipher_oracle.cpp` from ROOT (not src/). All other 9 files in the A4 11-file closure are unique sources (no variant pair).
4. The 11-file A4 closure's symbol surface should reproduce in the fresh ROOT compile (because the prebuilt audit was based on ROOT).

### D2 / D3 / D4 ruled out

- **D2** (use SRC) — contradicts every disambiguating signal. The SRC variants are not part of the canonical evidence build; choosing them would diverge from what the evidence tree ships.
- **D3** (history dive) — completed. Result is D1.
- **D4** (other / SPLIT-BRAIN) — no SPLIT-BRAIN evidence. Only ROOT has a build target; only ROOT has prebuilt artifacts; only ROOT is referenced by name in the Makefile inclusion list.

---

## Telemetry on disk

- `/tmp/v3_root_dispatch.o` — fresh compile of ROOT cipher_dispatch.cpp (md5 022c5176, **= prebuilt**)
- `/tmp/v3_src_dispatch.o` — fresh compile of SRC cipher_dispatch.cpp (md5 d09ebcdf, ≠ prebuilt)
- `/tmp/v3_sym_root.txt`, `/tmp/v3_sym_src.txt`, `/tmp/v3_sym_pre.txt` — symbol set dumps
- `/tmp/week2_step2_v3_history/root_files.txt` — root-level file inventory
- `/tmp/week2_step2_v3_history/cipher_dispatch_root.o`, `_src.o` — pair-1 .o
- `/tmp/week2_step2_v3_history/cipher_oracle_root.o`, `_src.o` — pair-2 .o

---

## Discipline notes

- Read-only diagnostic. No source-tree changes.
- The verdict is unambiguous (all 5 disambiguating signals align on ROOT), so this is a recommendation, not a coin-flip.
- The v3 PARTIAL doc's rollback path (§11) is still required before re-attempting Step 2 v3 with ROOT variants — the working tree currently contains the wrong (SRC) port.

---

## Awaiting

User confirmation of **D1**. After confirmation:

1. Rollback the dirty v3 working tree (rollback script in `WEEK_2_STEP_2_V3_RESULT.md` §11).
2. Quick re-validation of the A4 closure against ROOT-variant compiles (~30 min: rerun `/tmp/symbol_closure_a4.py` after replacing `src/cipher_dispatch.cpp` + `src/cipher_oracle.cpp` references with their ROOT counterparts in `find_obj()` — the resulting closure should remain ~11 files).
3. Re-attempt Step 2 v3 sub-step C onward, copying ROOT cipher_dispatch.cpp + ROOT cipher_oracle.cpp instead of the src/ variants. The remaining 9 source files (recipes, sense, structural_lookup, kernel_table, runtime, telemetry, green_ctx.cu, l2_persist.cu, liquid_state.cu) have no variant pair and remain unchanged.

D1 unblocks the Step 2 v3 retry.
