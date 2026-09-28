# Week 1 Step 3 — Result

**HEADLINE STATUS: PASS.**

**Date:** 2026-05-20
**Scope:** Cross-tree compile harness. New TU `cipher_rt_phase4/src/cipher_may13_harness.cpp` includes `may13/cipher_classify.hpp` and declares 3 static markers (one string + one enum value + one POD struct). Plus 3 Makefile changes (`-Iinclude` to INCLUDES, new .o in OBJS, new per-source rule). All other Step-3 outputs are derived measurements (build rc, warning delta, nm diff, regression).
**Outcome:** All six steps PASS. The harness compiles into `libcipher_rt.so`; nm diff shows 3 new internal-linkage symbols added, zero existing symbols removed or name-changed; the marker string is grep-able in the binary; CP 5.4 isolation 15/15 PASS unchanged from Step 2 v2 baseline. Commit `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` landed and tagged `week-1-step-3-cross-tree-harness`.

---

## Section A — Pre-harness verification

### A.1 Git state at entry

| Field | Value |
|---|---|
| HEAD | `50a6f2283032420d7bd664f29ba1d81a786f25c0` |
| Tag pointing here | `week-1-step-2-v2-header-port` |
| `pre-week-1-baseline` tag (preserved rollback anchor) | `b621453c4602f78d4751502a6873032ca1999e7a` |
| Working tree | clean |

### A.2 Pre-harness libcipher_rt.so md5 + closure state

- `libcipher_rt.so` md5: `83afd1ca4118dc651854ef751e4ef82d` — matches anchor.
- `include/may13/` file count: **12** (the Option B closure from Step 2 v2).
- Pre-harness DSO copy preserved at `/tmp/week1_step3/libcipher_rt.so.preharness`.
- Pre-harness nm output preserved at `/tmp/week1_step3/nm_preharness_D.txt` (186 dynamic symbols) and `/tmp/week1_step3/nm_preharness_all.txt` (353 total symbols).

### A.3 Pre-harness clean build

`make clean && make`:

- rc = 0
- Warning count: **18**
- Post-build libcipher_rt.so md5: `83afd1ca4118dc651854ef751e4ef82d` (anchor REPRODUCED byte-identical).
- Full log: `/tmp/week1_step3/pre_harness_build.log`.

**Section A verdict: PASS.** Baseline clean and reproducible; harness can land safely.

---

## Section B — Author harness TU

### B.1 may13 type chosen

Inspected `include/may13/cipher_classify.hpp` for candidate types:

- `enum class OpClass : uint8_t` (L45-54) — 7-class kernel-class enum with `UNCLASSIFIED=0xFF` sentinel. **Chosen.**
- `struct ClassifyResult` (L57-61) — 3-field POD: `{ OpClass op; uint8_t confidence; bool cache_hit; }`. **Chosen.**
- `struct KernelGeom` (L64-72) — declared but not used in the harness (has a function-pointer field; trivial to declare but adds nothing beyond the two chosen).

Both `OpClass` and `ClassifyResult` are pure type definitions; neither has a constructor with side effects; both can be declared as static variables without any may13 code running at static-init time. Both are reached via `#include "may13/cipher_classify.hpp"` and provide a compile-time proof that the may13/ prefix resolves and the type is visible to the cipher_rt_phase4 source-file scope.

### B.2 Harness TU contents

File: `/home/ubuntu/cipher_rt_phase4/src/cipher_may13_harness.cpp` (59 lines).

Top-level structure:

```cpp
#include "may13/cipher_classify.hpp"

namespace cipher_may13_harness {

static const char* const k_harness_marker
    __attribute__((used)) = "cipher_may13_harness_v1";

static const cipher::OpClass k_harness_class
    __attribute__((used)) = cipher::OpClass::GEMM;

static const cipher::ClassifyResult k_harness_result
    __attribute__((used)) = {
    cipher::OpClass::UNCLASSIFIED,
    0u,
    false,
};

static_assert(static_cast<int>(cipher::OpClass::GEMM) == 0, ...);
static_assert(static_cast<int>(cipher::OpClass::UNCLASSIFIED) == 0xFF, ...);
static_assert(sizeof(cipher::ClassifyResult) >= 3, ...);

}  // namespace cipher_may13_harness
```

### B.3 Iteration: `__attribute__((used))` was required

The first iteration used plain `static const ...` declarations without `__attribute__((used))`. At `-O2`, the compiler dead-stripped all three statics (they were never referenced from any non-inline function), and the resulting `.o` was 952 bytes with **zero** symbols. The marker string was not present in the linked library (`strings | grep` returned no match) and `nm cipher_may13_harness.o` was empty.

**Surfaced as Step D mid-iteration.** The brief requires "Create at least one symbol that nm can detect (the marker string suffices)." `__attribute__((used))` is the standard way to force the compiler to emit a static even when the static is unreferenced. Applied to all three declarations. Re-built and re-verified: 3 symbols emitted, marker string present.

### B.4 What the harness proves at compile time

1. `#include "may13/cipher_classify.hpp"` resolves under `cipher_rt_phase4`'s include path (Step C added `-Iinclude` to INCLUDES).
2. `cipher::OpClass` and `cipher::ClassifyResult` are reachable type names in the cipher_rt_phase4 source-file scope.
3. The 3 `static_assert` lines confirm the may13 enum layout (`GEMM == 0`, `UNCLASSIFIED == 0xFF`) and `ClassifyResult` size (`>= 3` bytes for the 3 fields).
4. No may13 function is called; no constructor with side effects runs at static-init.

**Section B verdict: PASS.** Harness TU compiles; chosen types documented; the `__attribute__((used))` iteration ensured the marker is detectable per brief requirement.

---

## Section C — Makefile integration

### C.1 Three changes

```diff
-INCLUDES := -I$(CIPHER_KMOD_DIR) -I$(CUDA_INCLUDE) -I$(CUPTI_INCLUDE)
+INCLUDES := -I$(CIPHER_KMOD_DIR) -I$(CUDA_INCLUDE) -I$(CUPTI_INCLUDE) -Iinclude

           cipher_rt_attn_dispatch.o cipher_rt_attn_test_actuator.o \
-          cipher_rt_audit.o
+          cipher_rt_audit.o \
+          cipher_may13_harness.o

+# Week 1 Step 3 — cross-tree compile harness. Includes one may13 header
+# (cipher_classify.hpp via may13/ prefix) and declares one static marker
+# string plus two static may13-typed values. No runtime effect; verified
+# by nm-diff that only the harness marker is added to libcipher_rt.so.
+cipher_may13_harness.o: src/cipher_may13_harness.cpp include/may13/cipher_classify.hpp
+	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<
```

Total Makefile delta: +12 / -2 lines. No other Makefile changes; `LDFLAGS`, `LIBS`, `C10_LIBS`, `CFLAGS`, `CXXFLAGS` untouched.

### C.2 What `-Iinclude` enables

Before this Step, the synthetic compile smoke at Step 2 v2 passed only by adding `-Iinclude` as a test-local flag. Production library TUs did not see the may13 directory. After this Step's `INCLUDES` patch, every library TU has `-Iinclude` in scope. **None of the existing TUs include any may13 header**, so this change has zero observable effect on the existing 17 TUs. The change is enabling — Week 2 wiring can `#include "may13/..."` directly without further Makefile changes.

**Section C verdict: PASS.** Three minimal, well-scoped Makefile changes.

---

## Section D — Build + nm diff

### D.1 Post-harness clean build

`make clean && make`:

- rc = 0
- Warning count: **18** (delta zero vs pre-harness)
- Full log: `/tmp/week1_step3/post_harness_build_v2.log`
- The new compile line is visible:

  ```
  g++ -O2 -Wall -Wextra -fPIC -I/home/ubuntu/cipher_kmod -I/usr/include -I/usr/include -Iinclude -c -o cipher_may13_harness.o src/cipher_may13_harness.cpp
  ```

- The new object joins the link line at the position dictated by OBJS order:

  ```
  g++ -shared -fPIC -o libcipher_rt.so ... cipher_rt_audit.o cipher_may13_harness.o -lpthread ...
  ```

### D.2 libcipher_rt.so md5 — changed (expected)

| State | md5 |
|---|---|
| Pre-harness (anchor) | `83afd1ca4118dc651854ef751e4ef82d` |
| Post-harness | `88ed35bb13b0524e889bb5b56d610fd9` |

The md5 change is expected — a new TU added new code to the .so. The Step 2 invariant ("md5 must not change") no longer applies because Step 3 deliberately adds code. The Step 3 invariant is the nm diff (D.3): the change in the binary must be *only* the new harness symbols, not perturbation of any existing symbol.

### D.3 nm diff — 3 symbols added, zero removed or name-changed

| Metric | Pre-harness | Post-harness | Delta |
|---|---|---|---|
| `nm -D` (dynamic) symbols | 186 | 186 | **0** (all 3 harness symbols are internal-linkage) |
| `nm` full (incl. static) | 353 | 356 | **+3** (the 3 harness symbols) |

Address-stripped name-only diff (the canonical correctness check — by symbol name, not by address):

```
247a248
> d _ZN20cipher_may13_harnessL16k_harness_markerE
281a283,284
> r _ZN20cipher_may13_harnessL15k_harness_classE
> r _ZN20cipher_may13_harnessL16k_harness_resultE
```

3 new symbol names added, **0 symbol names removed, 0 symbol names changed**. The L-prefix in `_ZN20cipher_may13_harnessL16k_harness_markerE` confirms internal linkage (Itanium ABI mangling — namespace-local statics).

### D.4 Address layout shift (expected, harmless)

Three linker-bookkeeping symbols moved address but kept name:

| Symbol | Pre address | Post address |
|---|---|---|
| `_DYNAMIC` | `0x1fd28` | `0x1fd30` |
| `__FRAME_END__` | `0x1e194` | `0x1e1ac` |
| `__GNU_EH_FRAME_HDR` | `0x1c3a0` | `0x1c3bc` |

These are layout artifacts: adding 952 bytes of new TU shifts the dynamic-section pointer, the eh_frame end marker, and the eh_frame header position. The address-stripped diff at D.3 confirms the names are unchanged — pure layout drift, no semantic change.

### D.5 Marker string visible

```
$ strings libcipher_rt.so | grep cipher_may13_harness_v1
cipher_may13_harness_v1
```

Marker present. The `__attribute__((used))` annotation prevented dead-strip; the string lives in `.rodata`.

### D.6 nm shows 3 harness symbols

```
$ nm libcipher_rt.so | grep cipher_may13_harness
000000000001c3a3 r _ZN20cipher_may13_harnessL15k_harness_classE
000000000001fd28 d _ZN20cipher_may13_harnessL16k_harness_markerE
000000000001c3a0 r _ZN20cipher_may13_harnessL16k_harness_resultE
```

- `k_harness_marker` in `.data` (mutable pointer to constant string).
- `k_harness_class` and `k_harness_result` in `.rodata` (read-only constants).

All three have lowercase letter codes (`d`, `r`) confirming local-static (internal) linkage — they are not exported in the dynamic symbol table (verified by D.3 dynamic-count delta = 0).

**Section D verdict: PASS.** Build rc=0; warning delta zero; nm diff is clean (3 new internal symbols added, zero existing symbol changes); marker string visible.

---

## Section E — Regression smoke

### E.1 CP 5.4 isolation 15/15

Re-ran `/home/ubuntu/cipher-fusion-evidence/cp_5_4/step1_3/cp54_isolation_test`:

| Test | Sub-assertions | Result |
|---|---|---|
| Test 1 — legacy nr-9 deactivated | 1 | PASS |
| Test 2 — ALLOCATE / FREE / QUERY | 4 | 4× PASS |
| Test 3 — pool resize | 4 | 4× PASS |
| Test 4 — do_exit reaper | 1 | PASS |
| Test 5 — disjointness | 3 | 3× PASS |
| Test 6 — concurrent stress | 2 | 2× PASS |
| **Total** | **15** | **15 PASS, 0 FAIL** |

Output line-for-line identical to the Step 2 v2 baseline. `/tmp/week1_step3/cp54_isolation.log` preserved.

### E.2 Implication

The harness TU compiles into libcipher_rt.so but the substrate behavior is unchanged. No may13 init function ran. No new ioctls were issued. The cipher_kmod side received no new traffic. The 15-group ledger, the do_exit reaper, the disjointness invariant, and the concurrent-stress test all behave byte-identically. The harness is functionally inert at runtime, exactly as the brief required.

**Section E verdict: PASS.** No regression introduced.

---

## Section F — Commit and tag

### F.1 Diff stat

```
 Makefile                     | 12 +++++++--
 src/cipher_may13_harness.cpp | 59 ++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 69 insertions(+), 2 deletions(-)
```

2 files changed (1 new TU + 1 Makefile patch), 69 insertions, 2 deletions. Matches the brief's expectation ("1 new file, 1 Makefile change").

### F.2 Commit + tag

| Field | Value |
|---|---|
| Commit SHA | `bcf8a83b10e5112a10e03afdb70cabb9b74d75c4` |
| Tag | `week-1-step-3-cross-tree-harness` (lightweight) at the same SHA |
| Previous tag preserved | `week-1-step-2-v2-header-port` at `50a6f228...` |
| Base baseline tag preserved | `pre-week-1-baseline` at `b621453c...` |
| Working tree post-commit | clean (zero pending) |
| Author | `Anil <anil.0666369@gmail.com>` |
| Co-Authored-By | `Claude Opus 4.7 (1M context) <noreply@anthropic.com>` |

### F.3 Rollback chain

- **Roll back this step:** `git reset --hard week-1-step-2-v2-header-port` returns to the post-Step-2-v2 state.
- **Roll back all of Week 1 in cipher_rt_phase4:** `git reset --hard pre-week-1-baseline` returns the tree to the state before Step 2 v2.
- **Tarball:** `/home/ubuntu/cipher-baselines/pre_week_1_cipher_rt_phase4_20260520.tar.gz` (md5 `57dc351077bc247c289ef44f82a9d17e`) holds the full pre-week-1 state including `.git`.

**Section F verdict: PASS.** Commit landed, tag placed, working tree clean.

---

## Headline summary

| Field | Value |
|---|---|
| Headline status | **PASS** |
| Step A — pre-verification | PASS |
| Step B — author harness TU | PASS (with one mid-step iteration to add `__attribute__((used))`) |
| Step C — Makefile integration | PASS (+12/-2 lines, 3 changes) |
| Step D — build + nm diff | PASS (rc=0; warning delta 0; 3 new internal symbols; 0 existing changed; marker visible) |
| Step E — regression smoke | PASS (CP 5.4 isolation 15/15, line-for-line identical to baseline) |
| Step F — commit + tag | PASS (SHA `bcf8a83b...`, tag `week-1-step-3-cross-tree-harness`) |
| Files changed | 2 (`src/cipher_may13_harness.cpp` new; `Makefile` patched) |
| Lines added | 69 |
| libcipher_rt.so md5 transition | `83afd1ca...` (anchor) → `88ed35bb...` (post-harness) |
| Rollback path | `git reset --hard week-1-step-2-v2-header-port` |

### Surfaced (informational, no mitigation proposed)

1. **`__attribute__((used))` requirement.** Without it, `-O2` dead-stripped all three static markers and the harness produced zero symbols. The brief's example showed `static const char* k_harness_marker = ...;` without `__attribute__((used))`; in practice this requires the annotation to satisfy the brief's "marker symbol grep-able" criterion.

### Week 1 closeout readiness

Per plan v1.2.2 §7 Week 1, Week 1 = "Compile-level classifier port + struct collision fix + snapshot reserved-tail bump." Step 1 fixed the struct collision (LP-7). Step 2 v2 ported the 12-header closure. Step 3 proved cross-tree linkage. The remaining Week 1 item is the **snapshot reserved-tail bump** (Cb.2 — adds 4 of 16 `__u32 reserved[16]` slots in `cipher_pid_stats` / `cipher_tenant_snapshot_user`) which lives in `cipher_kmod/cipher_internal.h` and `cipher_kmod/cipher_ioctl.h`, NOT in `cipher-may13-evidence/include/`. That requires a kmod-side change (recompile + re-load).

Once the snapshot reserved-tail bump lands, Week 1 is closeable.

**Week 1 closeout prompt can be sent.**

---

**End of WEEK_1_STEP_3_RESULT.md.**
