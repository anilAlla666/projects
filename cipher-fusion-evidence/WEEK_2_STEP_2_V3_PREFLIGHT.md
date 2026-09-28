# Week 2 Step 2 v3 — Static-Init Pre-Flight on 40-File Option A Closure

**HEADLINE STATUS: ADJUDICATION REQUIRED.**

**FLAG count: 2 files.** Both install CUDA driver hooks at libcipher_rt.so load time, in direct conflict with cipher_rt_phase4's existing GOT patcher (cipher_rt_got_patch.c). Option A as-briefed cannot proceed without resolving the conflict — and the conflict resolution is structural, not a wrapper.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 2, Step 2 v3 pre-flight (Option A audit before port)
**Tree state:** unchanged (read-only audit)

---

## 1. Method

For each of the 40 files in the Option A closure (from `WEEK_2_STEP_2_SYMBOL_CLOSURE_AUDIT.md` §3), read the corresponding pre-built `.o` in `cipher-may13-evidence/build/` with `readelf` + `objdump` + `nm`:

1. **Section-level signal** — does the `.o` have `.init_array` / `.init_array.NNNNN` sections? non-zero size implies a static initializer.
2. **Symbol-level signal** — does `nm` show `_GLOBAL__sub_I_*`, `__sti____cudaRegisterAll`, `__static_initialization_and_destruction`, or a custom-named init in `.text.startup`?
3. **Relocation trace** — for each found init target, read `.rela.text.startup` to see which external symbols its body calls.
4. **Body disassembly** — `objdump -d --disassemble=<symbol>` for each init target; grep for cuda*/cu*/nvml*/pthread_create/open/ioctl/dlopen/dlsym calls.
5. **Categorize** as CLEAN / WATCH / FLAG per the rubric.

Rubric:
- **CLEAN** — no static init, OR static init is stock nvcc fatbin registration (`__cudaRegisterFatBinary`), OR pure C++ object construction without side effects.
- **WATCH** — static init touches CUDA/NVML, but the call is conditional (e.g. env-var-gated, or returns harmless status without engaging driver state).
- **FLAG** — static init unconditionally invokes a side-effecting call (kmod ioctl, CUDA driver state change, hook installation, thread spawn).

Tooling produced: `/tmp/static_init_scan.py` (40-file scan), `/tmp/static_init_trace.py` (per-file disassembly trace). Reusable for future weeks.

---

## 2. Per-file summary table

`init_array` is `.init_array` + `.init_array.NNNNN` total bytes. `#sub_I` is count of `_GLOBAL__sub_I_*` or `__sti_*` symbols.

| file | init_array (bytes) | #sub_I | category | notes |
| --- | ---:| ---:| --- | --- |
| cipher_dispatch.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_oracle.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_recipes.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_sense.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_structural_lookup.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_kernel_table.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_10ops_impl.cpp | 0 | 0 | **CLEAN** | no static init |
| cipher_block_sub_kernel.cu | 8 | 0 | **CLEAN** | `__sti____cudaRegisterAll()` — stock nvcc fatbin registration |
| cipher_carbon.cpp | 0 | 0 | **CLEAN** | |
| cipher_comply.cpp | 0 | 0 | **CLEAN** | |
| cipher_continuity.cpp | 0 | 0 | **CLEAN** | |
| cipher_determinism.cpp | 0 | 0 | **CLEAN** | |
| cipher_edmd.cpp | 0 | 0 | **CLEAN** | |
| cipher_fairness.cpp | 0 | 0 | **CLEAN** | |
| cipher_flow_patterns.cpp | 0 | 0 | **CLEAN** | |
| cipher_flow_recorder.cpp | 0 | 0 | **CLEAN** | |
| cipher_flow_substitute.cpp | 0 | 0 | **CLEAN** | |
| cipher_green_ctx.cu | 8 | 0 | **CLEAN** | `__sti____cudaRegisterAll()` — fatbin registration |
| cipher_guard.cpp | 0 | 0 | **CLEAN** | |
| cipher_hibernate.cpp | 0 | 0 | **CLEAN** | |
| **cipher_intercept.cpp** | **8** | **0** | **FLAG** | `.text.startup:cipher_so_init()` calls `cipher_10ops_init()` then `cipher_intercept_init()`; the latter installs F1 hooks into libcuda.so cuGetProcAddress. See §4. |
| **cipher_intercept_cudart.cpp** | **16** | **1** | **FLAG** | two `.init_array` entries: priority-101 calls `cipher_hook_init()` (dl_iterate_phdr + GOT patch + sysconf for page size); default-priority `_GLOBAL__sub_I_cipher_intercept_cudart.cpp` constructs C++ std::unordered_map for in-flight kernel tracking. The GOT-patch path conflicts. See §4. |
| cipher_l2_persist.cu | 8 | 0 | **CLEAN** | `__sti____cudaRegisterAll()` — fatbin registration |
| cipher_liquid_state.cu | 8 | 0 | **CLEAN** | `__sti____cudaRegisterAll()` — fatbin registration |
| cipher_lnn.cpp | 0 | 0 | **CLEAN** | |
| cipher_loop.cpp | 0 | 0 | **CLEAN** | |
| cipher_persist.cpp | 0 | 0 | **CLEAN** | |
| cipher_pipeline.cpp | 0 | 0 | **CLEAN** | |
| cipher_predict.cpp | 0 | 0 | **CLEAN** | |
| cipher_pulse.cpp | 0 | 0 | **CLEAN** | |
| cipher_receipt.cpp | 0 | 0 | **CLEAN** | |
| cipher_runtime.cpp | 0 | 0 | **CLEAN** | g_cipher = {0} is a zero-initialized .bss object, not a runtime constructor |
| cipher_shield.cpp | 0 | 0 | **CLEAN** | |
| cipher_straggler.cpp | 0 | 0 | **CLEAN** | |
| cipher_sustain.cpp | 0 | 0 | **CLEAN** | |
| cipher_telemetry.cpp | 0 | 0 | **CLEAN** | |
| cipher_thermostat.cpp | 0 | 0 | **CLEAN** | |
| cipher_topology.cpp | 0 | 0 | **CLEAN** | |
| cipher_trace.cpp | 0 | 0 | **CLEAN** | |
| cipher_volt.cpp | 0 | 0 | **CLEAN** | |

---

## 3. Aggregate counts

| category | count | files |
| --- | ---:| --- |
| CLEAN | 38 | the 34 zero-init files + the 4 `.cu` files (stock nvcc fatbin registration) |
| WATCH | 0 | — |
| **FLAG** | **2** | **cipher_intercept.cpp**, **cipher_intercept_cudart.cpp** |

The 4 `.cu` files' `__sti____cudaRegisterAll()` calls `__cudaRegisterFatBinary` (and downstream `__cudaRegisterFunction` / `__cudaRegisterFatBinaryEnd`). These are stock CUDA runtime API calls injected by nvcc — they register the compiled kernel binary with the CUDA runtime's bookkeeping. They do NOT init the device, allocate memory, or touch the driver state. They are safe at libcipher_rt.so load time regardless of whether CUDA has been initialized. CLEAN.

---

## 4. FLAG file detail

### 4.1 cipher_intercept.cpp — F1 hook installer

`.text.startup` contains one function: `cipher_so_init()`. Its body (relocations decoded):

```
cipher_so_init():
    call cipher_10ops_init      # init the 10-ops registry (.text PLT32)
    call cipher_intercept_init  # F1 hook installer — see below
    test eax, eax               # return value of cipher_intercept_init
    je <ret>                    # success: return 0
    # failure path:
    __fprintf_chk(stderr, 1, "<format>", <err code>)  # log error
    ret
```

`cipher_intercept_init()` is the load-bearing F1 hook. Its FUNC table shows:

```
0x000000  cuGetProcAddress_v2   367 bytes  GLOBAL   ← our replacement implementation
0x000750  cuLaunchKernelEx      198 bytes  GLOBAL   ← our replacement implementation
0x000820  cipher_intercept_init 367 bytes  GLOBAL   ← installer
0x000990  cipher_f1_ring_write  430 bytes  GLOBAL
0x000b40  cipher_intercept_teardown ...
0x000bb0  cipher_passthrough    139 bytes
0x000c40  cipher_intercept_stats
```

The file *defines* `cuGetProcAddress_v2` and `cuLaunchKernelEx` (per `cipher_intercept.h`: "Hooks cuLaunchKernel via cuGetProcAddress before any GPU compute touches silicon. LD_PRELOAD on libcuda.so. Sub-100ns passthrough."). At `.init_array` time, `cipher_intercept_init()` dynamically installs these replacements via dlopen("libcuda.so") + cuGetProcAddress redirection.

**Why this is FLAG:**

cipher_rt_phase4 already has a GOT/PLT patcher (`cipher_rt_got_patch.c`, retired LD_PRELOAD link-order; replaces it with run-time GOT rewriting). The patcher installs hooks for cuBLAS / SDPA at first-call lazy time via `dl_iterate_phdr`. It is the canonical interception mechanism for the production substrate.

If `cipher_intercept.cpp` is linked into libcipher_rt.so with its `.init_array` intact, then at `dlopen(libcipher_rt.so)` time:
1. cipher_so_init() runs *during the dynamic linker's processing of .init_array* — i.e. before any user code in the host process has executed.
2. It invokes cipher_intercept_init(), which performs dlopen of libcuda.so and installs hooks via cuGetProcAddress redirection — **a parallel interception path** that races with the GOT patcher's deferred lazy installation.

Two hook-installer subsystems acting on the same driver entrypoints. The race is not detectable at load time (both succeed); the symptom would be incorrect dispatch on the second invocation of cuLaunchKernel — could be either passthrough-then-passthrough (correct by accident) or hook-then-hook (double-shim, wrong).

This is the explicit class of failure cipher_rt_phase4's GOT-patcher architecture was designed to *exclude* (see CP 2.5 / GOT-patcher retiring T4.5's cublas_version.map). Importing cipher_intercept.cpp's `.init_array` reintroduces the older mechanism.

### 4.2 cipher_intercept_cudart.cpp — cudart GOT patcher

Two `.init_array` entries:

```
.init_array.00101  (priority 101 — runs early)  → cipher_hook_init()
.init_array        (default priority 65535)     → _GLOBAL__sub_I_cipher_intercept_cudart.cpp
```

`cipher_hook_init()` body (disassembly trace, relocations resolved):

```
cipher_hook_init():
    mov  $0x1e, %edi          # _SC_PAGESIZE = 30
    call sysconf              # get page size — page-rounded mprotect for GOT rewrite
    fprintf(stderr, ...)      # log hook init banner
    check g_initialized flag
    call dl_iterate_phdr      # walk loaded modules
    # On each loaded shared object:
    #   patch_slot(got_slot_for_cudaLaunchKernel, &cudaLaunchKernel_dispatch)
    #   patch_slot(got_slot_for_cudaMemcpy, &log_memcpy_if_probing)
    #   patch_slot(got_slot_for_cudaMalloc, ...)
    ...
```

It uses `dl_iterate_phdr` + `mprotect(PROT_WRITE)` + GOT slot rewrite. **This is the same mechanism `cipher_rt_phase4/cipher_rt_got_patch.c` uses.** Direct conflict on:

- `cudaLaunchKernel` — already patched by cipher_rt_phase4's GOT patcher (recipe is at `[cipher_v2] GOT: patch applied -- 8 slot(s) across 70 module(s); 4 target(s) registered` per Step 1 smoke output)
- `cudaMalloc` / `cudaMemcpy` — patched at runtime by cipher_intercept_cudart.cpp for logging; would shadow whatever's already there

The second entry `_GLOBAL__sub_I_cipher_intercept_cudart.cpp` constructs C++ `std::unordered_map<void*, ...>` objects in `.data` for in-flight kernel tracking (the `g_kernel_track` map referenced by `_ZL12track_kernelPv`). That part is benign on its own (just zero-initialized container allocation), but the file's purpose IS the GOT-patch installer, so importing this `.o` brings the priority-101 init along whether we want it or not.

**Why this is FLAG (same root as 4.1):**

cipher_rt_phase4's GOT patcher and cipher_intercept_cudart.cpp's GOT patcher are two implementations of the same idea. Linking both into libcipher_rt.so creates either (a) double-patching (both apply, last-writer-wins, non-deterministic), or (b) lock-out (the first to install detects its own slot already patched and aborts/skips — depending on its self-check logic).

### 4.3 Side note — cipher_intercept_cudart.o also defines NCCL hook

```
nm grep ncclAllReduce → 0x000000  ncclAllReduce  702 bytes  GLOBAL
                                  cipher_ncclAllReduce_impl  702 bytes  GLOBAL
```

The file defines `ncclAllReduce` as a hook target plus its impl. If libcipher_rt.so exports `ncclAllReduce`, any process that LD_PRELOADs libcipher_rt.so will have its NCCL replaced. This is a deferred-NCCL design intent (per the v1.2.2 §8.4 NCCL deferral adjudication: "Option B scope, NCCL hardware-deferred"), so exporting ncclAllReduce contradicts the scope-lock. Informational; surfaces independently of the GOT conflict.

---

## 5. Cross-reference with cipher_rt_phase4 load-time environment

cipher_rt_phase4's libcipher_rt.so is loaded in three contexts:

1. **`/bin/true` LD_PRELOAD loader smoke** — CUDA is not initialized. cipher_so_init()'s call to cipher_intercept_init() would call dlopen("libcuda.so") + resolve cuGetProcAddress. On a pod with cipher_kmod loaded but no CUDA application running, dlopen of libcuda is safe (driver is loaded), but the cuGetProcAddress redirection now races with cipher_rt_phase4's lazy GOT patcher when CUDA *is* later initialized by an inference process. The race is not visible until first kernel launch.

2. **cp54_isolation_test context** — directly hits `/dev/cipher` kmod ioctls; does NOT use libcipher_rt.so's CUDA path. The hook installer's run at load time is wasted work here but does not actively interfere (the test's interaction with kmod is unaffected).

3. **Python+torch inference (production target)** — both interception subsystems would race. The Wave 2 LP-2 SDPA work and the T4.5.1 cuBLAS matmul substrate both depend on the GOT patcher being the sole hook installer.

The Step 1 LP-2 smoke (`tramp_calls=1 handled=0 passthrough=1`) succeeded only because the GOT patcher was the sole installer; with the F1 hook layer also running, the trampoline accounting would not be reliable.

---

## 6. Recommendation per discipline

Per brief rubric: **FLAG count ≥ 1 → ADJUDICATION REQUIRED**, no auto-wrap, per-file decision.

The two FLAG files are not "subtle" — they are alternative implementations of the same architectural role (CUDA driver / cudart hook installer) that cipher_rt_phase4's substrate already implements with a different mechanism (GOT/PLT patching at first-call lazy time, retiring LD_PRELOAD link-order interposition per CP 2.5 D2(iii)).

Surfaced for adjudication. The audit reproduces the relevant facts; the decision on how to handle the two FLAG files is structural, not paperwork:

- Both files are pulled into Option A's closure by transitive symbol refs (the symbol-closure audit showed cipher_intercept_cudart.cpp has in-degree 12 — the most-referenced TU in the tree; cipher_intercept.cpp is referenced indirectly).
- Replacing them with no-op stubs would break a large fraction of their callers (every TU that uses the `cipher_intercept_*` symbol surface, plus any TU using `_ZL21log_memcpy_if_probing` or `cuLibraryGetKernel` etc.).
- Option B's stub set (4 symbols, primarily g_cipher + 3 flow hooks) *did not* sever the cipher_intercept dependency. In fact, cipher_intercept_cudart.cpp was inside Option B's 9-file closure too.

This finding affects ALL of Options A / B / B′ / C+ — only Option C (the 4 already-clean .cpp) avoids cipher_intercept_cudart.cpp entirely.

No mitigation proposed in this document.

---

## 7. Telemetry on disk

- `/tmp/option_a_closure.txt` — the 40-file Option A manifest
- `/tmp/week2_step2_v3/file_to_obj.txt` — source-to-.o mapping
- `/tmp/week2_step2_v3/static_init_manifest.json` — Step A scan in JSON
- `/tmp/week2_step2_v3/step_a_scan.txt` — Step A table
- `/tmp/week2_step2_v3/step_b_trace_full.txt` — Step B per-file disassembly (truncated at 300 lines in display; full ~3 MB on disk)
- `/tmp/static_init_scan.py` — reusable Step A scanner
- `/tmp/static_init_trace.py` — reusable Step B tracer

---

## 8. Discipline notes

- Read-only diagnostic. No source-tree changes. cipher_rt_phase4 working tree still dirty from v2 attempt; rollback in `WEEK_2_STEP_2_V2_RESULT.md` §7 still applies.
- No commit / tag rotation in any of the 4 trees.
- No mitigation proposed; only facts + categorization + the surface for adjudication.
- The Step A scanner and Step B tracer are reusable. Recommend adding to the standard pre-flight workflow next to `symbol_closure.py`.

---

## 9. Awaiting

User adjudication on the 2 FLAG files. The structural choices visible from the audit (no recommendation):

- **A1** — port cipher_intercept.cpp + cipher_intercept_cudart.cpp as-is; accept the GOT-patcher race; the architecture explicitly excludes this (CP 2.5 D2(iii)) so this would reverse a prior closed adjudication
- **A2** — port both files but **strip** their `.init_array` (link-time `-Wl,--exclude-libs` or per-file `__attribute__((constructor(0)))` defeat) — keeps the symbols available but disables auto-install; cipher_rt_phase4 explicitly calls `cipher_intercept_init()` only when it wants to (or never)
- **A3** — replace both files with hand-written stubs that provide the symbol surface but no hook installation
- **A4** — exclude both files from the port (effectively narrows Option A's closure by removing the dispatch path's dependency on cipher_intercept_*); 12 transitive files vanish since cipher_intercept_cudart.cpp has in-degree 12
- **A5** — abandon Option A; re-evaluate against Options B / B′ / C / C+ now that this finding is visible (Option C is the only path that doesn't touch cipher_intercept_cudart.cpp at all)

The decision is structural and out of scope.
