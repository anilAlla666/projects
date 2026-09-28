# CP 2.1 — libcipher_hook port/keep/delete inventory — REPORT

**Date:** 2026-05-15. **Status:** read-only audit complete — **no code written**.

CP 2.1 canonical scope (verbatim, phase-2 audit): *"libcipher_hook audit,
port/keep/delete inventory."* The gate: for each intercepted symbol / shim of
the existing `libcipher_hook.so` LD_PRELOAD layer, decide whether it **ports**
to the v2 injection model, **stays** as a dev-only shim, or is **dropped** —
with rationale per item.

This is an atomic read-only STEP. Method: `nm`, `readelf`, `objdump`, `cmp`,
`grep`, file reads, two parallel source-digest agents. No file on disk was
modified; no workload was run.

---

## 1. Audit subject

| Item | Path | md5 |
|---|---|---|
| `libcipher_hook.so` (subject) | `cipher-may13-evidence/libcipher_hook.so` | `4841480aa28bb8d1a04f6f282ad6815d` |
| `.preroadmap` backup | `cipher-may13-evidence/libcipher_hook.so.preroadmap` | `4841480aa28bb8d1a04f6f282ad6815d` |
| `exports.map` (hook version script) | `cipher-may13-evidence/exports.map` | `8331bf36adbe6cd95409eda3ca389c44` |
| `hook_versions.map` | `cipher-may13-evidence/hook_versions.map` | `3ffacf530c174ac1ce5dd0b67053b655` |
| `Makefile` | `cipher-may13-evidence/Makefile` | `c86f10e821cda32c5846533e7f215269` |

138 576 bytes, ELF x86-64, not stripped, BuildID `3965722021…`. The
`.preroadmap` copy is **byte-identical** (`cmp` → IDENTICAL) — a backup, not a
separate audit subject. The hook is the old op31-prod LD_PRELOAD stack; it was
never inventoried in a Phase 2 document (hence CP 2.1 NOT DONE until now).

**The hook is built from exactly 7 source files** (`Makefile` `HOOK_SRC`):
`cipher_intercept_cudart.cpp` (138 KB — the interceptors), `cipher_persist.cpp`,
`cipher_graph_inspect.cpp`, `cipher_kernel_table.cpp`, `cipher_flow_recorder.cpp`,
`cipher_flow_patterns.cpp`, `cipher_flow_substitute.cpp`. The other ~70
`src/*.cpp` build `libcipher_rt.so`, **not** the hook — they are out of CP 2.1
scope. 52 exported `T` symbols (`nm -D`).

---

## 2. The classification framework

The v2 model is two carriers, because one mechanism cannot cover both layers:

- **`libcipher_v2.so`** — loaded by the CUDA driver via `CUDA_INJECTION64_PATH`
  at `cuInit` (`InitializeInjection2`). It receives the driver's export table
  and patches it. It can hook the **driver API (`cu*`)** — and *only* that.
  CP 2.2 shipped this skeleton.
- **`libcipher_rt.so` `.symver` substrate** — a targeted LD_PRELOAD/`.symver`
  layer for **userspace libraries** (cuBLAS, NCCL, ATen SDPA) that the driver
  injection model is structurally blind to. T4.5.1 / T4.6.1 already shipped
  this substrate.

**PORT** — production-needed; re-implemented under the v2 model. Rationale names
the destination carrier (`→ v2` or `→ rt substrate`).
**KEEP** — retained as a dev/CI-only LD_PRELOAD shim; not in the production path.
**DELETE** — dropped: experimental, dead, stub, superseded, or
LD_PRELOAD-mechanism-specific (meaningless once LD_PRELOAD is gone).

---

## 3. Structural findings (before the per-symbol table)

1. **`cublasGemmEx` is implemented but NOT exported.** Its body lives in
   `cipher_intercept_cudart.cpp:584`, but `cublasGemmEx` is **absent from the
   `exports.map` `global:` clause** — so it is not promoted to a globally
   interposable dynamic symbol (`nm -D` confirms: not in the export list).
   LD_PRELOAD symbol-shadowing therefore **never catches cuBLAS GEMM**; only
   the hook's GOT-patching path (`dl_iterate_phdr` rewriting caller GOT slots)
   could — and GOT patching from outside `libcublas` does not reach callers
   internal to it. Treat the hook's LD_PRELOAD GEMM interception as
   **effectively non-load-bearing**.
2. **`hook_versions.map` is an orphan.** 167 bytes, `.symver`-style sub-clauses
   for `libcublas.so.12` / `libcublasLt.so.12` / `libnccl.so.2` — but the
   `Makefile` feeds **only `exports.map`** to the hook link (line 67). It is
   stale Phase-2-era infrastructure, plausibly an unfinished pivot toward
   `.symver` interception — which T4.5 later actually built. Dead file.
3. **`exports.map` is over-broad.** Its `global:` clause lists ~80 symbols; the
   hook defines only 52 of them. The `cipher_koopman_*`, `cipher_attn_*`,
   `cipher_param_*`, `cipher_marlin` entries are defined in `libcipher_rt.so`,
   not the hook — the version script silently ignores listed-but-undefined
   names. The map is shared/over-scoped, not hook-specific.
4. **Weak undefined refs** — `cipher_attn_fsm_on_gemm/on_kernel`,
   `cipher_attn_koopman_enabled`, `cipher_f1_ring_write` — are satisfied by
   `libcipher_rt.so` at load; the attention-FSM / Koopman path is rt-side
   experimental, not hook surface.

---

## 4. Inventory — Driver API (`cu*`) → carrier: `libcipher_v2.so`

| # | Symbol / shim | Class | Rationale |
|---|---|---|---|
| 1 | `cuLaunchKernel` | **PORT → v2** | Core launch path: launch/tenant accounting + `cipher_dispatch` + persist fast-path + L2 access-policy-window injection. Production-load-bearing. The launch hook the v2 injection lib must carry. |
| 2 | `cuLaunchKernelEx` | **PORT → v2** | Extended-config launch; injects `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` into the config struct. Production. |
| 3 | `cuLaunchKernel_ptsz` | **PORT → v2** (coverage only) | No dedicated logic — pure count+forward. The v2 lib patches the `_ptsz` proc-table entry alongside the base entry; coverage is free, no shim code carries over. |
| 4 | `cuLaunchKernelEx_ptsz` | **PORT → v2** (coverage only) | Same — pure passthrough; coverage via table patch. |
| 5 | `cuModuleLoadData` | **PORT → v2** | Feeds the param-recovery metadata ingest (`cipher_param_ingest_image`). Ports. The LD_PRELOAD-specific auto-GOT-repatch side-effect is **dropped** — injection patches the driver table once; no GOT walking. |
| 6 | `cuModuleLoadDataEx` | **PORT → v2** | As above, options variant. |
| 7 | `cuModuleGetFunction` | **PORT → v2** | Feeds `cipher_param_register_cufunc`. Metadata pipeline. |
| 8 | `cuLibraryLoadData` | **PORT → v2** | CUDA-12+ module path — the path torch 2.11 actually uses. Modern primary for metadata ingest. |
| 9 | `cuLibraryGetKernel` | **PORT → v2** | CUDA-12 kernel-name mapping for the metadata pipeline. |
| 10 | `cuKernelGetFunction` | **PORT → v2** | `CUfunction → CUkernel` map for name lookup. |
| 11 | `cuGetProcAddress` | **PORT → v2** (becomes the mechanism) | Under LD_PRELOAD this is exported and returns hook trampolines — that *is* the interception pivot. Under `CUDA_INJECTION64_PATH` the driver hands the injection lib the proc table directly via `InitializeInjection2`; cuGetProcAddress interception is **replaced by the injection mechanism itself**. The intent ports; the symbol ceases to exist as an exported hook. |
| 12 | `cuGetProcAddress_v2` | **PORT → v2** | Same — CUDA-12 versioned variant. |
| 13 | `cuGraphInstantiate_v2` | **KEEP** (dev/CI shim) | `cipher_graph_inspect.cpp` — read-only post-capture graph introspection, env-gated `CIPHER_GRAPH_INSPECT` (default off). Not an actuator; was the tool that found the Phase-3 stream-capture bug. Retain as a dev/CI audit shim, not in the production injection path. |
| 14 | `cuGraphInstantiateWithFlags` | **KEEP** (dev/CI shim) | As above, flags variant. |

---

## 5. Inventory — Runtime API (`cuda*`, `__cuda*`) → mostly DELETE

| # | Symbol / shim | Class | Rationale |
|---|---|---|---|
| 15 | `cudaGraphInstantiate` | **KEEP** (dev/CI shim) | Runtime-side pair of `cipher_graph_inspect.cpp`; same dev/CI introspection shim as items 13–14. |
| 16 | `cudaGraphInstantiateWithFlags` | **KEEP** (dev/CI shim) | As above. |
| 17 | `cudaLaunchKernel` | **DELETE** | Redundant under driver injection — `cudaLaunchKernel` internally calls `cuLaunchKernel`, which the v2 hook (item 1) catches. The `CIPHER_KERNEL_PROBE` / `CIPHER_KERNEL_TRACE` diagnostics carried in `cudaLaunchKernel_dispatch()` are **deleted with the symbol** — their only consumers (the flow trio, items 31–33) are themselves DELETE, so the probes have no surviving consumer. Do not relocate them. |
| 18 | `cudaLaunchKernel_ptsz` | **DELETE** | As item 17. |
| 19 | `__cudaLaunchKernel` | **DELETE** | torch-2.11 host-stub launch path; same driver-layer redundancy. |
| 20 | `__cudaLaunchKernel_ptsz` | **DELETE** | As item 19. |
| 21 | `cudaLaunchKernelExC` | **DELETE** | Runtime extended-config launch; caught one layer down at `cuLaunchKernelEx`. |
| 22 | `cudaLaunchKernelExC_ptsz` | **DELETE** | As item 21. |
| 23 | `__cudaRegisterFatBinary` | **DELETE** | Legacy cudart static-registration metadata path. The driver-layer `cuLibraryLoadData` / `cuModuleLoadData` (items 5–8, PORT'd) already supply fatbin metadata to param-recovery. The LD_PRELOAD GOT-repatch it triggers is obsolete under injection. No non-cuLibrary framework in the current stack needs it. |
| 24 | `__cudaRegisterFatBinaryEnd` | **DELETE** | Pure passthrough on the legacy registration path. |
| 25 | `__cudaRegisterFunction` | **DELETE** | Legacy function-registration metadata; superseded by `cuModuleGetFunction` / `cuKernelGetFunction` (items 7, 10). |
| 26 | `cudaMemcpy` | **DELETE** | Only consumer is the `CIPHER_MEMCPY_PROBE` H2D weight-load diagnostic (default off, pure telemetry). Phase-1 probe, no production consumer. |
| 27 | `cudaMemcpyAsync` | **DELETE** | As item 26. |

---

## 6. Inventory — Library calls (cuBLAS / NCCL) → carrier: `libcipher_rt.so` substrate

| # | Symbol / shim | Class | Rationale |
|---|---|---|---|
| 28 | `cublasLtMatmul` | **PORT → rt `.symver` substrate** | cuBLAS is a userspace library the CUDA injection model **cannot see** — interception must remain a library-targeted `.symver` substrate. That substrate already exists (T4.5.1, `libcipher_rt.so`). The production-relevant payload — **FP8 compute substitution (Stage 13)** — ports into the substrate; that is CP 2.4 work. The hook's standalone LD_PRELOAD wrapper is superseded by the `.symver` substrate. |
| 29 | `cublasGemmEx` | **PORT → rt `.symver` substrate** (payload only) | See structural finding #1 — implemented but **not exported**, so LD_PRELOAD GEMM interception is non-load-bearing. The *payload* (FP8 compute substitution) ports into the rt cuBLAS substrate (CP 2.4). Its experimental decision-tree sub-paths — Koopman O(1) proxy, `CIPHER_KV_RDR_V3` redirect, weight-share, fairness-yield — are **DELETE** (see item 34). |
| 30 | `ncclAllReduce` | **PORT → NCCL tuner-plugin ABI** | NCCL is a separate library invisible to driver injection. NCCL provides a *supported* external tuner-plugin ABI; the stack already carries `libnccl-tuner-cipher.so` + `include/cipher_nccl_tuner_abi.h`. The CfC algo-decision + closed-loop-feedback payload ports onto that plugin ABI — and may already be superseded by `libnccl-tuner-cipher.so` (confirming that is a CP 2.4 prerequisite, outside this audit's subject). The LD_PRELOAD `ncclAllReduce` wrapper itself is dropped. |

---

## 7. Inventory — CIPHER control / telemetry surface (`cipher_*` — not interceptors)

| # | Symbol(s) | Class | Rationale |
|---|---|---|---|
| — | `cipher_intercept_count` | **PORT → v2** | Aggregate intercept counter; ports as a v2-lib telemetry stat. |
| — | `cipher_persist_enabled`, `_fast_path_count`, `_observe_count`, `_promotion_count`, `_report` | **PORT → v2** | Persistent-mode repeat-sequence fast-path (`cipher_persist.cpp`) — production hot-path, default ON (`CIPHER_PERSIST`). Logic moves into the v2 dispatch gate. Passive (no kernel mutation): recognises repeating launch sequences and skips dispatch overhead. |
| — | `cipher_kt_observe`, `cipher_kt_size`, `cipher_kt_dump_json`, `cipher_kernel_table_report` | **PORT → v2** | Kernel name/category registry (`cipher_kernel_table.cpp`), probed on every launch (~10 ns). Foundational name resolution. Ports. |
| — | `cipher_param_query_name`, `cipher_param_query_via_cuda` | **PORT → v2** | Kernel-name + param-info lookup; part of the metadata pipeline that items 5–10 feed. Ports with that pipeline. |
| — | `cipher_set_gemm_ptrs`, `cipher_tls_get_gemm_ptrs`, `cipher_tls_get_gemm_shape`, `cipher_tls_get_gemm_types`, `cipher_tls_relaunch` | **PORT → rt substrate** | TLS GEMM-shape pipeline + relaunch hook — the support machinery for cuBLAS substitution. Ports together with the cuBLAS substrate payload (items 28–29, CP 2.4). External consumers (e.g. `cipher_wrapper.py`) must be enumerated and preserved — that enumeration is a CP 2.4 prerequisite, not this audit. |
| — | `cipher_shim_tsc_total`, `cipher_shim_tsc_calls`, `cipher_shim_tsc_reset` | **DELETE** | These measure *LD_PRELOAD shim overhead* — the very cost CP 2.5 eliminates. Once LD_PRELOAD is gone the metric is meaningless. |
| — | `cipher_repatch` | **DELETE** | Manual GOT re-patch trigger. GOT patching via `dl_iterate_phdr` is LD_PRELOAD-specific; obsolete under injection (the v2 lib patches the driver proc table once at `InitializeInjection2`). |
| — | `cipher_read_sample`, `cipher_sample_wseq` | **DELETE** | EDMD ring-buffer sample readout — tied to the experimental EDMD subsystem (rt-side, not production). No production consumer. |

---

## 8. Inventory — whole-file components (the 7 `HOOK_SRC` files)

| # | File | Class | Rationale |
|---|---|---|---|
| — | `cipher_intercept_cudart.cpp` | **split** | Carries items 1–30 + most `cipher_*` rows. Splits PORT/DELETE per the tables above — not a single verdict. |
| — | `cipher_persist.cpp` | **PORT → v2** | Persistent-mode fast-path (rows in §7). Passive CPU-overhead reduction, default ON, production. |
| — | `cipher_kernel_table.cpp` | **PORT → v2** | Kernel registry (rows in §7). Foundational, every-launch. |
| — | `cipher_graph_inspect.cpp` | **KEEP** (dev/CI) | Read-only graph introspection (items 13–16). Useful CI audit tool; not production. |
| 31 | `cipher_flow_recorder.cpp` | **DELETE** | Stage 1 of an incomplete 3-stage RMSNorm-fusion research pipeline; env-gated off (`CIPHER_FLOW_RECORD`); only useful if Stages 2–3 are active. |
| 32 | `cipher_flow_patterns.cpp` | **DELETE** | Stage 2 — RMSNorm pattern matcher; experimental; depends on Stage 3. |
| 33 | `cipher_flow_substitute.cpp` | **DELETE** | Stage 3 — the substitution is a **non-functional SAFE STUB**: `cipher_flow_substitute_consider()` always returns 0 (passthrough). Real X/W/Y pointer extraction was never wired; blocked on the documented RoPE-timing correctness issue. It substitutes nothing in practice. Archive the source. |
| 34 | Experimental actuator sub-paths inside `cublasGemmEx`/`cublasLtMatmul` — Koopman O(1) proxy, `CIPHER_KV_RDR_V3`, attention-Koopman FSM, weight-share, fairness-yield | **DELETE** | All env-gated-off Phase-1 research. Only **FP8 compute substitution** is the production-relevant payload that ports (items 28–29). The rest is dropped. |

---

## 9. Tally

| Class | Count | What it means |
|---|---|---|
| **PORT** | 12 driver-API symbols + 3 library symbols + 5 `cipher_*` control groups | Survives into the v2 model — 14 → `libcipher_v2.so`, the rest → `libcipher_rt.so` substrate / NCCL tuner plugin. |
| **KEEP** | 4 graph-instantiate shims (`cipher_graph_inspect.cpp`) | Stays a dev/CI-only LD_PRELOAD audit shim. |
| **DELETE** | 11 runtime-API symbols + 3 `cipher_shim_tsc_*` + `cipher_repatch` + 2 EDMD-sample symbols + the flow trio (3 files) + experimental actuator sub-paths | Dropped: redundant-under-injection, LD_PRELOAD-mechanism-specific, experimental, or non-functional stub. |

**Headline:** the hook decomposes cleanly. The driver-API launch + module-load
+ metadata + persist + kernel-table surface **ports to `libcipher_v2.so`**. The
cuBLAS/NCCL surface **ports to the `libcipher_rt.so` `.symver` substrate** (FP8
payload only; CP 2.4). The runtime-API layer is **redundant** under driver
injection and is dropped. The 3-stage flow-fusion pipeline is an incomplete
research artifact (Stage 3 is a stub that never substitutes) and is dropped.
Graph introspection survives as a dev/CI tool. The LD_PRELOAD-mechanism
plumbing (GOT repatch, shim-TSC accounting) is dropped because the mechanism
itself goes away.

---

## 10. Handoff to CP 2.4 and CP 2.5

- **CP 2.4** ("Marlin + DVFS + speculative ported") consumes the **PORT** rows.
  The single production payload to port from the hook's library interceptors
  is **FP8 compute substitution** (items 28–29) → the `libcipher_rt.so` cuBLAS
  substrate. Prerequisite before CP 2.4 touches code: enumerate the external
  consumers of `cipher_set_gemm_ptrs` / `cipher_tls_*` (e.g. `cipher_wrapper.py`)
  so the TLS GEMM pipeline ports without breaking callers.
- **CP 2.5** ("LD_PRELOAD path dropped, `cipher-platform.deb`") consumes the
  **DELETE** rows: the runtime-API layer, GOT-repatch / shim-TSC plumbing, and
  the flow trio leave the production build. **KEEP** rows (`cipher_graph_inspect`)
  ship as a separate dev/CI shim, not in the `.deb`'s production injection path.
- **Open item for CP 2.4**, flagged not resolved: confirm whether
  `libnccl-tuner-cipher.so` already implements the NCCL tuner-plugin port
  (item 30) — if so, `ncclAllReduce` is DELETE rather than PORT.

---

## 11. Discipline — anchors held

CP 2.1 is a **read-only audit. No code written. No file modified. No workload
run.** Verification tools only (`nm`, `readelf`, `cmp`, `grep`, file reads).

- kmod **0.4.7** `2a69f9defd7730665e6b7f9d60e82b43` — unchanged.
- libcipher_v2 anchor `86618c30896470b642fcc6985d8dc632` — unchanged.
- kmod anchor `55ab8c0c` — untouched. ABI ioctl nrs — untouched. Taint 12288.
- The audit subject (`cipher-may13-evidence/`) was read, never written.

---

## 12. Artifacts

| Artifact | Path | md5 |
|---|---|---|
| this report | `cp_2_1/CP_2_1_REPORT.md` | (self) |
| audit subject | `cipher-may13-evidence/libcipher_hook.so` | `4841480aa28bb8d1a04f6f282ad6815d` |
| hook version script | `cipher-may13-evidence/exports.map` | `8331bf36adbe6cd95409eda3ca389c44` |
| orphan version map | `cipher-may13-evidence/hook_versions.map` | `3ffacf530c174ac1ce5dd0b67053b655` |
| build rules | `cipher-may13-evidence/Makefile` | `c86f10e821cda32c5846533e7f215269` |

`HOOK_SRC` source md5s (audit inputs, unmodified): `cipher_intercept_cudart.cpp`
`75c55528…`, `cipher_persist.cpp` `4b4fcc24…`, `cipher_graph_inspect.cpp`
`05b9ac71…`, `cipher_kernel_table.cpp` `0128be66…`, `cipher_flow_recorder.cpp`
`915ca5aa…`, `cipher_flow_patterns.cpp` `688b3942…`, `cipher_flow_substitute.cpp`
`e221a99c…`.

---

## 13. Bottom line

CP 2.1 delivers the port/keep/delete inventory the gate requires: every one of
the hook's 52 exported symbols and 7 source components is classified with
rationale. The hook is **not** monolithic — it decomposes along the v2 model's
two carriers: driver-API interception → `libcipher_v2.so` (CUDA injection),
library interception → `libcipher_rt.so` `.symver` substrate. The runtime-API
layer is redundant under driver injection and the 3-stage flow-fusion pipeline
is a non-functional research stub — both are dropped. Two structural findings
surfaced: `cublasGemmEx` is implemented but unexported (LD_PRELOAD never caught
cuBLAS GEMM), and `hook_versions.map` is dead Phase-2 infrastructure. The
inventory is the decision input for CP 2.4 (port the FP8 payload) and CP 2.5
(drop the LD_PRELOAD surface). No code was written; anchors held.

---

## ADDENDUM — 2026-05-15 (resolved during CP 2.4 audit-before-build)

CP 2.1 §10 left one open item: confirm whether `libnccl-tuner-cipher.so`
supersedes the LD_PRELOAD `ncclAllReduce` port (inventory **item 30**, §6).
CP 2.4's audit-before-build STEP resolved it.

**Evidence.** `nm -D /home/ubuntu/cipher-may13-evidence/libnccl-tuner-cipher.so`:

```
0000000000004060 D ncclTunerPlugin_v1
0000000000004080 D ncclTunerPlugin_v2
```

`libnccl-tuner-cipher.so` exports `ncclTunerPlugin_v1` and `ncclTunerPlugin_v2`
— the **NCCL external-tuner-plugin ABI structs** (type `D`, data symbols). That
is the *supported* NCCL tuner mechanism (`NCCL_TUNER_PLUGIN`), requiring no
LD_PRELOAD and no `ncclAllReduce` symbol interposition.

**Resolution — inventory item 30 (`ncclAllReduce`): PORT → DELETE.** The CfC
algo-decision / closed-loop-feedback function is already re-homed onto the
supported NCCL tuner-plugin ABI by `libnccl-tuner-cipher.so`. The hook's
LD_PRELOAD `ncclAllReduce` wrapper is therefore **superseded**, not ported —
it is dropped with the rest of the LD_PRELOAD surface at CP 2.5.

This addendum is appended, not a rewrite: §1–§13 above are the original
audit as adjudicated. Only item 30's classification changes — PORT → DELETE —
on the strength of evidence gathered after the original audit closed.
