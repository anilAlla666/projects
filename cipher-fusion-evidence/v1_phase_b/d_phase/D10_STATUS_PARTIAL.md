# D.10 LT-ROUTE — STATUS: routing BUILT + correct; blocked by a DEEPER interception gap

**Date:** 2026-05-29. **NOT closed — anchor NOT rotated** (`9fe23143`). PARTIAL with the
exact gap named (per Anil: not forced, not faked). Substrate edit: cipher_rt_phase4
`ed130e7` (on the D.7 checkpoint `7f9f98c`/`8b5e928`). D.10 container build `01d4effb`.

## BUILT + verified-correct (default-OFF)

- `cipher_rt_cublaslt_layout.c/.h` — decode M/N/K/dtype/transa/ld from the opaque cublasLt
  descriptors via the public `cublasLtMatmulDescGetAttribute`/`cublasLtMatrixLayoutGetAttribute`;
  conservative (returns -1 on any GetAttribute failure or dim inconsistency → the shim passes
  through; Mem #11 — never route on a bad decode).
- `cipher_rt_matmul_try_actuators()` — the actuator chain factored out of
  `cipher_rt_matmul_dispatch` (no gemmEx-shaped passthrough), so non-gemmEx callers route.
- HSH full shim (gated `CIPHER_LT_ROUTE`; OFF = telemetry passthrough = byte-identical vanilla)
  → decode → try_actuators → HANDLED?return:real; registered in place of the HSH trampoline.
- Compiles host+container, loads, `LT-LAYOUT resolved`, `17/17 armed (1 routed, route=ON)`.
- Build-flag bug found+fixed en route: d10 needed the `-Wl,-rpath,<cupti>` the d7 build had
  (without it the `.so` failed to dlopen on `libcupti.so.12` → CIPHER silently didn't load).

## PARTIAL — the routed shim NEVER FIRES (deeper interception gap)

Validation (LT_ROUTE=1, B=8 Mistral fp16, d10 staging bind-mounted):
- **`HSH_shim_calls=0`, `variants_aggregate=0`, `umbrella_ltMatmul=0`** (gemmEx=4500).
- cuBLAS's own log shows torch driving **1800 `cublasLtHSHMatmul`** — so torch DOES use cublasLt
  heavily, but **none of those calls reach any CIPHER cublasLt hook.**
- CIPHER GOT-registers BOTH `cublasGemmEx` (fires, 4500) and `cublasLtMatmul`
  (`cipher_rt_cublas_shim.c:317-324`) — but the cublasLtMatmul umbrella registers 0 calls.

**Root cause (one level below D.10's premise):** torch's cu13 build resolves the cublasLt
entry **once at cublasLt-init — before CIPHER's CUDA-injection hook arms — and calls the real
pointer directly.** So neither the GOT-patch on libtorch_cuda's `cublasLtMatmul` (the call
doesn't traverse that GOT entry at runtime) nor the dlsym-hook on the typed variants (torch
doesn't runtime-dlsym them) intercepts it. D.10's premise ("torch routes via dlsym at runtime"
/ "the umbrella is GOT-reachable") does not hold for this container's torch.

**Consequence:** D.10's routing (decode + dispatch + shim) is built and correct-by-construction,
but the **shim never executes** → the close gate (`marlin_engage>0` + big GEMMs observed via the
route) is **NOT met**. Marlin/all matmul-routed actuators remain blind in-container — which still
**gates D.7-positive and all of V.1.**

## Next-level fix surface (for Anil)

The unsolved piece is **intercepting torch's actual cublasLt call origin** before torch resolves
it. Candidate substrate-line approaches (need a scoping memo):
1. **LD_PRELOAD-level bind** of `cublasLtMatmul` (+ the typed variants) so the dynamic linker
   binds CIPHER's definitions FIRST, before torch's cublasLt-init resolves them. (CIPHER today
   loads via `CUDA_INJECTION64_PATH`, which may run after torch's cublasLt init — hence the miss.
   A genuine `LD_PRELOAD=libcipher_rt.so` ordering, or an early-init hook, may be required.)
2. **GOT-patch the real calling library** (not just libtorch_cuda) — identify which loaded lib's
   GOT carries the cublasLt call (libtorch_cuda_linalg / a cu13 dispatch lib) and patch it; will
   NOT work if the call is libcublasLt-internal (no GOT).
3. Confirm the call origin first (e.g., a backtrace from an LD_PRELOAD'd cublasLtMatmul, or
   `gdb`/perf on a single call) to choose (1) vs (2).

## Disposition

D.10 routing **BUILT + correct + default-OFF safe**, committed (`ed130e7`); **blocked by the
deeper "torch resolves cublasLt before CIPHER arms" interception gap** — PARTIAL, anchor
unrotated, marlin_engage not faked. **Options:** (1) authorize a follow-on interception substep
(confirm the call origin → LD_PRELOAD-ordering / right-lib GOT-patch); (2) reassess whether the
Marlin INT4 path is on Goal-1's critical path before more interception work (Goal-1 is bursty
decode; Marlin's INT4 lever is Goal-2 tok/W — VOLT/clock may carry Goal-2 without Marlin). No
anchor rotation, no V.1, until in-container actuator engagement is real.
