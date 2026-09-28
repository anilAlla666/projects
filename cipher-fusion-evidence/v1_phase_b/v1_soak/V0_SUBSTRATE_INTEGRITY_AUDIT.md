# CIPHER substrate-integrity audit — substrate-level vs app/framework-specific

**Date:** 2026-05-31. **Type:** READ-ONLY (cite file:line; no build, no commit, anchors unchanged).
Question: does CIPHER intercept at framework-agnostic primitives (→ substrate, vLLM is one workload) or
key on framework-specific symbols (→ per-framework adapter that breaks on SGLang/TGI/raw-torch)?

**Headline:** CIPHER is **MIXED — and the split is clean by substrate.** The **clock (VOLT), matmul
(cuBLAS API), classifier (driver shapes), and the KV-dedup PRIMITIVE (cuMemMap)** are genuine
framework-agnostic substrate — they transfer to any cuBLAS/CUDA stack free. The **attention engagement is
a vLLM-specific ADAPTER** — it hardcodes vLLM's compiled `.so` paths and mangled symbols (exactly Anil's
concern). The **KV-dedup ENTRY is vLLM-specific** (a vLLM plugin), though its primitive is substrate. And
the **one true CUDA-driver-dispatch boundary CIPHER touches (`cuLaunchKernel`) is OBSERVE-only** — no
substitution there. So: substrate at the cuBLAS/clock/VMM layer; per-framework adapter at attention + KV
entry.

---

## Per-substrate classification (cited)

### 1. matmul — SUBSTRATE (cuBLAS-library API; framework-agnostic among cuBLAS users)
- `cipher_rt_cublas_shim.c:327-328` GOT-registers the symbol **`"cublasGemmEx"`** (and `cublasLtMatmul`,
  `:120`) — the **cuBLAS-library API primitive**, not a framework symbol. Any stack (torch/vLLM/SGLang/TGI/
  raw-torch) that calls cuBLAS hits these → **transfers free**. `cublasGemmEx` routes to the actuator
  dispatch (`:231`).
- **CAVEAT (it's the LIBRARY boundary, not the driver primitive):** cuBLAS-internal **`nvjet`** kernels +
  Triton/CUTLASS-direct GEMMs dispatch *below* cuBLAS (via the driver) and **bypass** `cublasGemmEx` — the
  prefill-nvjet finding (`D9_CIPHER_DELIVERY_REPORT.md:33-37`). So matmul is "substrate-grade for cuBLAS
  GEMMs," not universal. **Verdict: SUBSTRATE (cuBLAS-API-level), framework-agnostic, bypassed only by
  non-cuBLAS GEMM paths — not by a framework change.**

### 2. attention — APP-SPECIFIC (both paths key on framework symbols)
- **`cipher_rt_attn_6pattern.c` is 100% vLLM-keyed:** `arm_pattern` `dlopen`s **hardcoded vLLM library
  paths** — `:159` `/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`,
  `:162` `_vllm_fa3_C.abi3.so`, `:166` `vllm/_flashmla_C.abi3.so`, `_C.abi3.so` — and `dlsym`s vLLM's
  mangled symbols (`SYM_P1-P6`, `:43-46`, e.g. `_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st`).
  **SGLang/TGI/raw-torch compile different attention libs with different symbols → these patches would not
  arm.** This is a **per-framework adapter** — exactly Anil's concern.
- **`cipher_rt_attn_dispatch.cpp` is torch-keyed:** intercepts torch ATen SDPA mangled symbols
  (`_ZN2at4_ops..._scaled_dot_product_*_attention::call`, `cipher_rt_attn_dispatch.h:11-15`) — torch-
  specific (and bypassed by vLLM, which doesn't use torch SDPA).
- **NO substrate-level attention substitution exists.** The driver-dispatch boundary (`cuLaunchKernel`) is
  intercepted only for **observe/classify**: CUPTI callback (`cipher_cupti.c:163`
  `CBID_cuLaunchKernel`) is a profiling hook (cannot substitute), and `cipher_rt_classify_substrate.cpp:11`
  — *"cipher_rt_classify_route() is NOT called from any hot path"* (unwired scaffold). **⇒ attention
  engagement is ENTIRELY app-specific today.**
- **Substrate-level alternative (v-next, NOT built):** intercept `cuLaunchKernel`/`cuLaunchKernelEx` (the
  driver-dispatch boundary EVERY attention lib bottoms out at) + recognize attention from the **launch
  signature** (cubin-name pattern / grid-block / operand shapes), framework-agnostically, + substitute.
  Today that boundary is observe-only. **Verdict: APP-SPECIFIC; substrate alternative is v-next.**

### 3. memory / KV-dedup — HYBRID (substrate primitive + app-specific entry)
- **Primitive = SUBSTRATE (CUDA driver VMM):** the dedup mechanism is `cuMemUnmap` + `cuMemRelease` +
  `cuMemMap` page-alias on CIPHER-VMM pages (`cipher_vllm_kvdedup.py:11`; the `cipher_kv_bridge` VMM
  allocator). These are **libcuda driver primitives** — framework-agnostic; the dedup itself is exact
  (KL=0).
- **Entry/trigger = APP-SPECIFIC (vLLM plugin):** registered under **`vllm.general_plugins`** and
  **monkey-patches `GPUModelRunner._allocate_kv_cache_tensors`** to locate the KV pages + trigger
  `dedup_now()` (`cipher_vllm_kvdedup.py:3-7,23-29,204-251`). A different framework needs a different
  plugin/entry to find its KV cache. **Verdict: substrate PRIMITIVE transfers; ENTRY needs per-framework
  porting.**

### 4. classifier (K.1) — SUBSTRATE (driver signals; no framework hints)
- `cipher_workload_observe_gemm(m, n, k, Atype, Btype, Ctype, stream)`
  (`src/cipher_workload_detect.cpp:1475`, `include/cipher_workload_detect.h:165`) is fed **GEMM shapes +
  dtype + stream/cadence** from the cuBLAS shim (`cipher_rt_cublas_shim.c:166`) — **driver/library signals,
  no framework-specific hint**. Classifies workload class from shape/cadence → framework-agnostic.
  **Verdict: SUBSTRATE** (caveat: fed via the cuBLAS shim, so it sees cuBLAS GEMMs — cuBLAS-library-level).

### (+) VOLT (tok/W) — SUBSTRATE (truly hardware/driver-level)
- `cipher_rt_volt.c:3,22-31`: NVML `nvmlDeviceSetGpuLockedClocks` + kmod ioctl `CIPHER_SET_CLOCK`
  (`cipher_ioctl.h`). **Zero symbol-keying** — pure device clock control. **The most framework-agnostic
  lever; transfers to any stack free.**

---

## THE VERDICT

| Intercept | Class | Transfers to SGLang/TGI/raw-torch? |
|---|---|---|
| **VOLT (clock)** | SUBSTRATE (NVML/kmod, hardware) | **YES, free** |
| **matmul (cublasGemmEx/Lt)** | SUBSTRATE (cuBLAS-API) | **YES, free** (any cuBLAS user; bypassed only by nvjet/Triton/CUTLASS-direct) |
| **classifier K.1** | SUBSTRATE (driver shapes) | **YES, free** (fed via cuBLAS shim) |
| **KV-dedup PRIMITIVE (cuMemMap)** | SUBSTRATE (driver VMM) | **YES** — primitive transfers |
| **KV-dedup ENTRY (vLLM plugin)** | APP-SPECIFIC (vllm.general_plugins + GPUModelRunner) | **NO** — needs a per-framework entry |
| **attention (6pattern, FA3 5280/cell)** | APP-SPECIFIC (hardcoded vLLM `.so` paths + symbols) | **NO** — per-framework adapter |
| **attention (attn_dispatch SDPA)** | APP-SPECIFIC (torch ATen symbols) | **NO** — torch-only (and vLLM-bypassed) |
| **cuLaunchKernel (true driver boundary)** | OBSERVE-ONLY (CUPTI/classify) | n/a — no substitution there |

**Is attention engagement substrate-level or app-specific?** **Entirely app-specific.** The W.5 FA3
engagement (5280/cell on vLLM) is the `_vllm_fa3_C.abi3.so` GOT-patch — it literally hardcodes vLLM's
install path and mangled symbol. It would not arm on SGLang/TGI/raw-torch. The substrate-level alternative
— `cuLaunchKernelEx` intercept + launch-signature recognition — is **NOT built** (the launch boundary is
observe-only today). **So attention is the one substrate where Anil's "per-framework adapter" concern is
literally true.**

**Is matmul genuinely framework-agnostic?** **Yes** — it keys on the cuBLAS API symbol `cublasGemmEx`, not
any framework. Caveat: it's the cuBLAS-*library* boundary (above the driver), so cuBLAS-internal nvjet +
Triton/CUTLASS-direct paths bypass it — but that is a *GEMM-path* limitation, not a *framework-coupling*
one (it doesn't break when the framework changes; it breaks when the GEMM library changes).

**How much of "CIPHER engages on vLLM" transfers free vs needs porting?**
- **SUBSTRATE — transfers free to any cuBLAS/CUDA stack:** VOLT tok/W (clock), the matmul intercept
  (cublasGemmEx — where actuators substitute), the K.1 classifier (driver shapes), the cuMemMap KV
  primitive. **The tok/W (VOLT) + GEMM levers are genuine substrate.**
- **APP-SPECIFIC — would break / need per-framework porting:** (a) the **attention engagement**
  (hardcoded vLLM `.so` paths — full re-patch for another framework, or build the cuLaunchKernel substrate
  alternative); (b) the **KV-dedup ENTRY** (vLLM plugin — needs a per-framework KV-locate entry, though the
  cuMemMap primitive is reused).

**Net — substrate or vLLM-coupled product?** **CIPHER is a substrate at the cuBLAS/clock/VMM/classifier
boundary** (vLLM is one validated workload there) **and a vLLM-specific adapter at the attention +
KV-entry boundary.** Two named app-specific intercepts + their substrate alternatives:
1. **Attention 6pattern (vLLM `.so` paths)** → substrate alt = `cuLaunchKernelEx` intercept + launch-
   signature attention recognition. **NOT built** (launch boundary is observe-only). v-next.
2. **KV-dedup ENTRY (vLLM plugin)** → substrate alt = a driver-level KV-cache recognizer (identify KV
   allocations by VMM size/pattern) feeding the existing cuMemMap primitive. **NOT built.** v-next.

To make CIPHER a true substrate for ALL three HBM levers (not just weights/clock), the **`cuLaunchKernel`/
`cuLaunchKernelEx` driver-dispatch boundary must move from observe-only to recognize-and-substitute** — the
single substrate investment that de-couples attention (and a launch-level fallback for nvjet-GEMMs) from
per-framework symbol-keying. Today: weights/clock/classifier = substrate; attention + KV-entry =
vLLM-adapter. Read-only — no build, no commit, anchors unchanged.
