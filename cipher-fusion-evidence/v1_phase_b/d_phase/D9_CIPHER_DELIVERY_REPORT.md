# D.9 CIPHER-DELIVERY — prereq FIXED; transparent FP8 delivery BLOCKED (binding finding)

**Date:** 2026-05-29. **Measure-only; anchors UNCHANGED; no FP8 actuator built** (it would see 0
GEMMs — see §2). The delivery contract = CIPHER transparently engaging FP8 at the intercept
boundary with vLLM unmodified. **Prerequisite passed; the delivery itself is blocked because vLLM's
prefill GEMMs bypass CIPHER's intercept entirely.** Per the directive ("if CIPHER can't engage,
that's the binding finding, surface it").

## §1 PREREQUISITE — FIXED + VERIFIED ✅
The CIPHER vLLM plugin no longer fails under the container's py3.12.
- **Root cause (precise):** `cipher_vllm_kv.py:57-59` does `sys.path.insert(0, "/home/ubuntu/
  cipher_rt_phase4")`; that dir's bare `cipher_kv_bridge.so` is a **symlink → the py3.10 build**
  (`5a3db034`), which shadowed the working py3.12 `.so` in dist-packages → `import cipher_kv_bridge`
  in vLLM's EngineCore worker hit the py3.10/3.12 mismatch.
- **Fix (no rebuild):** placed the working py3.12 `.so`
  (`cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so`, from the CDI-injected dist-packages build)
  into `/home/ubuntu/cipher_rt_phase4/` so the plugin's path-insert resolves the version-tagged
  py3.12 module (Python prefers the cpython-312 tag over the bare symlink). Persists on the host mount.
- **Verified:** vLLM init with **CIPHER plugins ACTIVE** (no `VLLM_PLUGINS=""`, no `unset`):
  **0 import errors, 57 cipher log lines, vLLM started AND generated** (`VLLM_UP_WITH_CIPHER_PLUGINS_
  ACTIVE`). The plugin loads. (This is a real durable win — CIPHER's KV-dedup/telemetry plugins can
  now run under the container vLLM for V.1.)

## §2 BINDING FINDING — vLLM's prefill GEMMs BYPASS CIPHER's intercept ⛔
With the plugin loaded and CIPHER active, a full 8×2048 vLLM prefill (real big GEMMs) was run with
CIPHER's intercept telemetry captured. **CIPHER intercepted ZERO GEMMs:**
- `MATMUL (cublasGemmEx): calls=0 handled=0 passthrough=0` — the actuatable GEMM path saw nothing.
- attention intercept: `tramp_calls=0`; classifier: `obs=0`; no BILLING GEMM dispatches.
- The GOT-patches ARE installed and armed (`CUBLAS-SHIM: real cublasGemmEx=0x… ltMatmul=0x… resolved`;
  `LT-VARIANT: 17/17 trampolines armed`) — so this is **not** a "CIPHER didn't load" failure. **vLLM
  simply never calls the public cuBLAS symbols CIPHER patches.**

**Mechanism (matches the D.10 abandonment + the D.7/§1b finding):** vLLM/torch on cu13/H100 dispatch
the prefill GEMMs through **libcublas-internal `nvjet`** kernels, not the public `cublasGemmEx` /
`cublasLtMatmul` symbols. CIPHER's GOT-patch interception is at the **public cuBLAS API**; the
internal nvjet dispatch is below it and was the **D.10-abandoned** interception depth ("torch
resolves cublasLt entries at init before the hook arms; the real dispatch is libcublas-internal").
CIPHER's CUPTI path observes the launches but **cannot substitute** a kernel (profiling callback, not
a replacement point).

**⇒ CIPHER cannot transparently substitute FP8 (or any GEMM actuator) on vLLM's prefill GEMMs at the
current intercept layer.** Building the FP8 actuator is pointless — it would see 0 calls.

## §3 Verdict + disposition (STOP at gate)
**DELIVERY BLOCKED — binding finding (intercept bypass), not the plugin (fixed).**
- **Reachability (accepted, unchanged):** vLLM-native FP8 = ~88% per-channel MFU vs 989 at +0.37%
  PPL. FP8 prefill *can* deliver the MFU at the lossless-grade bar — **but vLLM executes it, not
  CIPHER.** That is NOT the transparent-CIPHER-delivery contract.
- **Delivery (this task):** CIPHER's transparent intercept-level FP8 engagement on vLLM is **not
  achievable on this stack** — vLLM's hot GEMMs do not traverse CIPHER's public-symbol intercepts.

**Honest paths forward (none is "transparent CIPHER FP8 on unmodified vLLM tonight"):**
1. **Intercept the nvjet path vLLM actually uses** (libcublas-internal dispatch / recognize+replace
   nvjet GEMM kernels at `cuLaunchKernel`). This is the D.10-abandoned depth — high cost, was dropped
   for good reason; revisit only as a dedicated substrate track.
2. **vLLM-native FP8 with CIPHER as the policy/recipe layer** (CIPHER's classifier *decides* the FP8
   config; vLLM *executes* it via its quant path). Delivers the MFU+quality, but FP8 is run by vLLM,
   not transparently by CIPHER — a weaker "delivery" claim; honest if framed that way.
3. **Patch vLLM's Linear** — violates Mem #24 (not intercept-level / not transparent). Rejected.

**The honest product statement:** FP8 prefill reachability is proven (~88% MFU at +0.37% PPL via
vLLM-native FP8). CIPHER's *transparent intercept-level* delivery of it on vLLM is **blocked** because
vLLM's GEMM dispatch (nvjet) bypasses CIPHER's public-cuBLAS-symbol intercepts. The plugin loads (V.1
can use CIPHER's KV/telemetry plugins under vLLM), but GEMM *substitution* through the intercept does
not reach vLLM's prefill.

## §4 Discipline
Measure-only; anchors UNCHANGED (rt `01d4effb`/`d7-rh1-close`, kmod `0.7.0`); no rotation; no actuator
built (would be inert). The 0.3%→0.37% PPL bar amendment is moot here (delivery blocked before the
quality gate). Prereq fix (`cipher_kv_bridge.cpython-312…so` in `cipher_rt_phase4/`) is additive,
persists, and unblocks CIPHER's non-GEMM plugins under vLLM. Evidence: `d9_prereq2.log` (plugin
ACTIVE, 0 errors), `d9_feasibility.log` (cublasGemmEx calls=0, obs=0 on vLLM prefill).
