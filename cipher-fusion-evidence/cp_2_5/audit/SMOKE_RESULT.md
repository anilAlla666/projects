# CP 2.5 audit smoke — cuBLAS shim under CUDA_INJECTION64_PATH-only

**Date:** 2026-05-16. Pre-memo audit evidence. `cp25_smoke.py` = 20× fp16
512×4096 @ 4096×4096 matmul on CUDA. libcipher_rt `5e304549`, CIPHER_MARLIN=on.

## ARM 1 — CUDA_INJECTION64_PATH only (no LD_PRELOAD)
- InitializeInjection chain runs **fully**: tenant, ARB, SMP, PR, CUPTI,
  MATMUL substrate init, MARLIN engine init + actuator registered + ENABLED,
  attn substrate active, GREEN ctx, VOLT — all CUDA_INJECTION64_PATH-native.
- `MATMUL: exit totals — calls=0 handled=0 passthrough=0` — **the cuBLAS shim
  NEVER FIRED.** All 20 matmuls bypassed libcipher_rt → real libcublas direct.
- `[cipher-attn] exit totals - tramp_calls=0` — attn trampolines never fired.
- CUPTI DID see launches (`total_launches=8`) — driver-level hook works.

## ARM 2 — LD_PRELOAD + CUDA_INJECTION64_PATH (current deploy mode)
- `CUBLAS-SHIM: real cublasGemmEx resolved` — shim active.
- `MATMUL: exit totals — calls=20 handled=0 passthrough=20` — **all 20 matmuls
  routed through the shim** (handled=0: M=512 > Marlin M≤64 gate, correct
  passthrough — but the shim INTERCEPTED every call).

## Finding
CUDA_INJECTION64_PATH-only loses the cuBLAS shim and the attn substrate
entirely (calls 0 vs 20). The InitializeInjection actuator-init chain is
fully CUDA_INJECTION64_PATH-native. Exactly two subsystems are
LD_PRELOAD-dependent — both rely on link-order symbol interposition, which a
late-dlopen'd injection library cannot provide. Dropping LD_PRELOAD requires
re-architecting those two via GOT/PLT patching. Detail: CP_2_5_DESIGN_MEMO.md §0.
