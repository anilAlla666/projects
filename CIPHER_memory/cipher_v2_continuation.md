---
name: CIPHER v2 next session pickup
description: Exactly where to resume CIPHER v2 work, current state of code, pending decisions, what to do first next session
type: project
---

# CIPHER v2 — Session Pickup Instructions

**Read this first when resuming CIPHER v2 work.**
Full history: see `project_cipher_v2_session.md`.
MFU baseline rule: see `feedback_mfu_baseline.md`.

## Where I Left Off

The session ended mid-discussion about speedup benchmarks for investors. The user (at Neural Dynamics, Inc.) wants to send benchmarks to Devang at Nebius, but I refused to fabricate fake speedup numbers. The user asked for "Option C" (large-batch Koopman benchmark) to see if Koopman wins at M=1024+. It does NOT win at any batch size with the current kernel.

**Final benchmark (same shape 4096x4096 fp16, separate processes):**
```
     M        cuBLAS        CIPHER       Ratio
     1       9.98 us      30.13 us     0.33x
    32       9.10 us      31.86 us     0.29x
    64       9.68 us      32.14 us     0.30x
   128      15.46 us      32.68 us     0.47x
   256      20.78 us      45.44 us     0.46x
   512      25.91 us      84.55 us     0.31x
  1024      47.99 us     180.84 us     0.27x
  2048      97.23 us     328.20 us     0.30x
  4096     191.76 us     642.37 us     0.30x
```

**Conclusion:** Koopman kernel cannot beat cuBLAS with current design. User was told this honestly. No decision yet on which speedup path to pursue next.

## Current State of Code

**Working directory:** `/workspace/CIPHER_final_session7/`

**Build state:** Clean. `make clean && make all` succeeds. Both DSOs present and working:
- `libcipher_hook.so` (37KB)
- `libcipher_rt.so` (~182KB)

**Active features:**
- All 8 phases complete and verified (7/7 hardware validation passing)
- wmma kernel built but disabled by default (behind `CIPHER_USE_WMMA=1` env var) — it's 4x SLOWER than scalar
- L2 persistence active when shapes registered (verified pinning works, but doesn't help performance)
- Scalar Koopman kernel is default (30μs/call floor)
- Makefile has `-arch=sm_90` for wmma support

**Pending code issues (none blocking):**
- `include/cipher_koopman_runtime.h` may or may not have shape M/K/N fields (was in original Phase 1.2 plan)
- Chebyshev opportunity logging works but substitution does not fire (cuLaunchKernel params layout is opaque — known limitation)
- CUPTI unavailable on pod, so L2 hit rate and HBM BW are estimated not measured
- Billing "substituted" count includes registry hits even when cuBLAS still runs via relaunch — billing accuracy issue

## All 9 Checkpoints Saved

```
/workspace/CIPHER_v2_phase0_complete       — Makefile + path fix
/workspace/CIPHER_v2_phase1_complete       — K=4096 gate removed, 8 shapes intercept
/workspace/CIPHER_v2_phase1_2_complete     — generic Koopman kernel (any K/N)
/workspace/CIPHER_v2_phase2_complete       — NVML telemetry + windowed MFU
/workspace/CIPHER_v2_phase3_complete       — MFU oracle feedback + billing
/workspace/CIPHER_v2_phase4_complete       — NCCL intercept (ncclAllReduce GOT patch)
/workspace/CIPHER_v2_phase6_complete       — attention shapes (K=128)
/workspace/CIPHER_v2_final                 — substitution proof complete (20/20)
/workspace/CIPHER_final_session7           — LIVE (contains L2 pin + wmma experiments, Phase 7 validation suite)
```

## The Pending Decision

User asked "how do we make Koopman faster than cuBLAS" for investors. I offered three paths and explained the reality:

**Path 1: Cached output memcpy (1-2 days)**
- Precompute static outputs, substitute via cudaMemcpy
- Real 10-20x for static inputs (bias, decode KV lookups)
- Fastest path to a real number
- Not started

**Path 2: Multi-warp tensor-core Koopman kernel (5-7 days)**
- Persistent threads, auto-tuned tile sizes, multi-warp cooperation
- Would get 400+ TFLOPS, beats cuBLAS at M>256
- Current wmma kernel is 1-warp-per-block which is wrong (4x slower than scalar)
- Would need proper multi-warp cooperative design

**Path 3: Fused operator kernels (3-5 days)**
- GEMM+bias+GeLU in one kernel
- Saves 20-30μs per layer via eliminated launches
- L2 fusion infrastructure exists but never fires

**User has not chosen. Session ended when they asked me to write this memory file.**

## What To Do First Next Session

1. **Verify current state**: `cd /workspace/CIPHER_final_session7 && make clean && make all` — should succeed
2. **Re-run validation**: `python3 tests/test_hw_validation.py` — should be 7/7
3. **Re-run demo**: `LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" CIPHER_FORCE_PERMIT=1 python3 cipher_demo.py 2>/tmp/cipher_demo.log` — should show all 12 ops
4. **Ask the user**: which speedup path to pursue (cached output / tensor-core kernel / fused ops). Do NOT start building without their direction.

## What NOT To Do

- **Do NOT fabricate speedup numbers.** The user pushed hard and I refused. Investors would run it themselves and catch the fraud. Architecture story is real and defensible; fake speedups are not.
- **Do NOT chase 850 or 989 TFLOPS.** H100 sustained ceiling is ~700 TFLOPS. The 989 number is boost spec marketing. User corrected me on this earlier. See `feedback_mfu_baseline.md`.
- **Do NOT introduce model-specific logic.** DRIFT RULE: CIPHER sees only geometry (M, N, K, function pointer, grid dims). No layer indices, no QKV vs O projection knowledge, no kernel name matching, no Mistral/Llama hooks. Old code had drift (layer_idx = s_gemm_count % 64 / 2 with hardcoded layers 2,9,10,11,15) — this was removed in Phase 1. Don't let it come back.
- **Do NOT try to substitute via cuLaunchKernel params.** The params layout is kernel-specific ABI. Attempted in Phase 5 and crashed the process. Chebyshev substitution must happen at Python wrapper level where tensor shapes are known.
- **Do NOT re-enable wmma kernel** until it's rewritten with multi-warp cooperation. Current design is 4x slower than scalar.

## What IS Genuinely Sendable to Investors (from Phase 7 validation)

- 5,148/5,148 kernels intercepted — zero escapes
- max_diff=0.000000 across 8 production shapes (Mistral/Llama)
- 2.1% overhead on passthrough (competitive with profilers)
- 12 operations firing concurrently on real H100 hardware
- 100% substitution on registered shapes (20/20, cuBLAS fully suppressed)
- 7/7 hardware validation passing
- NVML clocks + power + utilization at 500Hz
- Real windowed MFU measurement that survives async cuBLAS pipelining
- HMAC-SHA256 chained audit trail
- Full intercept coverage: cuBLAS, cuBLASLt, NCCL, cuLaunchKernel (8 variants), cudaLaunchKernel

## Product Context

- **Company:** Neural Dynamics, Inc. (neuraldynamicsai.com)
- **Product name:** X1 (formerly CIPHER) — user renamed in demo
- **Target customer:** Devang at Nebius (cloud GPU platform)
- **Positioning:** "Neural Execution Primitive" — zero-app-change GPU compute interception layer
- **Value prop:** Not a faster GEMM library. A PRIMITIVE that sits between PyTorch and CUDA with rich telemetry, billing, and substitution infrastructure.

## Critical Files to Reference

```
/workspace/CIPHER_final_session7/RESULTS.md                      — hardware validation summary
/workspace/CIPHER_final_session7/docs/superpowers/plans/2026-04-04-cipher-v2-build-plan.md
                                                                  — full plan with all 8 phases
/workspace/CIPHER_final_session7/cipher_demo.py                   — 12-op live demo script
/workspace/CIPHER_final_session7/cipher_startup.py                — synthetic matrix registration
/workspace/CIPHER_final_session7/tests/test_hw_validation.py      — 7-test validation suite
/workspace/CIPHER_final_session7/Makefile                         — build system (has -arch=sm_90)
```

## Key Technical Facts To Remember

- cuBLAS has ~9μs launch floor on H100 for any shape — cannot be beaten by naive kernels
- Koopman kernel achieves ~53 TFLOPS vs cuBLAS ~178 TFLOPS at M=4096 (3x gap)
- wmma with 1 warp per block is wrong — tile setup overhead (~30 cycles/load_matrix_sync) dominates
- L2 pinning doesn't help matrices <1MB (they fit naturally in 50MB L2 after warmup)
- CUPTI PM counters not available on this pod (no headers, no libcupti.so)
- NCCL 2.27.3 installed at /usr/lib/x86_64-linux-gnu/libnccl.so.2
- GPU: H100 SXM5 80GB, sm_90, CUDA 12.8, PyTorch 2.8.0+cu128
- Thermal throttling drops SM clock to 1395 MHz under sustained load (from 1980 boost)
- LD_PRELOAD overhead of 2.1% is inherent — cannot be reduced without breaking architecture
- `get_shim()` in intercept layer has prefix filter: accepts only symbols starting with 'c' or 'n' (for nccl). Any new symbol types need this updated.
- `tls_shape_valid` flag indicates "cublasGemmEx shim already processed this GEMM" — cuLaunchKernel and cudaLaunchKernel shims skip dispatch when set to avoid double-work
- `CIPHER_FORCE_PERMIT=1` bypasses oracle warmup denial (needed for demos where topological phase detection hasn't kicked in yet)

## Environment Variables Used

- `LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so"` — activate CIPHER
- `CIPHER_SAFE_MODE=1` — hook-only mode, no full runtime init (used for regression tests)
- `CIPHER_SAFE_MODE=0` — full runtime (used for substitution tests)
- `CIPHER_FORCE_PERMIT=1` — bypass oracle gates (used for demos)
- `CIPHER_USE_WMMA=1` — enable wmma kernel (currently slower than scalar, leave off)

## Build Quick Reference

```bash
cd /workspace/CIPHER_final_session7
make clean && make all    # Full rebuild, ~30 seconds
make hook                 # Just libcipher_hook.so (fast)
make rt                   # Just libcipher_rt.so (slower due to .cu files)
nm -D libcipher_hook.so | grep <symbol>
nm -D libcipher_rt.so | grep <symbol>
```

## Quick Regression Test

```bash
LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" CIPHER_SAFE_MODE=1 python3 -c "
import torch, ctypes
hook = ctypes.CDLL('./libcipher_hook.so')
hook.cipher_intercept_count.restype = ctypes.c_uint64
w = torch.randn(64, 64, dtype=torch.float16, device='cuda')
torch.mm(w, w); torch.cuda.synchronize()
for M, K, N in [(1,4096,4096),(1,4096,14336),(32,4096,4096),(1,8192,28672)]:
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    before = hook.cipher_intercept_count()
    for _ in range(50): torch.mm(a, b)
    torch.cuda.synchronize()
    print(f'({M},{K},{N}): {hook.cipher_intercept_count()-before}/50')
"
```

Expected: each shape shows 50/50 or 100/50 (cuBLAS splits large K internally).

## First Question To Ask The User

"Last session we ended with three paths to real Koopman speedup: cached output memcpy (1-2 days, 10-20x), multi-warp tensor-core kernel (5-7 days, 2-5x at large batches), or fused operator kernels (3-5 days, saves 20-30μs/layer). You hadn't chosen. Which path do you want to start with, or has the priority changed?"
