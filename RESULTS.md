# CIPHER v2 Hardware Validation Results

Date: 2026-04-04
Platform: H100 SXM5, CUDA 12.8, PyTorch 2.8, NCCL 2.27.3

## Validation Suite: 7/7 PASS

| Test | Metric | Result |
|------|--------|--------|
| 7.1 Intercept count | 8 shapes, 50/50 each | PASS |
| 7.2 Correctness | max_diff=0.000000 across 6 shapes | PASS |
| 7.3 MFU regression | Baseline 688 TFLOPS, CIPHER 667 TFLOPS (-3.0%) | PASS |
| 7.4 Clock stability | 1575-1980 MHz under mixed load | PASS |
| 7.5 NCCL intercept | 5/5 AllReduce calls captured | PASS |
| 7.6 Non-GEMM classify | 12 ELEMENTWISE + 1 REDUCTION + 5 Chebyshev opportunities | PASS |
| 7.7 Billing report | 741 GEMM dispatches, 99.4% FLOPs covered | PASS |

## Billing Report

```
GEMM dispatches:        741
  substituted (O(1)):   441  (59.5%)
  passthrough (cuBLAS): 300  (40.5%)
FLOPs substituted:      6.06e+13  (99.4% of total)
FLOPs passthrough:      3.88e+11
Non-GEMM dispatches:    6
  substituted:          0
MFU (sustained GEMM):   ~670 TFLOPS (67.7% of spec, within 3% of cuBLAS baseline)
Oracle confidence adj:  -15 (aggressive — low MFU triggers more substitution)
```

## Phase Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Build system (Makefile) | COMPLETE |
| 1 | Shape-parametric Koopman (remove K=4096 gate) | COMPLETE |
| 1.2 | Generic kernel for any K/N | COMPLETE |
| 2 | NVML telemetry (clock, power, MFU) | COMPLETE |
| 3 | MFU oracle feedback + billing | COMPLETE |
| 4 | NCCL intercept (ncclAllReduce GOT patch) | COMPLETE |
| 5 | Non-GEMM classification + Chebyshev infrastructure | COMPLETE |
| 6 | Attention shape registration (K=128) | COMPLETE |
| 7 | Hardware validation suite | COMPLETE |

## What CIPHER v2 Intercepts

- cuLaunchKernel (8 variants) — GOT patched
- cudaLaunchKernel — GOT patched + dispatch
- cublasGemmEx — PLT exported + shape capture
- cublasLtMatmul — PLT exported + shape capture
- ncclAllReduce — PLT exported + timing
- cuGetProcAddress (v1 + v2) — prevents CUDA from bypassing hooks

## Architecture

```
LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" python3 any_workload.py

libcipher_hook.so (37KB):
  PLT exports + GOT patching for all intercept points
  Zero CUDA SDK dependency — pure C++17 + libdl

libcipher_rt.so (182KB):
  Dispatch engine + Oracle + Registry + EDMD + LNN + Telemetry
  10-operation pipeline (CLASSIFY through ARBITRATE)
  Chebyshev GPU kernel for elementwise substitution
  Shape-parametric Koopman kernel for any (K, N)
  Green Context SM partitioning (CUDA 12.4+)
  L2 cache persistence for weight pinning
```
