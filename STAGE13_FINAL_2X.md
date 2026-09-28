# Stage 13 — Final 2× tok/W stack measurement

Date: 2026-04-30
Pod: H100 80GB SXM, CUDA 12.8, libcublasLt 12.8.4, sm_90.

## Methodology

- **Baseline**: DEFAULT clock (`nvidia-smi -rgc`), no LD_PRELOAD, no
  CIPHER, eager mode. Plain PyTorch + Mistral-7B fp16. The H100 auto-
  clocks under load (typically 1980 MHz boost down to ~1350 MHz under
  the 700 W cap).
- **CIPHER**: Per-batch optimal clock locked via `sudo nvidia-smi -lgc`,
  `LD_PRELOAD=libcipher_hook.so:libcuda.so`, libcipher_rt.so loaded via
  ctypes. Every op enabled (CIPHER_SENSE / SHIELD / SUSTAIN / THERMOSTAT
  / PULSE / VOLT / HIBERNATE / LOOP / CONTINUITY / SUBSTITUTE_V2 /
  WEIGHT_COMPRESS / FUSION_KERNELS / FP8_COMPUTE / + the rest).

Per-batch optimal clock (from `p5_optimal_clocks.json`):

| Batch | locked MHz | rationale |
|-------|------------|-----------|
| 1     | 1000       | decode-bound, fewest flops/sec needed |
| 8     | 1200       | FP8 sweet spot (memory + compute balanced) |
| 32    | 1350       | sustained ceiling — bigger batches need flops |
| 64    | 1000       | KV-bandwidth dominates; clock saves power |

Run window: 8 s decode loop after a 10-step warmup, prefill = 128
tokens, batch unchanged across the window. Power sampled by
nvidia-smi at ~150 ms cadence. NaN sentinel checked before and after
the timed loop.

## Headline results

```
  B  baseline tps  baseline W  baseline tok/W   cipher tps  cipher W  cipher tok/W    ×tps  ×tok/W  fp8_calls
  1         48.83       203.7          0.2397        52.54     145.4        0.3613    1.08    1.51          0
  8        390.32       254.1          1.5363       401.31     154.8        2.5930    1.03    1.69      68850
 32       1307.23       390.3          3.3495      1186.19     250.6        4.7328    0.91    1.41      66825
 64       1753.72       444.2          3.9479      1122.01     199.0        5.6379    0.64    1.43      31725
```

**Tokens-per-watt: 1.41× – 1.69× across the four batch sizes.**

B=8 hits **1.69× tok/W**, the FP8 sweet spot — bandwidth-bound decode
where the 50 % weight-byte reduction dominates and the locked 1200 MHz
clock saves substantial power.

## Where each lever contributes

| B | clock | FP8 hits | watts saved | tps delta | net tok/W |
|---|-------|----------|-------------|-----------|-----------|
| 1 | 1980→1000 | 0 (gated, M=1)     | -29 % | +8 %  | **1.51×** |
| 8 | 1980→1200 | 68 850 (every linear) | -39 % | +3 %  | **1.69×** |
| 32| 1980→1350 | 66 825             | -36 % | -9 %  | **1.41×** |
| 64| 1980→1000 | 31 725             | -55 % | -36 % | **1.43×** |

At B=1 the win is clock locking only — the FP8 dispatch deliberately
skips M=1 so INT4 GEMV (graph-mode) can claim that slot.

At B=8 FP8 fires on every decode-step linear (~32 layers × 7 linears ×
~308 decode steps in the 8-second window = 68 992 expected; we observe
68 850, a near-perfect substitution rate). Bandwidth saving lets tps
*increase* slightly even with the lower clock.

At B≥32 the locked clock is below what the H100 would have run under
auto-DVFS, so tps drops; power drops more, so tok/W still wins.

## Op regression with FP8 alongside

`run_full_regression.sh` runs every `tests/test_*.py` with all 22 ops +
all 9 actuators + FP8 enabled simultaneously.

```
 REGRESSION SUMMARY: 31 passed, 4 failed, 8 skipped
```

The 8 skipped tests are heavy / model-loading benchmarks (test_int4_*,
test_megakernel, test_attention_koopman, test_pattern6_paraminfo,
test_kv_v3_rope, test_edmd_live_calibration). The 4 failures are
**pre-existing on this pod, not regressions**:

| failing test | reason | unrelated to FP8? |
|--------------|--------|-------------------|
| `test_packaging.py` | imports the missing `cipher_runtime` Python package | yes — packaging never built on this pod |
| `test_dep2.py` | same `cipher_runtime` import | yes |
| `test_hw_validation.py` | hardcoded path `/workspace/CIPHER_final_session7` from a previous pod | yes |
| `test_persist_dispatch.py` | persist-shim `cuLaunchKernelEx` overhead gate | yes — Stage 13 only modifies `cublasGemmEx`, not `cuLaunchKernelEx` |

The 31 passing tests include every op listed in the regression mandate
(SENSE, SHIELD, SUSTAIN, THERMOSTAT, PULSE, VOLT, HIBERNATE, LOOP,
CONTINUITY) plus PIPELINE / PREDICT / GUARD / DETERMINISM / TOPOLOGY /
TRACE / FAIRNESS / CARBON / RECEIPT / COMPLY / STRAGGLER / POWER_CAP /
PERSIST_ENGINE / FP8_CORRECTNESS / actuation_full / fusion_correctness
/ persist_engine / nccl_neural_loop.

No NaN, no Inf, no correctness regression in any timed run.

## Files

- `STAGE13_FP8_REPORT.md` — full Stage-13 implementation writeup
- `STAGE13_FINAL_2X.md` — this file
- `run_full_regression.sh` — env-loaded regression runner
- `run_final_2x_measurement.sh` — baseline-vs-CIPHER measurement runner
- `step7_fp8_eager.py` — Mistral-7B harness (modes: baseline/fp8/stack)
- `tests/test_fp8_correctness.py` — Step 6 16-shape correctness gate
- `include/cipher_fp8_compute.h`, `src/cipher_fp8_compute.cpp`
- 70-line FP8 dispatch block in `src/cipher_intercept_cudart.cpp`
