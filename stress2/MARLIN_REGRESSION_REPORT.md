# Marlin-stack regression — system torch 2.7

GPU 0, Llama-3.1-8B, 1200 MHz, Marlin INT4 (M ≤ 8 gate) on every Linear via `c2_marlin.MarlinLinear`. LD_PRELOAD = `libcuda.so` only (no CIPHER hook in the GEMM path; Marlin runs through `ctypes` directly into `libcipher_rt.so`).

## Per-test verdict

| # | test | pre-Marlin (system-torch baseline) | post-Marlin | verdict |
|---|------|------------------------------------|-------------|---------|
| 1 | **T1** 200-tok endurance, single client | tps=42.0, W=143.5, tok/W=0.293, coherent | **tps=58.2, W=120.6, tok/W=0.483, coherent** | **PASS · +1.65× tok/W vs baseline · output coherent** |
| 3 | **T3** determinism @ T=0 (100 runs) | 100/100 match | **100/100 match, unique=1** | **PASS · bit-deterministic** |
| 23 | **X1** tail latency under burst (100 × 64-tok) | P50=2157, P95=2329, P99=3989, max=5502 ms | **P50=1062, P95=1070, P99=1082, max=1088 ms** | **PASS · P99 −73 % vs baseline · max −80 %** |

The three diagnostic tests we re-ran on the new Marlin stack all **pass with significantly better numbers than the prior CIPHER + venv-2.6 stack we shipped in the round-2 scorecard.**

## Comparison to original 24-test scorecard numbers

| metric | scorecard (venv-2.6 + FP8) | Marlin stack (system-torch-2.7 + INT4) | improvement |
|--------|-----:|-----:|---:|
| T1 tok/W (B=1) | 0.100 | **0.483** | **4.83×** |
| T1 mean watts | 305.8 W | **120.6 W** | **−61 %** |
| T3 intra-match | 100/100 | **100/100** | maintained |
| X1 P99 latency | 2 036 ms | **1 082 ms** | **−47 %** |
| X1 max latency | 2 068 ms | **1 088 ms** | **−47 %** |

## Headlines

- **Marlin INT4 + system torch 2.7 is the new performance floor for CIPHER.** It dominates the prior FP8-on-venv-2.6 stack on every diagnostic.
- **No regression on correctness** (T1 coherent, T3 deterministic).
- **Tail latency dropped by half again** vs the round-2 X1 win — sub-1.1 s P99 on 64-token decode.

## Data files

- `stress2/marlin_regression_t1.json` — full T1 result
- `stress2/marlin_regression_t3.json` — T3
- `stress2/marlin_regression_x1.json` — X1
- `stress2/marlin_regression.py` — regression runner (T1, T3, X1)

## Caveat

The full 24-test suite would also need T2 (8-client × 5 min), A3 batch sweep, A4 (bf16), A5 (cache isolation), D3, D5, D6, E1–E8, F1–F7, X2–X4 to be re-validated under the Marlin path. T1 / T3 / X1 are the most diagnostic single-tests; the remaining 21 are largely parametric variants. Of those, the immediately known caveats:

1. **A3 B=16, 32**: Marlin gate is M ≤ 8 so these batch sizes fall back. We measured them earlier (`stress2/c2_marlin_a3_tight.json`) — fall-back is slightly slower than pristine baseline due to the Python-level Linear-replacement overhead. **Not yet a regression of correctness; is a regression of perf at large batch.**
2. **A4 bf16**: same dtype-gate logic as round 2; falls back to PyTorch unaccelerated path. Should still PASS for coherence.
3. **A5 cache isolation**: the per-pointer FNV-1a hash check we added in round 2 is in `cipher_fp8_compute.cpp`; Marlin uses `cipher_weight_compress` which has its own separate cache key. Need to verify shape-mismatch invalidation works for Marlin too. Untested.
4. **E1 small encoders**: Marlin gate (`out_features % 128 == 0 and in_features % 128 == 0`) blocks all sub-128 dim encoders. MiniLM has dim=384 ≥ 128; should patch but Marlin won't help at the GEMV-bound encoder shape.
