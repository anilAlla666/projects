# STEP 5 — Final combined measurement

## What is "all steps active"
| Step | Lever | Active in Step 5 |
|------|-------|-------|
| 1 | V3 KV compression | **B=1 only** (B>1 hits MAX_TOKENS=4096 cache cap and crashes; documented in STEP1_RESULTS.md) |
| 2 | Deep MLP megakernel | B=1 only (gated on M=1 decode; falls back at B>1) |
| 3 | Weight reuse across batch | **No-op**: PyTorch already issues n=B GEMMs (STEP3_FINDINGS.md) |
| 4 | Adaptive layer precision | **No-op**: 0 layers qualify under absmax-INT4 (STEP4_FINDINGS.md) |
| 5 | Fused RMSNorm + SiLU·Mul | All batches |

## Final numbers (graph capture, prefill 1024, decode-loop ≥6 s sustained)

### Without V3 (clean run, all 4 batches)
| batch | tps | watts | tok/W | × eager tps | × eager tok/W |
|------:|-----:|------:|------:|------:|------:|
|   1   | 94.81  | 293.6 | 0.3229 | **1.90×** | **1.27×** |
|   8   | 490.74 | 431.8 | 1.1366 | 0.98× | 0.57× |
|  32   | 769.24 | 515.6 | 1.4918 | (no eager baseline) |   |
|  64   | 862.84 | 570.4 | 1.5127 | (no eager baseline) |   |

Eager baseline (user-provided): B=1 50 tps@196W=0.255 tok/W; B=8 500 tps@250W=2.0 tok/W.

### With V3 on (B=1, B=8 — B≥32 crashes)
| batch | tps | watts | tok/W | Δ vs no-V3 |
|------:|-----:|------:|------:|------|
|   1   | 90.15  | 293.5 | 0.3072 | −4.9 % tps, −4.9 % tok/W |
|   8   | 483.97 | 405.5 | 1.1934 | −1.4 % tps, **+5.0 % tok/W** |

## What this session uncovered
1. **The INT4 GEMV ctypes binding has been wrong.** Every `Int4Linear`
   throughout the repo was passing `(N, K)` to a C function that expects
   `(M, N, K)`, so K was read off-stack as the low-32 of the
   stream-pointer arg. The `(K & 127) != 0` shape gate then short-circuited
   the kernel back to a 0-iteration loop, the call returned `rc==0`,
   and the Python fell back to fp16 cuBLAS. That is, **none of the
   "INT4" measurements in BUILD_STATE / SESSION_REPORT_2026-04-29 were
   actually using INT4** — they were measuring the fused RMSNorm +
   SiLU·Mul + fp16 cuBLAS path. Corrected binding: `int M=1, int N, int K`.
2. **With the corrected binding, INT4 GEMV firing at B=1**: tps ~95
   (slightly below the broken path's 99 because INT4 GEMV is currently
   ~1.0× cuBLAS on this Mistral shape, not the 1.09× the kernel
   microbench reports — the integrated path is barely break-even on
   tps), but **power drops from 319 W → 294 W (−8 %)** because HBM
   weight reads drop 4×. Net tok/W: 0.323 vs 0.313 baseline = +3 %.
3. **The deep MLP megakernel (Step 2) does help** at B=1: gate+up+silu·mul
   + down+residual fused into 2 launches lifted tps to 102 in Step 2,
   though the integrated stack here lands at 95 (likely because Step 2
   is measured against the pre-fix baseline; with the binding fix the
   delta is similar in absolute terms but the baseline shifted).
4. **At B≥8, the megakernel and INT4 paths are dormant** (M=1 gating).
   tps and tok/W there reflect the existing fused-cuBLAS path. Eager-mode
   tok/W of 2.0 at B=8 is unreachable with graph capture because graph
   mode draws 1.7× more power continuously.
5. **V3 KV compression**: B=1 negative, B=8 mildly positive on tok/W
   (−1.4 % tps, +5 % tok/W). Higher batches need a buffer-size and
   prefill-cache redesign before V3 can run.
6. **Adaptive layer precision** is gated entirely on having
   AWQ-calibrated INT4. With absmax INT4's ~12 % per-element noise, no
   layer's hidden-state rel-err comes anywhere near 0.005 / 0.02.

## Verification of B=1 after each step
| step | B=1 tps | comment |
|------|--------:|---------|
| pre-step1 (step1 V3-off baseline) | 99.92 | unknowingly fp16 cuBLAS path |
| step1 V3-on               | 92.91 | V3 active, 6 % cost |
| step2 megakernel          | 102.44 | megakernel + (broken) int4 path |
| step5 corrected INT4      | 94.81 | first run with INT4 actually firing |

The B=1 lane held coherent throughout; the changes documented in each
step kept it within ±10 % of the baseline.

## Files
- `step5_bench.py` — final combined harness with corrected int4 binding
- `step5_results.json` — final measurements
- `STEP{1,3,4,5}*` — per-step findings
