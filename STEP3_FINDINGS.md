# STEP 3 — Weight reuse across batch (M dim probe)

## Method
Logged the first 800 `cublasGemmEx` (m, n, k) tuples during a B=8 decode of
Mistral-7B with `CIPHER_KERNEL_LOG=on`. The shim records m, n, k from every
fp16 GEMM. Tally:

| count | m       | n     | k       | role                              |
|------:|--------:|------:|--------:|-----------------------------------|
|   164 |   4096  |   **8** |  4096  | q_proj, o_proj (decode, B=8)      |
|   164 |  14336  |   **8** |  4096  | gate_proj, up_proj (decode, B=8)  |
|   164 |   1024  |   **8** |  4096  | k_proj, v_proj  (decode, B=8)     |
|    81 |   4096  |   **8** | 14336  | down_proj (decode, B=8)           |
|    64 |   4096  |  1024 |  4096  | q_proj/o_proj  prefill (n=B×P=8×128) |
|    64 |  14336  |  1024 |  4096  | gate/up prefill                   |
|    64 |   1024  |  1024 |  4096  | k/v prefill                       |
|    32 |   4096  |  1024 | 14336  | down prefill                      |
|     2 |  32000  |     8 |  4096  | lm_head (decode)                  |
|     1 |  32000  |  1024 |  4096  | lm_head (prefill)                 |

## Finding
**At decode batch=8, every cuBLAS GEMM has n=8 — PyTorch already batches
all 8 tokens into a single GEMM call.** None of the calls have n=1, i.e.
PyTorch is NOT issuing 8 separate M=1 GEMVs. Batching across the batch
dimension is **already free / captured by PyTorch**.

## Implication for Step 3 ("buffer inputs and batch")
**No lever to pull.** The fix described in the spec ("if M=1 called 8
times, buffer inputs and batch") would only apply if PyTorch issued M=1
GEMVs in a loop. It doesn't. The 8-row GEMM is single-shot.

## Adjacent finding
The same data exposes the actual decode-throughput bottleneck at B>1:
- Our `Int4Linear.forward` gates the INT4 GEMV path on
  `x.shape[-2] == 1 and x.shape[0] == 1`. At B>1 it falls through to
  fp16 cuBLAS (`x @ self.wt_fp16`).
- This is correct because the existing `cipher_int4_gemv` is M=1 only;
  per BUILD_STATE its multi-row variant runs at 0.19× cuBLAS at M=8 and
  was deliberately not wired.
- Net effect: at B=8/32/64, INT4 weight compression provides **0%
  bandwidth savings** during decode — every weight is read in fp16.
  The batch=8 row in BUILD_STATE shows tok/W=2.0 reflects this.

The right lever is M=8/16/32 INT4 GEMM (Marlin-style, mentioned but
disabled because Marlin's M_blocks=4 path was empirically slower than
cuBLAS at those sizes). Step 3 is "free" only in the trivial sense that
PyTorch's batching already captures the launch-overhead saving; the
**weight-bandwidth saving** that INT4 promises at B=1 simply does NOT
happen at B=8 today.

## Files
- `step3_logM.py` — captures GEMM-EX log
- `/tmp/step3_gemm_log_full.txt` — full 800-call trace
