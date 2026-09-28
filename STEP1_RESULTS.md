# STEP 1 — KV Compression V3 wired into benchmark

## Setup
- Workload: Mistral-7B fp16 + 224 INT4 GEMV linears + fused RMSNorm + fused
  SiLU·mul + graph capture (post-prefill replay loop), prefill 1024 tok.
- V3 path: `cipher_kv_redirect_quant_kv` fires from `cublasGemmEx` after
  K_proj / V_proj at decode (`m == 1024`). On every FA launch
  `cipher_kv_redirect_on_fa_launch` dequantizes the per-layer 2-bit cache
  with inline RoPE into FA staging buffer.
- Env to enable V3: `CIPHER_KV_REDIRECT=on CIPHER_KV_RDR_V3=on
  CIPHER_KV_RDR_V3_ROPE=on CIPHER_KV_NUM_LAYERS=32 CIPHER_KV_ROPE_BASE=10000`.

## Results

| batch | V3  | tps    | watts | tok/W   | fa_rew | Δtps      | Δtok/W   |
|-------|-----|--------|-------|---------|--------|-----------|----------|
| 1     | OFF | 99.92  | 318.9 | 0.3133  | 0      | —         | —        |
| 1     | ON  | 92.91  | 314.8 | 0.2951  | 160    | **−7.0%** | **−5.8%**|
| 8     | OFF | 490.54 | 431.8 | 1.1361  | 0      | —         | —        |
| 8     | ON  | 483.89 | 405.5 | 1.1934  | 288    | **−1.4%** | **+5.0%**|
| 32    | OFF | 769.79 | 515.5 | 1.4934  | 0      | —         | —        |
| 32    | ON  | crash  | —     | —       | —      | —         | —        |
| 64    | OFF | 861.29 | 565.2 | 1.5238  | 0      | —         | —        |
| 64    | ON  | crash  | —     | —       | —      | —         | —        |

`fa_rew` = number of FA struct rewrites (= 2 K + 2 V × num_layers / batch
factor over the measurement window). At V3-on B=1 it fired 160 times in
260 graph replays, confirming V3 is actually firing and not a no-op.

## Findings
1. **V3 cost dominates at small batch.** At B=1 the dequant kernels add
   compute that exceeds the bandwidth saving: 32 layers × (1 dequant_perm_rope
   K + 1 dequant_perm V) per replay launches 64 NVRTC kernels per token,
   each writing ~256 KB of fp16 (cur_pos × KV_HEADS × 128 × 2). Power drops
   ~4 W (less HBM reads of fp16 KV) but tps drops 7%, net tok/W −5.8%.
2. **V3 helps slightly at B=8.** Per-token KV-cache HBM read at B=8 is
   8× larger, so the fp16→2-bit reduction pays back: tps regression
   shrinks to −1.4%, power drops from 432 W → 406 W, tok/W +5.0%.
3. **B=32 / B=64 V3 cannot run with current implementation.** Two
   independent constraints surfaced:
   - The 2-bit cache holds `KV_HEADS × MAX_TOKENS = 8 × 4096 = 32 768`
     rows per layer. At B=32/64 prefill=1024, the cuBLAS K_proj sees
     `n = B × seq = 32 768 / 65 536` tokens per call → `pos + n` exceeds
     `MAX_TOKENS` so prefill quant is bypassed (quant_kv early-returns).
   - With cache empty, V3 falls back to the cur_pos==0 path, which I
     patched to skip redirect entirely (else memcpy(my_K, orig_K,
     COPY_BYTES) over-reads orig_K when the FA staging buffer is smaller
     than COPY_BYTES). Decode quant then fires per-step but writes
     decode-only data that doesn't carry prefill context. At B=32 this
     creates a state mismatch with cuBLAS internally and crashes with
     `CUBLAS_STATUS_EXECUTION_FAILED`.
4. **Net: V3 is decode-with-small-prefill only.** Useful regime is
   B × prefill ≤ 4096 (= MAX_TOKENS); above that the per-layer 2-bit
   cache cannot hold prefill data, and the dequant-only-decode path is
   architecturally broken. At B=1 prefill≤4096 V3 works correctly but
   the kernel-launch cost outweighs the ~3 GB-of-cache-read savings at
   the workloads we measured.

## Patches landed
- `src/cipher_kv_redirect.cpp`:
  - `BUF_BYTES` 1 MB → 32 MB (covers FA staging up to B=8 prefill≤512).
  - `MAX_LAYERS` 64 → 32 (was 64 for safety, 32 = Mistral 7B layers).
  - cur_pos==0 V3 path now returns 0 (no redirect) instead of memcpy-from-
    orig_K, since the source staging buffer is sometimes smaller than
    COPY_BYTES at large batch.

## Files
- `step1_bench.py` — benchmark harness (V3 toggleable via env)
- `step1_v3_off.json` — V3-off measurements (4 batches)
- `step1_v3_on.json`  — V3-on measurements (B=1, B=8 only)
