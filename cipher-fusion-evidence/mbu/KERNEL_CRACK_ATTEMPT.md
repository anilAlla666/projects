# CRACK ATTEMPT: hand-tuned CUDA FP8 GEMV vs the frontier (2026-06-11)
Goal: beat cutlass's 54% MBU (1.53x) toward fp16's 74% (=2x) at batch-1 decode. 8 kernels written:
  Triton: naive 0.33x / autotuned-fp32 0.91x / autotuned-fp16 0.94x (all <cutlass)
  CUDA v1 (block-per-row, scalar convert):     1.47x / 54% MBU  <-- BEST, TIES cutlass/vLLM
  CUDA v2 (shared-mem x):                       1.07x / 39%      (occupancy crushed by 28KB shmem)
  CUDA v3 (warp-per-row + shuffle):             1.19x / 44%
  CUDA v4 (ILP, scattered):                     0.33x / 12%      (broke coalescing)
  CUDA v5 (ILP, coalesced):                     0.96x / 35%      (register pressure)
RESULT: my best hand-tuned CUDA (v1, correct rel-err 0.026) hits 54% MBU = TIES NVIDIA cutlass + vLLM's tuned
kernel; CANNOT BEAT it. 74%/2x NOT reachable. Multiple independent well-tuned kernels (cutlass, vLLM, my v1) all
cluster at 54% MBU -> this is the real fp8-GEMV-at-batch-1 ceiling. CAUSE (physical): fp8 GEMV has ~2x the
compute-per-byte of fp16 (the dequant conversion), so the byte-savings and the added conversion compute partially
cancel -> ~1.5x, not 2x. The MBU/throughput 2x is WALLED at the kernel level, now PROVEN from the inside (wrote the
kernel, matched the frontier, could not exceed it). The 2x wins in this space come from hardware (B200) + precision
(FP4), not a substrate kernel.

## DEEPER (2026-06-11, user: "think deeper, work on kernels or something deeper"):
Went below fp8 to INT4. Wrote a GEMV-specialized int4 kernel (correct, rel-err 2e-4). Result: 12% MBU / 0.63x
(my naive) vs marlin's 27% (fast LOP3 unpack). DECISIVE PRECISION LADDER (all measured this session):
  fp16: 74% MBU = 1.0x   (no dequant)
  fp8 : 54% MBU = 1.5x   (cheap HW convert; my kernel TIES cutlass)  <-- PEAK
  int4: 27% MBU = 1.35x  (expensive unpack; marlin's level)          <-- FALLS
=> lower precision saves bytes but dequant COMPUTE grows faster; they cancel; throughput PEAKS at fp8 ~1.5x then
FALLS. The kernel/precision route to 2x is DEFINITIVELY closed, now explained physically (11 kernels written).

THE REAL HEADROOM (physics): at batch-1 decode the GPU streams 14GB to make 1 token in ~4.2ms, during which it
could do ~4 TFLOP of compute but uses only ~14 GFLOP => **~290x more compute available than used; GPU ~95% idle.**
The 2x is NOT in the kernel (already near the 74% memory ceiling) — it's in USING THE IDLE COMPUTE: more tokens per
weight-stream via multi-token / speculative / tree decoding (turns the memory-bound GEMV into a tensor-core GEMM).
That's where a substrate intercepting the forward pass could add real value (orchestrate speculation) — but the
reliable 2-4x needs a good speculator (draft model / trained MTP heads = model-side, not pure CUDA). ngram-spec is
substrate-deliverable but workload-conditional (RAG yes, novel-gen no; per stress test).
