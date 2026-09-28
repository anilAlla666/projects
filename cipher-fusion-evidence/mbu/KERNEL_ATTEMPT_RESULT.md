# FP8 GEMV kernel attempt (MBU-itself 2x) — UNSUCCESSFUL, 2026-06-11
Goal: a GEMV-specialized FP8 kernel @ ~74% MBU = 2x. Three autotuned Triton iterations on real Mistral shapes:
- naive fp32-convert: 0.33x fp16 (12% MBU)
- autotuned fp32-convert: 0.91x (33% MBU on fp8 bytes)
- autotuned native fp8->fp16 convert: 0.94x (35% MBU)
All correct (rel_err 0.026). NONE beats cutlass's 1.53x (vLLM FP8). Root cause: Triton can't vectorize fp8 1-byte
loads / hardware-convert enough to become memory-bound at batch-1 GEMV — it stays conversion/transaction-bound,
same wall-time as fp16 despite half the bytes. cuBLAS fp16 = 74% MBU (bar); cutlass FP8 = 54% (frontier, 1.53x);
my best = 35%. Beating cutlass to 74% needs hand-tuned CUDA w/ fp8 vector intrinsics (__nv_cvt) = uncertain R&D
against NVIDIA's frontier. CONCLUSION: MBU-itself 2x not reached; fall back to FP8 x spec-decode for decode-2x.
