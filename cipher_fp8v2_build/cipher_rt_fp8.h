/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_fp8.h -- D.9 FP8 actuator (per-tensor scalar fused FP8 E4M3) on the
 * Phase 4.5 matmul-dispatch substrate. Net-new, default-OFF (CIPHER_FP8=on).
 *
 * Sits alongside Marlin as a SECOND actuator (priority 20). Marlin gate is M<=64
 * (decode); the FP8 gate is the COMPLEMENT, M>64 (large-batch / prefill /
 * training-forward) — the regime the D.9 probe proved fires at the public
 * cublasGemmEx intercept (3375 calls, Mistral B=64). Marlin + cuBLAS shim are
 * UNTOUCHED (Mem #13 additive).
 *
 * Scheme: per-tensor scalar FP8 — the ONLY fused path on this cu13 cuBLASLt
 * (rowwise/OUTER_VEC NOT_SUPPORTED, measured). Static per-tensor weight prequant
 * (one-time, cached) + inline per-tensor activation quant + cuBLASLt FP8 scalar
 * GEMM (FAST_ACCUM) producing fp16/bf16 DIRECTLY (no epilogue). fp16/bf16
 * in/out preserved to the app. Worst case = cuBLAS passthrough (Stage 0 sacred).
 */
#ifndef CIPHER_RT_FP8_H
#define CIPHER_RT_FP8_H

#ifdef __cplusplus
extern "C" {
#endif

/* Actuator init — env-gated (CIPHER_FP8). Called from cipher_inject.c after
 * cipher_rt_marlin_init. Registers the FP8 actuator (priority 20) on the
 * matmul-dispatch substrate when enabled. Returns 0 (never blocks CUDA init). */
int cipher_rt_fp8_init(void);

/* --- C ABI bridges to the C++/CUDA engine (cipher_rt_fp8_engine.cpp). --- */
int  cipher_rt_fp8_engine_init(void);
/* observe a weight pointer (K=in_features, N=out_features); returns hit count.
 * v2: cache keyed (ptr,K,N) — orientation swap = distinct slot, no reset. */
int  cipher_rt_fp8_engine_observe_weight(const void *w_ptr, int K, int N);
int  cipher_rt_fp8_engine_is_ready(const void *w_ptr, int K, int N);
/* one-time per-tensor weight quant -> E4M3 + scalar scale, cached by w_ptr.
 * dtype: 2 = CUDA_R_16F, 14 = CUDA_R_16BF. Returns 0 on success. */
int  cipher_rt_fp8_engine_quantize_weight(const void *w_fp, int dtype,
                                          int K, int N, void *stream);
/* the substitution: per-tensor quant the activation inline + scalar FP8 GEMM
 * -> c_out (same dtype as input). m=out_features, n=batch, k=in_features.
 * weight key = w_ptr. Returns 0 on success, non-zero -> caller passthroughs. */
int  cipher_rt_fp8_engine_matmul(const void *w_key, const void *activation,
                                 void *c_out, int dtype,
                                 int m, int n, int k, void *stream);
unsigned long cipher_rt_fp8_engine_weights_count(void);

/* Accessors (tests / cipher-platform / counter dump). */
int           cipher_rt_fp8_is_active(void);
unsigned long cipher_rt_fp8_calls_total(void);     /* eligible by env+dtype+gate */
unsigned long cipher_rt_fp8_calls_handled(void);   /* dispatched to FP8 */
unsigned long cipher_rt_fp8_calls_skipped(void);
/* v2: non-(T,N)-layout calls refused to fp16 passthrough (subset of skipped). */
unsigned long cipher_rt_fp8_calls_layout_refused(void);
unsigned long cipher_rt_fp8_weights_quantized(void);
/* Guard 3: largest engaged N (cublas n = batch) and per-class engaged shape log. */
unsigned int  cipher_rt_fp8_max_n(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_FP8_H */
