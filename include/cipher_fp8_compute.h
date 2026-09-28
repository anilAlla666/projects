// CIPHER FP8 Compute Substitution — Stage 13
//
// Substitutes cublasGemmEx fp16 calls with cublasLtMatmul FP8 (E4M3) calls.
// One-time per weight: NVRTC kernel quantizes fp16 → FP8 + per-tensor scale.
// Per call: NVRTC kernel quantizes activation fp16 → FP8 (~2us at decode M),
// then cublasLtMatmul runs FP8 × FP8 → fp16 with implicit (A_scale × B_scale)
// rescale via the matmul descriptor's A/B scale pointers.
//
// Default OFF. Env: CIPHER_FP8_COMPUTE=on.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherFp8ComputeStats {
    int      enabled;
    uint64_t weights_quantized;
    uint64_t matmul_calls;
    uint64_t passthroughs;
    uint64_t correctness_failures;
    size_t   bytes_fp16_weights;
    size_t   bytes_fp8_weights;
} CipherFp8ComputeStats;

int  cipher_fp8_compute_init(void);
int  cipher_fp8_compute_enabled(void);

// Stable-weight detection.  Increment a hit counter for (ptr, rows, cols);
// returns the new count.  When the count crosses CIPHER_FP8_STABILITY
// (default 2), the cublasGemmEx shim should call _quantize_weight.
int  cipher_fp8_compute_observe(void* weight_fp16, int rows, int cols);

// One-time per weight: quantize fp16 weight (rows × cols, row-major) to
// FP8 E4M3 with per-tensor absmax scale.  Both buffers are allocated by
// the engine and retained for the engine's lifetime.  Idempotent on the
// (weight_fp16, rows, cols) tuple.  Returns 1 on success, 0 on failure.
int  cipher_fp8_compute_quantize_weight(
        void* weight_fp16, int rows, int cols, void* stream);

// Returns 1 if `weight_fp16` has an FP8 substitute available.
int  cipher_fp8_compute_weight_ready(void* weight_fp16);

// Lookup helper: returns 1 and fills the FP8 buffer / fp32 scale device
// pointer / shape if the weight is registered, else 0.
int  cipher_fp8_compute_lookup(
        void* weight_fp16,
        void** out_fp8_buf, void** out_scale_dev,
        int* out_rows, int* out_cols);

// Substitute path called by the cublasGemmEx shim.  Mirrors the cuBLAS
// convention used by PyTorch: m_cublas, n_cublas, k_cublas with
// transA = OP_T (weight is k × m row-major) and transB = OP_N
// (activation is k × n col-major == n × k row-major).  Internally:
//   1. Quantize B (activation) inline to FP8 + scale (~2us).
//   2. cublasLtMatmul on the supplied stream, FP8 × FP8 → fp16,
//      with A_scale + B_scale device pointers descaling the output.
//   3. Output written into `c_fp16` (the original cublasGemmEx C buffer).
// Returns 1 on success, 0 on failure (caller falls back to fp16 cuBLAS).
int  cipher_fp8_compute_matmul(
        void* weight_fp16_key,            // pointer used to look up FP8 weight
        const void* activation_fp16,      // B in cuBLAS terms
        void* c_fp16,                     // C in cuBLAS terms (output)
        int m_cublas, int n_cublas, int k_cublas,
        void* stream);

int  cipher_fp8_compute_stats(CipherFp8ComputeStats* out);
void cipher_fp8_compute_report(void);

#ifdef __cplusplus
}
#endif
