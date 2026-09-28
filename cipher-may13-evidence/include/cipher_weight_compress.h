// CIPHER Weight Transport Compression — Stage 7 (W4A16 Machete-class)
//
// Detects stable weight tensors (same address across 1000+ GEMM calls,
// size > 1MB) and quantizes fp16→INT4 with per-channel absmax scales.
// Compressed weights live in cipher_vmm pool. Substitution is gated by
// per-shape relative-error threshold: if ||fp16_out - cmp_out|| / ||fp16_out||
// > threshold, the shape stays on fp16 passthrough.
//
// Default OFF. Env: CIPHER_WEIGHT_COMPRESS=on.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherWeightCompressStats {
    int      enabled;
    uint64_t observe_calls;
    uint64_t weights_seen;            // unique pointers
    uint64_t weights_compressed;      // crossed stability threshold
    uint64_t correctness_pass;
    uint64_t correctness_fail;
    uint64_t substitutions;
    uint64_t passthroughs;
    size_t   bytes_fp16_input;
    size_t   bytes_int4_output;
    double   compression_ratio;       // bytes_fp16 / bytes_int4
} CipherWeightCompressStats;

int  cipher_weight_compress_init(void);
int  cipher_weight_compress_enabled(void);

// Observe a candidate weight at GEMM intercept time. Threshold = 1000 hits
// at the same (ptr, bytes). When crossed, we mark the weight as compressible
// (the actual quantization is performed lazily on the next observation that
// follows promotion).
int  cipher_weight_compress_observe(void* ptr, size_t bytes);

// Returns 1 if `ptr` has a compressed substitute available and the
// correctness gate has passed.
int  cipher_weight_compress_ready(void* ptr);

// Force the correctness gate result for a pointer (called after offline
// validation by the substitution path).
int  cipher_weight_compress_set_gate(void* ptr, double rel_error, double threshold);

int  cipher_weight_compress_stats(CipherWeightCompressStats* out);
void cipher_weight_compress_report(void);

// Stage 7 actuation: quantize a stable fp16 weight to INT4 with per-channel
// absmax scales. Allocates output buffers via cudaMalloc; caller does NOT
// own them (engine retains for lifetime). Returns 1 on success.
//   `rows`, `cols` describe the weight as a 2D matrix of fp16 elements.
int cipher_weight_compress_quantize(void* fp16_weight, int rows, int cols);

// Lookup: returns 1 and fills out_int4 / out_scales / out_rows / out_cols if
// `fp16_weight` has a compressed copy stored, else 0. The output pointers
// are valid for the lifetime of the engine.
int cipher_weight_compress_lookup(void* fp16_weight,
                                  void** out_int4_buf,
                                  void** out_scale_buf,
                                  int* out_rows, int* out_cols);

// Lookup the transposed (GEMV-optimized) buffer. Layout: B_int4_T[N, K/2].
int cipher_weight_compress_lookup_T(void* fp16_weight,
                                    void** out_int4_T_buf,
                                    void** out_scale_buf,
                                    int* out_rows, int* out_cols);

// Run the dequant-fused INT4 GEMM:  C[M,N] = A[M,K] @ dequant(B_int4[K,N/2], scales[K/128,N])
// All pointers are device pointers. Returns 1 on success.
// Layout matches Stage-7 quantize output (along-K groupwise, AWQ-compatible).
// Internally dispatches:
//    M == 1  → optimized GEMV kernel (memory-bandwidth bound, decode path)
//    M >= 2  → naive tiled kernel (correct, slower)
int cipher_weight_compress_int4_gemm(
    void* a_fp16, void* b_int4, void* b_scales, void* c_fp16,
    int M, int N, int K, void* stream_handle);

// Multi-row INT4 GEMV reading the **transposed** B_T[N, K/2] layout.
// `b_int4` here is the transposed buffer pointer (use cipher_weight_compress_lookup_T).
// gridDim = (N/8, M); each (block_x, block_y) computes 8 output cols of one row.
// Used by the GEMM dispatcher for M < 16; exposed for direct use too.
int cipher_weight_compress_int4_gemv(
    void* a_fp16, void* b_int4_T, void* b_scales, void* c_fp16,
    int M, int N, int K, void* stream_handle);

// Step 2 megakernels — fused gate+up+silu·mul and down+residual at M=1.
// Both kernels read the same (N, K/2) transposed INT4 layout produced by
// cipher_weight_compress_quantize. Each warp computes one output column.
int cipher_weight_compress_int4_silu_mul(
    void* a_fp16, void* bg_T, void* bs_g, void* bu_T, void* bs_u, void* c_fp16,
    int M, int N, int K, void* stream_handle);

int cipher_weight_compress_int4_down_residual(
    void* a_fp16, void* b_T, void* b_scales, void* residual_fp16, void* c_fp16,
    int M, int N, int K, void* stream_handle);

// Phase 2: Marlin INT4 GEMM.
// Repack our existing int4_packed[K, N/2] + scales[K/G, N] into Marlin's
// XOR-swizzled int32 layout. Stores the result in the engine's per-weight
// slot and returns 1 on success. Lookup the result via
// cipher_weight_compress_lookup_marlin().
int cipher_weight_compress_repack_marlin(void* fp16_weight);

// Lookup Marlin-format buffers for a weight (created by repack_marlin).
//   out_B: K/16 × N*2 int32      — Marlin packed weight
//   out_S: K/G × N fp16          — column-permuted scales
int cipher_weight_compress_lookup_marlin(void* fp16_weight,
                                           void** out_B, void** out_S,
                                           int* out_K, int* out_N, int* out_G);

// Run Marlin INT4 GEMM. C[M,N] = A[M,K] @ dequant(Marlin_B, S).
// Picks a (thread_k, thread_n) configuration based on M; allocates a
// transient int32 workspace internally.
int cipher_weight_compress_marlin_gemm(
    void* a_fp16, void* marlin_B, void* marlin_S, void* c_fp16,
    int M, int N, int K, int G, void* stream_handle);

#ifdef __cplusplus
}
#endif
