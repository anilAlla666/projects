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

#ifdef __cplusplus
}
#endif
