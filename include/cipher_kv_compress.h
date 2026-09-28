// CIPHER KV Cache Compression — Stage 8 (KIVI-class 2-bit asymmetric).
//
// Tracks KV-cache regions (large allocations re-used across many decode
// steps), with key-per-channel and value-per-token quantization to 2 bits.
// Residual window of W ≈ 32 recent tokens kept full precision. Layer
// allowlist (first/last layers stay full precision).
//
// Default OFF. Env: CIPHER_KV_COMPRESS=on. Layer-id allow/deny via env
// CIPHER_KV_FULL_LAYERS="0,1,30,31".

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherKvCompressStats {
    int      enabled;
    int      residual_window;
    uint64_t observe_calls;
    uint64_t kv_regions_seen;
    uint64_t kv_regions_compressed;
    uint64_t layers_full_precision;
    uint64_t layers_compressed;
    uint64_t substitutions;
    size_t   bytes_fp16_input;
    size_t   bytes_2bit_output;
    double   compression_ratio;
} CipherKvCompressStats;

int  cipher_kv_compress_init(void);
int  cipher_kv_compress_enabled(void);

// Observe a KV region candidate (layer_id, ptr, bytes).
int  cipher_kv_compress_observe(int layer_id, void* ptr, size_t bytes);

// Returns 1 if `ptr` has a compressed substitute kernel ready.
int  cipher_kv_compress_ready(void* ptr);

int  cipher_kv_compress_stats(CipherKvCompressStats* out);
void cipher_kv_compress_report(void);

// Stage 8 actuation: quantize a fp16 KV block to 2-bit asymmetric per-channel
// (key) or per-token (value). `mode` 0 = key (per-channel), 1 = value
// (per-token). `rows` × `cols` are fp16 elements. Returns 1 on success.
int cipher_kv_compress_quantize(void* fp16_kv, int rows, int cols, int mode);

#ifdef __cplusplus
}
#endif
