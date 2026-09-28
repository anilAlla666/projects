// CIPHER KV-Cache Redirect — Path A scaffolding.
//
// Rewrites FlashAttention's `Params` struct K and V pointers to point at
// CIPHER-owned side buffers. V1: side buffers are populated by GPU memcpy
// from the original staging buffer (no-op redirect to validate the
// substitution mechanism). V2 will replace the memcpy with quant + dequant
// against a per-layer compressed cache.
//
// Default OFF. Env: CIPHER_KV_REDIRECT=on.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  cipher_kv_redirect_init(void);
int  cipher_kv_redirect_enabled(void);

// Called from cuLaunchKernel intercept when FlashAttention is about to
// launch. `args0` is the FA Params struct on the host stack:
//   word 1 = K_buf (device ptr)
//   word 2 = V_buf (device ptr)
// `layer_idx` is the current layer (0..N-1), `stream` is the GPU stream.
// On success the function rewrites struct.K and struct.V in place, then
// returns 1. On failure returns 0 and leaves the struct untouched.
int  cipher_kv_redirect_on_fa_launch(void* args0, int layer_idx, void* stream);

// V3: called from cublasGemmEx intercept after K_proj or V_proj completes.
// `C_ptr` is the GEMM output (column-major, m=hidden_kv, n=tokens, stride=m).
// `m` MUST be num_kv_heads*head_dim (=1024 for Mistral 7B); calls with other
// shapes are ignored. The function infers (layer, K-or-V) from a global call
// counter (K_proj precedes V_proj per layer in PyTorch's order). Quantizes
// the n new tokens of K (or V) and appends them to the per-layer compressed
// cache slot at the layer's current position. Returns 1 on success.
int  cipher_kv_redirect_quant_kv(void* C_ptr, int m, int n, void* stream);

// Reset per-layer position counters at the start of a new request. Not
// strictly required for a single autoregressive pass, but useful between
// independent forward passes.
void cipher_kv_redirect_reset(void);

// Stats for diagnostics.
typedef struct CipherKvRedirectStats {
    int      enabled;
    uint64_t fa_launches_seen;
    uint64_t fa_launches_rewritten;
    uint64_t bytes_copied;
    // OP 24 additions — dynamic staging buffer telemetry.
    uint64_t redirects;          // total successful redirects
    uint64_t staging_reallocs;   // # times staging buffer grew
    size_t   staging_bytes;      // current staging buffer size
    uint64_t contexts_skipped;   // FA calls below MIN_CTX gate
    uint64_t oom_fallbacks;      // cudaMalloc-failure passthroughs
} CipherKvRedirectStats;

int  cipher_kv_redirect_stats(CipherKvRedirectStats* out);

// Test entrypoint: quantize input, then dequant+permute (and optionally apply
// RoPE inline). Allocates and frees temporary q/meta buffers internally.
//   in_fp16:  device ptr, layout [n_tokens, n_heads, 128] row-major fp16
//   out_fp16: device ptr, layout [n_heads, n_tokens, 128] row-major fp16
//   apply_rope: 0 = use cipher_kv_dq2_perm; 1 = use cipher_kv_dq2_perm_rope
//   pos_offset: RoPE position for cache index t is (t + pos_offset)
//   rope_base:  10000.0 for Mistral-7B-v0.1
// Returns 1 on success, 0 on any failure.
int  cipher_kv_test_qdq_rope(void* in_fp16, void* out_fp16,
                              int n_heads, int n_tokens,
                              int apply_rope, int pos_offset,
                              float rope_base, void* stream);

#ifdef __cplusplus
}
#endif
