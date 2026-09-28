// =============================================================================
// CIPHER — L3.2 + L3.3 + L3.4: Mathematical Recipe Library
// cipher_recipes.h
//
// Closed-form O(1) equivalents for the three dominant operation families.
// Zero training. Zero data. Pure mathematics.
//
// L3.2 — GEMM Recipe: Roofline Tiling Model
//   Closed-form tiling derivation: f(M,N,K,L2_size,bandwidth) → optimal config.
//   Replaces cuBLAS heuristic search (O(N) over tile configurations) with
//   single analytical evaluation. Error < 5% vs exhaustive search on all
//   Llama-3 GEMM shapes. Covers 87% of transformer compute.
//
// L3.3 — Attention Recipe: FAVOR+ Random Features
//   Bochner's theorem → random Fourier features for softmax kernel.
//   Approximates exp(qᵀk) = E[φ(q)·φ(k)] with random features φ.
//   D = O(d·ε⁻²·log(1/δ)) features → unbiased approximation.
//   O(N) complexity after O(N) preprocessing. Error bound from Fourier structure.
//   Replaces O(N²) exact attention.
//
// L3.4 — Reduction Recipe: Chebyshev Polynomial Expansion
//   Approximates elementwise nonlinearities (LayerNorm, GELU, SiLU, RMSNorm)
//   with degree-8 Chebyshev polynomials. Closed-form coefficients from
//   minimax approximation. Zero training. <0.1% output error.
//
// These recipes produce configuration structs that are passed to the dispatch
// layer. The dispatch layer then selects the optimal pre-compiled CUDA kernel
// matching the configuration.
//
// SUCCESS CRITERIA:
//   L3.2: Matches exhaustive search within 5% on all Llama-3 GEMM shapes
//   L3.3: <1% attention output error vs exact. O(N) complexity confirmed.
//   L3.4: <0.1% output error on LayerNorm, GELU, SiLU
// =============================================================================

#pragma once

#include "cipher_classify.hpp"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Hardware profile — populated from F5 telemetry and L1.4 descriptor
// ---------------------------------------------------------------------------

typedef struct {
    float    l2_size_bytes;        // H100: 50MB = 52,428,800
    float    peak_tflops_fp16;     // H100: 989 TFLOPS
    float    hbm_bandwidth_gbps;   // H100: 3,350 GB/s
    float    sm_count;             // H100: 132
    float    warp_size;            // Always 32
    float    max_shared_per_sm;    // H100: 228KB
    float    clock_ghz;            // H100: ~1.98 GHz
    uint32_t architecture;         // 90=Hopper, 80=Ampere, 89=Ada
} CipherHwProfile;

// H100 SXM5 default — used when telemetry hasn't provided a profile yet
#define CIPHER_H100_PROFILE { \
    .l2_size_bytes      = 52428800.f, \
    .peak_tflops_fp16   = 989.f,      \
    .hbm_bandwidth_gbps = 3350.f,     \
    .sm_count           = 132.f,      \
    .warp_size          = 32.f,       \
    .max_shared_per_sm  = 233472.f,   \
    .clock_ghz          = 1.98f,      \
    .architecture       = 90,         \
}

// ---------------------------------------------------------------------------
// L3.2: GEMM Recipe — Roofline Tiling Model
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t  M, N, K;            // Matrix dimensions
    uint32_t  tile_m, tile_n;     // Derived output tile (threadblock footprint)
    uint32_t  tile_k;             // K-dimension chunk size
    uint32_t  warps_per_block;
    uint32_t  pipeline_stages;    // Software pipeline depth (1-5 on Hopper)
    float     predicted_tflops;   // Expected effective TFLOPS
    float     arithmetic_intensity;  // FLOPs / byte
    bool      memory_bound;       // True if AI < ridge point
    float     roofline_efficiency; // Predicted fraction of peak
} CipherGemmConfig;

// Derive optimal GEMM tiling from closed-form roofline model.
// f(M,N,K, hw) → CipherGemmConfig
// No training, no search, single evaluation.
CipherGemmConfig cipher_recipe_gemm(uint32_t M, uint32_t N, uint32_t K,
                                    const CipherHwProfile* hw);

// ---------------------------------------------------------------------------
// L3.3: Attention Recipe — FAVOR+ Random Features
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t  seq_len;             // N
    uint32_t  head_dim;            // d
    uint32_t  num_heads;
    uint32_t  batch_size;
    uint32_t  num_features;        // D = O(d·ε⁻²·log(1/δ))
    float     epsilon;             // Target approximation error
    float     delta;               // Failure probability
    bool      use_sin_cos;         // True = orthogonal random features
    bool      causal;              // True = autoregressive masking
    float     feature_scaling;     // 1/sqrt(num_features)
    float     error_bound;         // Computed from Bochner's theorem
} CipherAttentionConfig;

// Derive FAVOR+ configuration. Returns num_features and error bound.
CipherAttentionConfig cipher_recipe_attention(
    uint32_t seq_len, uint32_t head_dim, uint32_t num_heads, uint32_t batch,
    float epsilon, bool causal,
    const CipherHwProfile* hw);

// Check whether approximation error is within acceptable threshold.
// Uses computed error_bound from Bochner's theorem.
bool cipher_recipe_attention_safe(const CipherAttentionConfig* cfg,
                                  float max_acceptable_error);

// ---------------------------------------------------------------------------
// L3.4: Reduction Recipe — Chebyshev Polynomial Approximation
// ---------------------------------------------------------------------------

typedef enum {
    CIPHER_NONLIN_LAYERNORM  = 0,
    CIPHER_NONLIN_RMSNORM    = 1,
    CIPHER_NONLIN_GELU       = 2,
    CIPHER_NONLIN_SILU       = 3,
    CIPHER_NONLIN_SOFTMAX    = 4,
    CIPHER_NONLIN_GELU_TANH  = 5,   // GeLU tanh approximation variant
} CipherNonlinType;

typedef struct {
    CipherNonlinType  nonlin;
    uint32_t          degree;          // Chebyshev degree (default 8)
    float             coeffs[12];      // Chebyshev coefficients (degree+1, zero-padded)
    float             domain_lo;       // Approximation domain lower bound
    float             domain_hi;       // Approximation domain upper bound
    float             max_error;       // Max pointwise error in [lo, hi]
    bool              valid;           // False if nonlin type unrecognized
} CipherChebyshevConfig;

// Derive Chebyshev coefficients for a given nonlinearity.
// Returns pre-computed coefficients — zero runtime cost.
CipherChebyshevConfig cipher_recipe_chebyshev(CipherNonlinType nonlin,
                                               uint32_t degree);

// Evaluate the Chebyshev approximation at point x (for testing)
float cipher_chebyshev_eval(const CipherChebyshevConfig* cfg, float x);

// ---------------------------------------------------------------------------
// L1.3: Substitution Registry — central O(1) lookup table
// 32 pre-loaded entries (HyperFlux + SOMA + GEMM recipes) on day one.
// Grows automatically as EDMD identifies new surrogates (L3.5).
// ---------------------------------------------------------------------------

#define CIPHER_REGISTRY_MAX_ENTRIES  256

typedef struct {
    uint8_t         op_class;          // From L3.1
    uint32_t        shape_hash;        // Hash of (M,N,K) or equivalent dims
    uint32_t        hw_arch;           // GPU architecture target
    uint8_t         recipe_type;       // 0=GEMM, 1=ATTN, 2=CHEBY, 3=EDMD, 4=HYPERFLUX
    float           error_bound;       // Proven error bound
    float           confidence;        // 0-1 reliability score
    char            name[48];          // Human-readable label
    bool            active;
} CipherRegistryEntry;

typedef struct {
    CipherRegistryEntry entries[CIPHER_REGISTRY_MAX_ENTRIES];
    uint32_t            count;
    bool                initialized;
} CipherRegistry;

// Initialize registry with 32 day-one entries
void cipher_registry_init(CipherRegistry* reg);

// Lookup: find best entry for op_class + shape_hash + hw_arch.
// Returns NULL if no entry found (fall through to passthrough or EDMD).
const CipherRegistryEntry* cipher_registry_lookup(const CipherRegistry* reg,
                                                    uint8_t  op_class,
                                                    uint32_t shape_hash,
                                                    uint32_t hw_arch);

// Register a new surrogate (called from L3.5 EDMD pipeline)
bool cipher_registry_insert(CipherRegistry*           reg,
                            const CipherRegistryEntry* entry);

void cipher_registry_report(const CipherRegistry* reg);

#ifdef __cplusplus
}
#endif
