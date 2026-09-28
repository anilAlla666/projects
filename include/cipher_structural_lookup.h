// =============================================================================
// CIPHER — L3.8: Accuracy Oracle — Structural Lookup
// cipher_structural_lookup.h
//
// Zero-compute, zero-latency table of operations that are ALWAYS run at full
// precision regardless of what the classification engine says. These rules are
// hardcoded from transformer architecture analysis — no learning needed.
//
// RULES (from build plan):
//   ALWAYS FULL PRECISION:
//     - Multi-Head Self-Attention (MHSA) — causal masking must be exact
//     - Last 3 transformer layers — output quality-critical
//     - Loss computation — gradient signal integrity
//     - Weight update kernels — optimizer step must be exact
//     - First layer embeddings — distribution shift if approximated
//   SUBSTITUTION-TOLERANT:
//     - FFN middle layers (residual networks tolerate per-op error)
//     - Intermediate LayerNorm (bounded error via Chebyshev)
//     - Non-final attention projections (residual path absorbs error)
//
// IMPLEMENTATION:
//   kernel_name string matching + structural position check.
//   O(1) lookup via hash on kernel_name prefix. Ships day one.
//
// SUCCESS CRITERION: Rule table complete. <10ns lookup.
// DEPENDENCY: L3.1 (classification engine provides op_class + kernel_name).
// =============================================================================

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Structural rule result
// ---------------------------------------------------------------------------

typedef enum {
    CIPHER_STRUCT_SUBSTITUTABLE   = 0,   // Safe to substitute
    CIPHER_STRUCT_FULL_PRECISION  = 1,   // Must run full precision
    CIPHER_STRUCT_UNKNOWN         = 2,   // No rule matched — oracle decides
} CipherStructResult;

// Reason codes for telemetry / debugging
typedef enum {
    CIPHER_STRUCT_REASON_NONE             = 0,
    CIPHER_STRUCT_REASON_MHSA             = 1,   // Multi-head self-attention
    CIPHER_STRUCT_REASON_LAST_LAYERS      = 2,   // Last 3 transformer layers
    CIPHER_STRUCT_REASON_LOSS             = 3,   // Loss computation
    CIPHER_STRUCT_REASON_WEIGHT_UPDATE    = 4,   // Optimizer step
    CIPHER_STRUCT_REASON_FIRST_LAYER      = 5,   // Embedding / first layer
    CIPHER_STRUCT_REASON_WARMUP           = 6,   // Training warmup phase
    CIPHER_STRUCT_REASON_USER_OVERRIDE    = 7,   // Manually pinned by user
} CipherStructReason;

typedef struct {
    CipherStructResult  result;
    CipherStructReason  reason;
    bool                cache_hit;   // Whether lookup hit the name hash cache
} CipherStructLookupResult;

// ---------------------------------------------------------------------------
// Context provided to the lookup — from kernel descriptor + liquid state
// ---------------------------------------------------------------------------

typedef struct {
    const char*  kernel_name;       // From cuFuncGetName() or cuModuleGetFunction()
    uint32_t     layer_idx;         // Transformer layer index (0 = first)
    uint32_t     total_layers;      // Total transformer layers in model
    uint8_t      op_class;          // From L3.1 classification engine
    uint8_t      training_phase;    // 0=warmup, 1=convergence, 2=finetune
    bool         is_backward;       // True if this is a backward pass kernel
    bool         is_optimizer_step; // True if Adam/SGD weight update
} CipherStructContext;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize the lookup table. Call once at cipher_init().
void cipher_struct_lookup_init(void);

// Main lookup — <10ns on hot path (pure logic + string prefix cache)
CipherStructLookupResult cipher_struct_lookup(const CipherStructContext* ctx);

// Override: force a specific layer to always run full precision.
// Used for debugging / safety testing.
void cipher_struct_override_layer(uint32_t layer_idx, bool full_precision);

// Print the full rule table to stderr.
void cipher_struct_lookup_report(void);

#ifdef __cplusplus
}
#endif
