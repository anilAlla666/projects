// =============================================================================
// CIPHER — L3.8: Structural Lookup Implementation
// cipher_structural_lookup.cpp
// =============================================================================

#include "cipher_structural_lookup.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Kernel name prefix table — patterns that always force full precision
// Derived from cuBLAS / cuDNN / PyTorch compiled kernel naming conventions
// ---------------------------------------------------------------------------

typedef struct {
    const char*        prefix;         // Match if kernel_name starts with this
    CipherStructReason reason;
    bool               force_fp;       // true = full precision, false = ok
} KernelNameRule;

// Ordered from most-specific to least — first match wins
static const KernelNameRule g_name_rules[] = {
    // Loss computation kernels
    { "nll_loss",                    CIPHER_STRUCT_REASON_LOSS,          true  },
    { "cross_entropy",               CIPHER_STRUCT_REASON_LOSS,          true  },
    { "softmax_xentropy",            CIPHER_STRUCT_REASON_LOSS,          true  },
    { "binary_cross_entropy",        CIPHER_STRUCT_REASON_LOSS,          true  },
    { "kl_div",                      CIPHER_STRUCT_REASON_LOSS,          true  },
    { "mse_loss",                    CIPHER_STRUCT_REASON_LOSS,          true  },
    { "LossCrossEntropy",            CIPHER_STRUCT_REASON_LOSS,          true  },

    // Optimizer / weight update kernels
    { "adam",                        CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "Adam",                        CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "sgd_update",                  CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "adamw",                       CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "AdamW",                       CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "optimizer_step",              CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "weight_update",               CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "step_kernel",                 CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "fused_adam",                  CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "multi_tensor_adam",           CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },

    // Multi-head self-attention — causal masking must be exact
    { "flash_attn",                  CIPHER_STRUCT_REASON_MHSA,          true  },
    { "FlashAttn",                   CIPHER_STRUCT_REASON_MHSA,          true  },
    { "fmha_",                       CIPHER_STRUCT_REASON_MHSA,          true  },
    { "attention_kernel",            CIPHER_STRUCT_REASON_MHSA,          true  },
    { "scaled_dot_product_attention",CIPHER_STRUCT_REASON_MHSA,          true  },
    { "mha_varlen",                  CIPHER_STRUCT_REASON_MHSA,          true  },
    { "attn_fwd",                    CIPHER_STRUCT_REASON_MHSA,          true  },
    { "dropout_attn",                CIPHER_STRUCT_REASON_MHSA,          true  },

    // Backward pass GEMM on attention weights — gradient fidelity
    { "attn_bwd",                    CIPHER_STRUCT_REASON_MHSA,          true  },
    { "flash_bwd",                   CIPHER_STRUCT_REASON_MHSA,          true  },
    { "fmha_bwd",                    CIPHER_STRUCT_REASON_MHSA,          true  },

    // Gradient accumulation kernels
    { "grad_accum",                  CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "reduce_scatter",              CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },
    { "all_reduce_grad",             CIPHER_STRUCT_REASON_WEIGHT_UPDATE, true  },

    // Embedding / first-layer ops
    { "embedding_lookup",            CIPHER_STRUCT_REASON_FIRST_LAYER,   true  },
    { "VocabParallelEmbedding",      CIPHER_STRUCT_REASON_FIRST_LAYER,   true  },
    { "token_embed",                 CIPHER_STRUCT_REASON_FIRST_LAYER,   true  },
    { "position_embed",              CIPHER_STRUCT_REASON_FIRST_LAYER,   true  },

    // Explicitly safe for substitution (FFN, intermediate norms)
    { "linear_gelu",                 CIPHER_STRUCT_REASON_NONE,          false },
    { "linear_silu",                 CIPHER_STRUCT_REASON_NONE,          false },
    { "fused_mlp",                   CIPHER_STRUCT_REASON_NONE,          false },
    { "layer_norm",                  CIPHER_STRUCT_REASON_NONE,          false },
    { "rms_norm",                    CIPHER_STRUCT_REASON_NONE,          false },
    { "gelu",                        CIPHER_STRUCT_REASON_NONE,          false },
    { "silu",                        CIPHER_STRUCT_REASON_NONE,          false },

    // Sentinel
    { NULL,                          CIPHER_STRUCT_REASON_NONE,          false },
};

// ---------------------------------------------------------------------------
// Op-class structural rules (from L3.1 classification)
// These apply when no kernel name match is found
// ---------------------------------------------------------------------------

// op_class 0=GEMM, 1=ATTENTION, 2=CONV, 3=ELEMENTWISE,
//          4=REDUCTION, 5=MEMCPY_TRANSPOSE, 6=ITERATIVE_CUSTOM

static const bool g_opclass_default_fp[7] = {
    false,   // 0: GEMM — substitutable (roofline model)
    true,    // 1: ATTENTION — full precision by default (structural rule)
    false,   // 2: CONV — substitutable
    false,   // 3: ELEMENTWISE — substitutable (Chebyshev)
    false,   // 4: REDUCTION — substitutable (Chebyshev)
    false,   // 5: MEMCPY_TRANSPOSE — substitutable
    true,    // 6: ITERATIVE_CUSTOM — full precision until EDMD derives surrogate
};

// ---------------------------------------------------------------------------
// Per-layer override table (user-controlled, or set by EMA divergence detector)
// ---------------------------------------------------------------------------

#define CIPHER_MAX_OVERRIDE_LAYERS 256
static uint8_t g_layer_overrides[CIPHER_MAX_OVERRIDE_LAYERS] = {0};
// 0 = no override, 1 = forced full precision

// ---------------------------------------------------------------------------
// Kernel name hash cache — 512 slots, open addressing
// Avoids repeated string comparisons on hot path
// ---------------------------------------------------------------------------

#define CIPHER_NAME_CACHE_SIZE  512
#define CIPHER_NAME_CACHE_MASK  (CIPHER_NAME_CACHE_SIZE - 1)

typedef struct {
    uint32_t           name_hash;
    CipherStructResult result;
    CipherStructReason reason;
    bool               valid;
} NameCacheSlot;

static NameCacheSlot g_name_cache[CIPHER_NAME_CACHE_SIZE] = {0};

// FNV-1a 32-bit on first 32 bytes of name (enough for prefix matching)
static uint32_t fnv1a_prefix(const char* s) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < 32 && s[i]; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static CipherStructLookupResult name_cache_lookup(const char* name) {
    CipherStructLookupResult out = {CIPHER_STRUCT_UNKNOWN,
                                    CIPHER_STRUCT_REASON_NONE, false};
    if (!name) return out;
    uint32_t h    = fnv1a_prefix(name);
    uint32_t slot = h & CIPHER_NAME_CACHE_MASK;
    const NameCacheSlot* c = &g_name_cache[slot];
    if (c->valid && c->name_hash == h) {
        out.result    = c->result;
        out.reason    = c->reason;
        out.cache_hit = true;
    }
    return out;
}

static void name_cache_insert(const char* name,
                              CipherStructResult  result,
                              CipherStructReason  reason)
{
    if (!name) return;
    uint32_t h    = fnv1a_prefix(name);
    uint32_t slot = h & CIPHER_NAME_CACHE_MASK;
    g_name_cache[slot].name_hash = h;
    g_name_cache[slot].result    = result;
    g_name_cache[slot].reason    = reason;
    g_name_cache[slot].valid     = true;
}

// ---------------------------------------------------------------------------
// cipher_struct_lookup_init
// ---------------------------------------------------------------------------

void cipher_struct_lookup_init(void) {
    // Clear caches
    for (int i = 0; i < CIPHER_NAME_CACHE_SIZE; i++)
        g_name_cache[i].valid = false;
    for (int i = 0; i < CIPHER_MAX_OVERRIDE_LAYERS; i++)
        g_layer_overrides[i] = 0;

    // Pre-warm cache with common kernel names for zero cold-path cost
    // on first real workload kernel
    for (int i = 0; g_name_rules[i].prefix != NULL; i++) {
        CipherStructResult res = g_name_rules[i].force_fp
            ? CIPHER_STRUCT_FULL_PRECISION
            : CIPHER_STRUCT_SUBSTITUTABLE;
        name_cache_insert(g_name_rules[i].prefix, res, g_name_rules[i].reason);
    }

    fprintf(stderr, "[CIPHER L3.8] Structural lookup initialized. "
                    "%zu rules. Name cache pre-warmed.\n",
            sizeof(g_name_rules)/sizeof(g_name_rules[0]) - 1);
}

// ---------------------------------------------------------------------------
// cipher_struct_lookup — main entry, <10ns on cache hit
// ---------------------------------------------------------------------------

CipherStructLookupResult cipher_struct_lookup(const CipherStructContext* ctx) {
    CipherStructLookupResult out = {CIPHER_STRUCT_SUBSTITUTABLE,
                                    CIPHER_STRUCT_REASON_NONE, false};

    // 1. Warmup phase: always full precision (zero substitution during warmup)
    if (ctx->training_phase == 0) {
        out.result = CIPHER_STRUCT_FULL_PRECISION;
        out.reason = CIPHER_STRUCT_REASON_WARMUP;
        return out;
    }

    // 2. Optimizer step: always full precision
    if (ctx->is_optimizer_step) {
        out.result = CIPHER_STRUCT_FULL_PRECISION;
        out.reason = CIPHER_STRUCT_REASON_WEIGHT_UPDATE;
        return out;
    }

    // 3. Per-layer override table (set by EMA divergence or user)
    if (ctx->layer_idx < CIPHER_MAX_OVERRIDE_LAYERS
        && g_layer_overrides[ctx->layer_idx]) {
        out.result = CIPHER_STRUCT_FULL_PRECISION;
        out.reason = CIPHER_STRUCT_REASON_USER_OVERRIDE;
        return out;
    }

    // 4. Last-3-layers rule — output quality critical
    if (ctx->total_layers > 3
        && ctx->layer_idx >= ctx->total_layers - 3) {
        out.result = CIPHER_STRUCT_FULL_PRECISION;
        out.reason = CIPHER_STRUCT_REASON_LAST_LAYERS;
        return out;
    }

    // 5. Kernel name cache lookup (fast path, ~1ns)
    if (ctx->kernel_name) {
        CipherStructLookupResult cached = name_cache_lookup(ctx->kernel_name);
        if (cached.cache_hit) return cached;

        // Cache miss: walk prefix table
        for (int i = 0; g_name_rules[i].prefix != NULL; i++) {
            if (strncmp(ctx->kernel_name, g_name_rules[i].prefix,
                        strlen(g_name_rules[i].prefix)) == 0)
            {
                out.result = g_name_rules[i].force_fp
                    ? CIPHER_STRUCT_FULL_PRECISION
                    : CIPHER_STRUCT_SUBSTITUTABLE;
                out.reason = g_name_rules[i].reason;
                // Insert into cache for next time
                name_cache_insert(ctx->kernel_name, out.result, out.reason);
                return out;
            }
        }
    }

    // 6. Op-class default rule
    if (ctx->op_class < 7) {
        out.result = g_opclass_default_fp[ctx->op_class]
            ? CIPHER_STRUCT_FULL_PRECISION
            : CIPHER_STRUCT_SUBSTITUTABLE;
        // op_class 1 = ATTENTION is always full precision structurally
        if (ctx->op_class == 1)
            out.reason = CIPHER_STRUCT_REASON_MHSA;
        return out;
    }

    // 7. No rule matched — unknown, let oracle decide
    out.result = CIPHER_STRUCT_UNKNOWN;
    return out;
}

// ---------------------------------------------------------------------------
// Override control
// ---------------------------------------------------------------------------

void cipher_struct_override_layer(uint32_t layer_idx, bool full_precision) {
    if (layer_idx < CIPHER_MAX_OVERRIDE_LAYERS) {
        g_layer_overrides[layer_idx] = full_precision ? 1 : 0;
        fprintf(stderr, "[CIPHER L3.8] Layer %u override: %s\n",
                layer_idx, full_precision ? "FULL_PRECISION" : "cleared");
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

void cipher_struct_lookup_report(void) {
    int cache_filled = 0;
    for (int i = 0; i < CIPHER_NAME_CACHE_SIZE; i++)
        if (g_name_cache[i].valid) cache_filled++;

    int override_count = 0;
    for (int i = 0; i < CIPHER_MAX_OVERRIDE_LAYERS; i++)
        if (g_layer_overrides[i]) override_count++;

    fprintf(stderr,
        "[CIPHER L3.8] Structural Lookup\n"
        "  Rules:            %zu\n"
        "  Name cache:       %d / %d slots filled\n"
        "  Layer overrides:  %d\n"
        "  Op-class FP:      ATTENTION(1), CUSTOM(6)\n"
        "  Op-class OK:      GEMM(0), CONV(2), EW(3), REDUCE(4), XPOSE(5)\n",
        sizeof(g_name_rules)/sizeof(g_name_rules[0]) - 1,
        cache_filled, CIPHER_NAME_CACHE_SIZE,
        override_count);
}
