// =============================================================================
// CIPHER — L3.6 + L3.7 + L3.9: Full Accuracy Oracle
// cipher_oracle.h
//
// Three coordinated safety mechanisms that gate every substitution decision:
//
// L3.6 — N≤4 Rule (substitution_counter gate)
//   Tracks consecutive substitutions per layer. Forces full-precision
//   passthrough at counter=4. Theoretical basis: for residual nets with
//   per-op error ε and Lipschitz constant 1+δ, N consecutive substitutions
//   accumulate error N·δ·ε. N=4: (1+δ)^4 ≈ 1.46 (manageable).
//   N=32: (1+δ)^32 ≈ 21.1 (catastrophic).
//
// L3.7 — EMA Gradient Monitor (divergence detection)
//   Per-layer gradient norm EMA (κ=0.999). Tracks every 100 steps.
//   Divergence > 2σ from baseline → permanently demote layer to full
//   precision. Self-correcting: layer is never re-enabled after demotion.
//   Zero false positives by design (2σ threshold is conservative).
//
// L3.9 — Phase Detector (warmup suppression)
//   Infers training phase from gradient norm variance.
//   Warmup (steps 0-499): gradient variance HIGH → all substitution DISABLED.
//   Convergence (500+): gradient variance LOW → substitution ENABLED.
//   Fine-tune: tighter thresholds.
//
// COMBINED DECISION LOGIC:
//   PERMIT substitution only if ALL of:
//     1. Phase detector: NOT in warmup
//     2. Structural lookup: NOT a protected op (L3.8)
//     3. N≤4 counter: layer has < 4 consecutive substitutions
//     4. EMA monitor: layer NOT permanently demoted
//     5. Confidence: classifier confidence >= MIN_CONFIDENCE threshold
//
// SUCCESS CRITERIA:
//   L3.6: Zero divergence events across 10K training steps
//   L3.7: Detects gradient explosion within 100 steps. Zero false positives.
//   L3.9: Correctly identifies warmup vs convergence on Llama-3 training
// =============================================================================

#pragma once

#include "may13/cipher_liquid_state.h"
#include "may13/cipher_classify.hpp"
#include "may13/cipher_structural_lookup.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Oracle configuration
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  n_max;               // N≤4 rule: max consecutive substitutions (default 4)
    float    ema_kappa;           // EMA decay (default 0.999)
    float    divergence_sigma;    // Sigma multiplier for divergence (default 2.0)
    uint32_t warmup_steps;        // Steps before substitution is enabled (default 500)
    uint8_t  min_confidence;      // Classifier confidence threshold (default 60)
    uint32_t ema_baseline_steps;  // Steps to establish EMA baseline (default 1000)
} CipherOracleConfig;

// Defaults matching build plan spec
#define CIPHER_ORACLE_DEFAULT_CONFIG { \
    .n_max              = 4,     \
    .ema_kappa          = 0.999f,\
    .divergence_sigma   = 2.0f,  \
    .warmup_steps       = 500,   \
    .min_confidence     = 60,    \
    .ema_baseline_steps = 1000,  \
}

// ---------------------------------------------------------------------------
// Oracle state — owns phase detection, per-layer EMA, divergence records
// ---------------------------------------------------------------------------

#define CIPHER_ORACLE_MAX_LAYERS  256

typedef struct {
    float    ema[CIPHER_ORACLE_MAX_LAYERS];       // Per-layer gradient EMA
    float    baseline[CIPHER_ORACLE_MAX_LAYERS];  // Baseline once established
    float    sigma[CIPHER_ORACLE_MAX_LAYERS];     // Running std dev
    bool     baseline_set[CIPHER_ORACLE_MAX_LAYERS];
    bool     permanently_demoted[CIPHER_ORACLE_MAX_LAYERS];
    uint32_t steps_since_update;
    uint32_t demotion_count;
} CipherEmaState;

// Phase detector state
typedef struct {
    float    grad_var_ema;         // EMA of gradient variance
    float    grad_mean_ema;        // EMA of gradient mean
    uint32_t step_count;
    uint8_t  detected_phase;       // 0=warmup, 1=convergence, 2=finetune
    uint32_t phase_entry_step;
    bool     manually_overridden;  // True if set by user, not auto-detected
} CipherPhaseState;

typedef struct {
    CipherOracleConfig  cfg;
    CipherEmaState      ema;
    CipherPhaseState    phase;
    CipherLiquidStateMgr* liquid;  // Shared liquid state (F4)

    // Per-layer substitution counters (mirrors F4 state for fast local access)
    uint8_t  sub_counter[CIPHER_ORACLE_MAX_LAYERS];
    uint8_t  force_passthrough[CIPHER_ORACLE_MAX_LAYERS];

    // Topological phase detection — kernel class histogram
    uint64_t topo_class_counts[8];   // per op_class counter for inference detection

    // Stats
    uint64_t total_decisions;
    uint64_t permitted;
    uint64_t denied_warmup;
    uint64_t denied_structural;
    uint64_t denied_n4;
    uint64_t denied_ema_demotion;
    uint64_t denied_low_confidence;

    // MFU feedback — read from telemetry, adjusts min_confidence
    float    last_mfu_fraction;       // Last observed MFU (0-1)
    int8_t   mfu_confidence_adjust;   // Current adjustment to min_confidence

    // Billing counters — geometry only, no model knowledge
    uint64_t billing_gemm_total;        // Total GEMM dispatches
    uint64_t billing_gemm_substituted;  // GEMMs where Koopman fired
    uint64_t billing_gemm_passthrough;  // GEMMs that fell through to cuBLAS
    double   billing_flops_substituted; // FLOPs handled by O(1) path
    double   billing_flops_passthrough; // FLOPs handled by cuBLAS
    uint64_t billing_nongemm_total;     // Non-GEMM kernel dispatches
    uint64_t billing_nongemm_substituted; // Non-GEMM substitutions (Chebyshev etc)

    bool initialized;
} CipherOracleState;

// ---------------------------------------------------------------------------
// Oracle decision input — all context for a single kernel dispatch
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t            layer_idx;
    uint32_t            total_layers;
    uint8_t             op_class;        // From L3.1
    uint8_t             confidence;      // From L3.1 (0-100)
    const char*         kernel_name;     // For structural lookup
    bool                is_backward;
    bool                is_optimizer;
} CipherOracleQuery;

// Oracle decision
typedef enum {
    CIPHER_ORACLE_PERMIT   = 0,   // Go ahead and substitute
    CIPHER_ORACLE_DENY     = 1,   // Run full precision
} CipherOracleDecision;

typedef struct {
    CipherOracleDecision  decision;
    const char*           reason;    // Human-readable denial reason
} CipherOracleResult;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize oracle. Must call after cipher_liquid_state_init().
void cipher_oracle_init(CipherOracleState*   state,
                        CipherLiquidStateMgr* liquid,
                        const CipherOracleConfig* cfg);  // NULL = use defaults

// Main decision gate — called on every kernel dispatch.
// Hot path: reads counters + phase flag, <50ns.
CipherOracleResult cipher_oracle_decide(CipherOracleState*       state,
                                        const CipherOracleQuery* query);

// Feedback: record that a substitution was executed for this layer.
// Updates N≤4 counter in oracle + liquid state.
void cipher_oracle_record_substitution(CipherOracleState* state,
                                       uint32_t layer_idx);

// Feedback: record that full precision was used (resets N≤4 counter).
void cipher_oracle_record_passthrough(CipherOracleState* state,
                                      uint32_t layer_idx);

// Training loop hook: update EMA gradient norms + phase detection.
// Call every 100 training steps with per-layer gradient norms.
// grad_norms: array of per-layer ||∇L||, length = num_layers
void cipher_oracle_update_gradients(CipherOracleState* state,
                                    const float*        grad_norms,
                                    uint32_t            num_layers,
                                    uint32_t            global_step);

// Force phase override (useful for fine-tuning without warmup)
void cipher_oracle_set_phase(CipherOracleState* state, uint8_t phase);

// Report
void cipher_oracle_report(const CipherOracleState* state);

// Billing: record a GEMM dispatch outcome (geometry only).
void cipher_oracle_bill_gemm(CipherOracleState* state,
                             uint32_t M, uint32_t N, uint32_t K,
                             bool substituted);

// Billing: record a non-GEMM dispatch outcome.
void cipher_oracle_bill_nongemm(CipherOracleState* state, bool substituted);

// Billing report — print all counters to stderr.
void cipher_oracle_billing_report(const CipherOracleState* state);

// MFU feedback — call periodically with current MFU from telemetry.
// Adjusts oracle aggressiveness based on observed hardware utilization.
void cipher_oracle_update_mfu(CipherOracleState* state, float mfu_fraction);

#ifdef __cplusplus
}
#endif
