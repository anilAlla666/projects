// =============================================================================
// CIPHER — L3.10: Full LNN Integration (CfC Architecture)
// cipher_lnn.h
//
// Replaces the classify→structural→oracle→recipe dispatch chain with a single
// Closed-form Continuous-time (CfC) neural network forward pass.
//
// ARCHITECTURE:
//   Input (48-dim):
//     [0-6]   op class one-hot (7 families)
//     [7-9]   grid dims: log2(gx), log2(gy), log2(gz)  (normalized)
//     [10]    log2(block_size) / 10
//     [11]    shmem / 65536.0
//     [12]    substitution_counter / 4.0       (from liquid state)
//     [13]    ema_gradient_norm (normalized)    (from liquid state)
//     [14]    training_phase (0=warmup, 1=convergence, 2=finetune)
//     [15]    error_accumulation (normalized)   (from liquid state)
//     [16]    sm_idle_fraction                  (hardware telemetry)
//     [17]    hbm_bw_utilized                   (hardware telemetry)
//     [18]    l2_hit_rate                       (hardware telemetry)
//     [19]    nvlink_utilization                (hardware telemetry)
//     [20-27] workload_rhythm (last 8 op classes, one-hot encoded)
//     [28-31] nccl_congestion_history (last 4 AllReduce durations, normalized)
//     [32-47] reserved (zeros) — for L2.7 LNN unification
//
//   Hidden (64-dim CfC):
//     Persistent liquid state h ∈ ℝ^64, updated on every forward pass.
//     Closed-form ODE: h' = g·(1 - e^{-τΔt}) + h·e^{-τΔt}
//     where τ = softplus(W_τ·[x;h] + b_τ)    (time constant)
//           g = tanh(W_g·[x;h] + b_g)         (target state)
//
//   Output (12-dim):
//     [0]     substitute_logit (>0 = substitute, <0 = passthrough)
//     [1-7]   recipe_type logits (7 operation families)
//     [8]     confidence (sigmoid → [0,1])
//     [9-11]  recipe params (e.g. tile_log2, stages, eff_hint)
//
// WEIGHT INITIALIZATION:
//   Analytical encoding of Layer 3 rules.
//   W_g encodes: GEMM/ATTN/EW/REDUCE → positive substitute signal.
//   W_τ encodes: warmup phase → high time constant (slow to adapt during warmup).
//   W_out encodes: recipe type selection from op class.
//   Zero training required for day-one baseline accuracy.
//   Koopman-linearity loss improves accuracy during online learning.
//
// LATENCY TARGET: <2µs forward pass (from build plan L3.10).
// RECOVERY TARGET: 95%+ TFLOPS recovery.
// DEPENDENCIES: F4 (liquid state), L3.1 (classification features).
// =============================================================================

#pragma once

#include "may13/cipher_liquid_state.h"
#include "may13/cipher_classify.hpp"
#include <stdint.h>
#include <stdbool.h>

#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------

#define CIPHER_LNN_INPUT_DIM     48
#define CIPHER_LNN_HIDDEN_DIM    64
#define CIPHER_LNN_OUTPUT_DIM    12
#define CIPHER_LNN_CONCAT_DIM    (CIPHER_LNN_INPUT_DIM + CIPHER_LNN_HIDDEN_DIM)

// Output indices
#define CIPHER_LNN_OUT_SUBSTITUTE  0   // substitute logit
#define CIPHER_LNN_OUT_RECIPE_BASE 1   // recipe_type logits [1-7]
#define CIPHER_LNN_OUT_CONFIDENCE  8   // confidence ∈ [0,1]
#define CIPHER_LNN_OUT_PARAM_0     9   // recipe param 0 (tile log2)
#define CIPHER_LNN_OUT_PARAM_1    10   // recipe param 1 (stages)
#define CIPHER_LNN_OUT_PARAM_2    11   // recipe param 2 (efficiency hint)

// ---------------------------------------------------------------------------
// LNN decision — output of one forward pass
// ---------------------------------------------------------------------------

typedef struct {
    bool             should_substitute;
    uint8_t  recipe_type;  // 0-6, matches cipher::OpClass
    float            confidence;        // [0,1]
    float            recipe_params[3];  // Passed to recipe engine
    float            substitute_logit;  // Raw logit before threshold
    uint64_t         forward_pass_ns;   // Latency of this pass
} CipherLnnDecision;

// ---------------------------------------------------------------------------
// CfC weights — two gate matrices + output head
// ---------------------------------------------------------------------------

typedef struct {
    // Gate τ (time constant): W_τ ∈ ℝ^{H×(I+H)}, b_τ ∈ ℝ^H
    float W_tau[CIPHER_LNN_HIDDEN_DIM][CIPHER_LNN_CONCAT_DIM];
    float b_tau[CIPHER_LNN_HIDDEN_DIM];

    // Gate g (target state): W_g ∈ ℝ^{H×(I+H)}, b_g ∈ ℝ^H
    float W_g[CIPHER_LNN_HIDDEN_DIM][CIPHER_LNN_CONCAT_DIM];
    float b_g[CIPHER_LNN_HIDDEN_DIM];

    // Output head: W_out ∈ ℝ^{O×H}, b_out ∈ ℝ^O
    float W_out[CIPHER_LNN_OUTPUT_DIM][CIPHER_LNN_HIDDEN_DIM];
    float b_out[CIPHER_LNN_OUTPUT_DIM];
} CipherLnnWeights;

// ---------------------------------------------------------------------------
// LNN runtime state
// ---------------------------------------------------------------------------

typedef struct {
    CipherLnnWeights  weights;
    float             h[CIPHER_LNN_HIDDEN_DIM];  // Persistent CfC hidden state
    float             delta_t;                    // Time step (default 1.0)

    // Statistics
    uint64_t  forward_pass_count;
    float     avg_forward_ns;
    float     substitution_rate;
    uint64_t  substitutions;
    uint64_t  passthroughs;

    // Koopman linearity loss tracking
    float     koopman_loss_ema;   // EMA of linearity loss

    bool      initialized;
} CipherLnnState;

// ---------------------------------------------------------------------------
// Input feature vector (for explicit construction in tests)
// ---------------------------------------------------------------------------

typedef struct {
    float x[CIPHER_LNN_INPUT_DIM];
} CipherLnnInput;


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize LNN. Loads analytical weights encoding Layer 3 rules.
// No training required — starts at baseline accuracy immediately.
void cipher_lnn_init(CipherLnnState* state);

// Build input feature vector from kernel launch parameters + liquid state.
CipherLnnInput cipher_lnn_build_input(
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid);

// Run one CfC forward pass. Updates persistent hidden state h.
// This is the hot path — must complete in <2µs.
CipherLnnDecision cipher_lnn_forward(CipherLnnState*      state,
                                     const CipherLnnInput* input);

// Online weight update via Koopman-linearity gradient.
// Called after each substitution with observed outcome.
// Enforces: K·ψ(h_t) ≈ ψ(h_{t+1}) in latent space.
void cipher_lnn_koopman_update(CipherLnnState* state,
                               const float*    h_before,
                               const float*    h_after,
                               float           learning_rate);

// Compatibility shim: run LNN and merge with existing Layer 3 signal.
// If LNN confidence < threshold, falls back to rule-based dispatch.
CipherLnnDecision cipher_lnn_decide(
    CipherLnnState*             state,
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid,
    float                       confidence_threshold);

void cipher_lnn_reset_hidden(CipherLnnState* state);
void cipher_lnn_report(const CipherLnnState* state);

#ifdef __cplusplus
}
#endif
