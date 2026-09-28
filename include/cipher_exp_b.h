// =============================================================================
// CIPHER — EXP.B: N≤4 Training Validation
// cipher_exp_b.h + cipher_exp_b.cpp
//
// Validates that the N≤4 substitution rule prevents divergence during
// simulated transformer training.
//
// EXPERIMENT DESIGN:
//   - Synthetic 6-layer transformer (feed-forward residual network)
//   - Each layer: x_{l+1} = x_l + GEMM_approx(x_l, W_l)
//   - Two conditions run for 1000 steps each:
//       BASELINE: exact GEMM every layer, every step
//       CIPHER:   substituted GEMM (random Fourier features) with N≤4 rule
//   - SGD with constant LR=0.01, MSE loss on random targets
//   - Success: CIPHER loss tracks baseline within 10% at step 1000
//             Zero divergence events (gradient norm explosion)
//
// RANDOM FEATURE APPROXIMATION:
//   Approximate W·x using random Fourier features (Rahimi & Recht 2007):
//   φ(x) = sqrt(2/D) · [cos(ω_1·x + b_1), ..., cos(ω_D·x + b_D)]
//   W·x ≈ (W_rf)·φ(x)
//   This introduces per-step approximation error ε ~ O(1/sqrt(D)).
//   With D=64 features, ε ≈ 0.125 — measurable but bounded.
//
// N≤4 ENFORCEMENT:
//   sub_counter tracks consecutive substitutions per layer.
//   At counter=4: force exact GEMM, reset counter.
//   Error accumulation from (1+δ)^4 ≈ 1.46 vs (1+δ)^∞ → divergence.
//
// VALIDATES: Theorem 3 (compositional error) + Theorem 4 (convergence preservation)
// =============================================================================

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experiment configuration
#define EXPB_LAYERS        6
#define EXPB_HIDDEN_DIM   32
#define EXPB_BATCH_SIZE   16
#define EXPB_SEQ_LEN       8
#define EXPB_STEPS      1000
#define EXPB_RF_FEATURES  64    // Random Fourier features (D)
#define EXPB_LR         0.01f
#define EXPB_N_MAX          4   // N≤4 rule

// Per-step training metrics
typedef struct {
    float loss;
    float grad_norm;
    uint32_t substitutions;
    uint32_t passthroughs;
    uint32_t n4_fires;        // Times N≤4 rule forced passthrough
} ExpBStepMetrics;

// Full run result
typedef struct {
    ExpBStepMetrics steps[EXPB_STEPS];
    float   final_loss;
    float   avg_loss_last100;
    float   max_grad_norm;
    uint32_t divergence_events;   // grad_norm > 10× initial
    uint32_t total_substitutions;
    uint32_t total_n4_fires;
    float   substitution_rate;
    bool    converged;            // final loss < 2× baseline final loss
} ExpBResult;

// Run baseline (exact GEMM every step)
ExpBResult expb_run_baseline(uint64_t seed);

// Run CIPHER condition (N≤4 substitution schedule)
ExpBResult expb_run_cipher(uint64_t seed);

// Compare two results and print validation report
// Returns true if CIPHER condition passes all criteria
bool expb_validate(const ExpBResult* baseline, const ExpBResult* cipher);

void expb_print_result(const char* label, const ExpBResult* r);

#ifdef __cplusplus
}
#endif
