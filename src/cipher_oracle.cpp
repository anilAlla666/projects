// =============================================================================
// CIPHER — L3.6 + L3.7 + L3.9: Accuracy Oracle Implementation
// cipher_oracle.cpp
// =============================================================================

#include "cipher_oracle.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

// ---------------------------------------------------------------------------
// cipher_oracle_init
// ---------------------------------------------------------------------------

void cipher_oracle_init(CipherOracleState*    state,
                        CipherLiquidStateMgr* liquid,
                        const CipherOracleConfig* cfg)
{
    memset(state, 0, sizeof(*state));
    state->liquid = liquid;

    // Apply config (defaults if NULL)
    if (cfg) {
        state->cfg = *cfg;
    } else {
        CipherOracleConfig def = CIPHER_ORACLE_DEFAULT_CONFIG;
        state->cfg = def;
    }

    // Initialize EMA sigma to 1.0 — prevents 2σ=0 false positives at start
    for (int i = 0; i < CIPHER_ORACLE_MAX_LAYERS; i++) {
        state->ema.sigma[i] = 1.0f;
    }

    // Start in warmup phase
    state->phase.detected_phase = 0;
    state->phase.step_count     = 0;

    cipher_struct_lookup_init();

    state->initialized = true;
    fprintf(stderr,
        "[CIPHER ORACLE] Initialized.\n"
        "  N_max:          %u\n"
        "  EMA κ:          %.3f\n"
        "  Divergence:     %.1fσ\n"
        "  Warmup steps:   %u\n"
        "  Min confidence: %u%%\n",
        state->cfg.n_max,
        state->cfg.ema_kappa,
        state->cfg.divergence_sigma,
        state->cfg.warmup_steps,
        state->cfg.min_confidence);
}

// ---------------------------------------------------------------------------
// L3.9: Phase detection — infer training phase from gradient variance
// ---------------------------------------------------------------------------

static void update_phase_detector(CipherOracleState* state,
                                  const float*        grad_norms,
                                  uint32_t            num_layers,
                                  uint32_t            global_step)
{
    // Always advance step counter — even in manual override mode
    // so EMA baseline establishment is not blocked.
    state->phase.step_count = global_step;

    if (state->phase.manually_overridden) return;

    // Hard warmup: first N steps always disabled regardless of gradients
    if (global_step < state->cfg.warmup_steps) {
        state->phase.detected_phase = 0;  // WARMUP
        return;
    }

    // Compute gradient variance across layers
    if (num_layers == 0) return;

    float mean = 0.0f;
    for (uint32_t i = 0; i < num_layers; i++)
        mean += grad_norms[i];
    mean /= (float)num_layers;

    float var = 0.0f;
    for (uint32_t i = 0; i < num_layers; i++) {
        float d = grad_norms[i] - mean;
        var += d * d;
    }
    var /= (float)num_layers;

    // EMA of variance and mean
    const float alpha = 0.01f;   // Fast-tracking phase signal
    state->phase.grad_var_ema  = (1.0f - alpha) * state->phase.grad_var_ema
                                 + alpha * var;
    state->phase.grad_mean_ema = (1.0f - alpha) * state->phase.grad_mean_ema
                                 + alpha * mean;

    // Phase transitions:
    // WARMUP → CONVERGENCE: variance drops below 10% of mean squared
    // CONVERGENCE → FINETUNE: mean gradient norm drops below 1% of initial
    uint8_t prev_phase = state->phase.detected_phase;

    if (prev_phase == 0) {
        // Exit warmup when gradient variance is low (condensation complete)
        float cv = state->phase.grad_mean_ema > 0.0f
            ? sqrtf(state->phase.grad_var_ema) / state->phase.grad_mean_ema
            : 1.0f;
        if (cv < 0.15f && global_step >= state->cfg.warmup_steps) {
            state->phase.detected_phase = 1;  // CONVERGENCE
            state->phase.phase_entry_step = global_step;
            fprintf(stderr,
                "[CIPHER L3.9] Phase: WARMUP → CONVERGENCE at step %u "
                "(grad CV=%.3f). Substitution ENABLED.\n",
                global_step, cv);
        }
    } else if (prev_phase == 1) {
        // Enter finetune when grad mean is very small and stable
        float cv = state->phase.grad_mean_ema > 1e-6f
            ? sqrtf(state->phase.grad_var_ema) / state->phase.grad_mean_ema
            : 0.0f;
        if (cv < 0.05f && state->phase.grad_mean_ema < 0.001f) {
            state->phase.detected_phase = 2;  // FINETUNE
            state->phase.phase_entry_step = global_step;
            fprintf(stderr,
                "[CIPHER L3.9] Phase: CONVERGENCE → FINETUNE at step %u\n",
                global_step);
        }
    }

    // Sync detected phase to liquid state
    if (state->liquid && state->liquid->initialized) {
        state->liquid->device->phase = state->phase.detected_phase;
    }
}

// ---------------------------------------------------------------------------
// L3.7: EMA gradient monitor — per-layer divergence detection
// ---------------------------------------------------------------------------

static void update_ema_monitor(CipherOracleState* state,
                               const float*        grad_norms,
                               uint32_t            num_layers)
{
    uint32_t n = num_layers < CIPHER_ORACLE_MAX_LAYERS
        ? num_layers : CIPHER_ORACLE_MAX_LAYERS;

    state->ema.steps_since_update = 0;
    const float kappa = state->cfg.ema_kappa;

    for (uint32_t i = 0; i < n; i++) {
        if (state->ema.permanently_demoted[i]) continue;

        float g = grad_norms[i];

        // Update per-layer EMA
        state->ema.ema[i] = kappa * state->ema.ema[i] + (1.0f - kappa) * g;

        // Establish baseline after ema_baseline_steps
        if (!state->ema.baseline_set[i]) {
            if (state->phase.step_count >= state->cfg.ema_baseline_steps) {
                state->ema.baseline[i]     = state->ema.ema[i];
                // sigma = max deviation seen so far (initialised to 10% of baseline)
                state->ema.sigma[i]        = state->ema.baseline[i] * 0.1f + 1e-4f;
                state->ema.baseline_set[i] = true;
            }
            continue;
        }

        // Update running sigma — tracks typical deviation from baseline
        float dev = fabsf(g - state->ema.baseline[i]);
        state->ema.sigma[i] = 0.99f * state->ema.sigma[i] + 0.01f * dev;

        // Divergence: current raw gradient > (baseline + divergence_sigma * sigma)
        // AND gradient is more than 5× the EMA baseline (filters early noise)
        float threshold = state->ema.baseline[i]
                          + state->cfg.divergence_sigma * state->ema.sigma[i];
        bool spike = (g > threshold) && (g > 5.0f * state->ema.baseline[i] + 1e-4f);

        if (spike) {
            state->ema.permanently_demoted[i] = true;
            state->ema.demotion_count++;

            if (state->liquid && state->liquid->initialized)
                state->liquid->device->layer[i].perm_passthrough = 1;
            cipher_struct_override_layer(i, true);

            fprintf(stderr,
                "[CIPHER L3.7] DIVERGENCE: Layer %u demoted. "
                "grad=%.4f  baseline=%.4f  threshold=%.4f\n",
                i, g, state->ema.baseline[i], threshold);
        }
    }
}

// ---------------------------------------------------------------------------
// cipher_oracle_update_gradients — called every 100 training steps
// ---------------------------------------------------------------------------

void cipher_oracle_update_gradients(CipherOracleState* state,
                                    const float*        grad_norms,
                                    uint32_t            num_layers,
                                    uint32_t            global_step)
{
    if (!state->initialized) return;

    // L3.9: Phase detection (first — phase gates the EMA monitor)
    update_phase_detector(state, grad_norms, num_layers, global_step);

    // L3.7: EMA divergence monitor (only during convergence/finetune)
    if (state->phase.detected_phase >= 1)
        update_ema_monitor(state, grad_norms, num_layers);

    // Sync global gradient EMA to liquid state
    if (state->liquid && state->liquid->initialized && num_layers > 0) {
        float global_norm = 0.0f;
        for (uint32_t i = 0; i < num_layers; i++)
            global_norm += grad_norms[i];
        global_norm /= (float)num_layers;
        cipher_liquid_update_grad_ema(state->liquid, global_norm);
    }
}

// ---------------------------------------------------------------------------
// cipher_oracle_decide — main gate, called on every kernel dispatch
// ---------------------------------------------------------------------------

CipherOracleResult cipher_oracle_decide(CipherOracleState*       state,
                                        const CipherOracleQuery* q)
{
    static const CipherOracleResult PERMIT = { CIPHER_ORACLE_PERMIT, "ok" };

    if (!state->initialized)
        return (CipherOracleResult){ CIPHER_ORACLE_DENY, "oracle-not-init" };

    state->total_decisions++;

    // --- Gate 1: Phase (L3.9) ---
    if (state->phase.detected_phase == 0) {
        state->denied_warmup++;
        return (CipherOracleResult){ CIPHER_ORACLE_DENY, "warmup" };
    }

    // --- Gate 2: Minimum confidence from classifier (L3.1) ---
    if (q->confidence < state->cfg.min_confidence) {
        state->denied_low_confidence++;
        return (CipherOracleResult){ CIPHER_ORACLE_DENY, "low-confidence" };
    }

    // --- Gate 3: Structural lookup (L3.8) ---
    CipherStructContext sctx = {
        .kernel_name      = q->kernel_name,
        .layer_idx        = q->layer_idx,
        .total_layers     = q->total_layers,
        .op_class         = q->op_class,
        .training_phase   = state->phase.detected_phase,
        .is_backward      = q->is_backward,
        .is_optimizer_step= q->is_optimizer,
    };
    CipherStructLookupResult slr = cipher_struct_lookup(&sctx);
    if (slr.result == CIPHER_STRUCT_FULL_PRECISION) {
        state->denied_structural++;
        return (CipherOracleResult){ CIPHER_ORACLE_DENY, "structural-rule" };
    }

    // --- Gate 4: EMA permanent demotion (L3.7) ---
    if (q->layer_idx < CIPHER_ORACLE_MAX_LAYERS
        && state->ema.permanently_demoted[q->layer_idx]) {
        state->denied_ema_demotion++;
        return (CipherOracleResult){ CIPHER_ORACLE_DENY, "ema-demoted" };
    }

    // --- Gate 5: N≤4 rule (L3.6) ---
    if (q->layer_idx < CIPHER_ORACLE_MAX_LAYERS) {
        uint8_t cnt = state->sub_counter[q->layer_idx];
        if (cnt >= state->cfg.n_max) {
            // Fire the rule: reset counter, force this one to passthrough
            state->sub_counter[q->layer_idx] = 0;
            state->force_passthrough[q->layer_idx] = 1;
            state->denied_n4++;

            // Sync to liquid state
            if (state->liquid && state->liquid->initialized) {
                cipher_liquid_record_passthrough(state->liquid, (int)q->layer_idx);
            }
            return (CipherOracleResult){ CIPHER_ORACLE_DENY, "n4-rule" };
        }
    }

    // All gates passed — PERMIT
    state->permitted++;
    return PERMIT;
}

// ---------------------------------------------------------------------------
// Feedback: substitution executed
// ---------------------------------------------------------------------------

void cipher_oracle_record_substitution(CipherOracleState* state,
                                       uint32_t layer_idx)
{
    if (layer_idx < CIPHER_ORACLE_MAX_LAYERS) {
        state->sub_counter[layer_idx]++;
        state->force_passthrough[layer_idx] = 0;
        if (state->liquid && state->liquid->initialized)
            cipher_liquid_record_substitution(state->liquid, (int)layer_idx);
    }
}

// Feedback: full precision used — reset counter
void cipher_oracle_record_passthrough(CipherOracleState* state,
                                      uint32_t layer_idx)
{
    if (layer_idx < CIPHER_ORACLE_MAX_LAYERS) {
        state->sub_counter[layer_idx] = 0;
        state->force_passthrough[layer_idx] = 0;
        if (state->liquid && state->liquid->initialized)
            cipher_liquid_record_passthrough(state->liquid, (int)layer_idx);
    }
}

// ---------------------------------------------------------------------------
// Manual phase override
// ---------------------------------------------------------------------------

void cipher_oracle_set_phase(CipherOracleState* state, uint8_t phase) {
    state->phase.detected_phase   = phase;
    state->phase.manually_overridden = true;
    const char* names[] = {"WARMUP", "CONVERGENCE", "FINETUNE"};
    fprintf(stderr, "[CIPHER L3.9] Phase manually set to: %s\n",
            phase < 3 ? names[phase] : "UNKNOWN");
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

void cipher_oracle_report(const CipherOracleState* state) {
    const char* phases[] = {"WARMUP", "CONVERGENCE", "FINETUNE"};
    uint64_t denied = state->total_decisions - state->permitted;
    fprintf(stderr,
        "[CIPHER ORACLE] Decision Report\n"
        "  Phase:          %s (step %u)\n"
        "  Total decisions:%lu\n"
        "  Permitted:      %lu  (%.1f%%)\n"
        "  Denied total:   %lu  (%.1f%%)\n"
        "    warmup:       %lu\n"
        "    structural:   %lu\n"
        "    N≤4 rule:     %lu\n"
        "    EMA demotion: %lu\n"
        "    low confidence:%lu\n"
        "  Layers demoted: %u\n",
        phases[state->phase.detected_phase], state->phase.step_count,
        state->total_decisions,
        state->permitted,
        state->total_decisions > 0
            ? (double)state->permitted * 100.0 / state->total_decisions : 0.0,
        denied,
        state->total_decisions > 0
            ? (double)denied * 100.0 / state->total_decisions : 0.0,
        state->denied_warmup,
        state->denied_structural,
        state->denied_n4,
        state->denied_ema_demotion,
        state->denied_low_confidence,
        state->ema.demotion_count);
}
