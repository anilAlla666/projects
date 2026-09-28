// =============================================================================
// CIPHER — L3.10: CfC LNN Implementation
// cipher_lnn.cpp
// =============================================================================

#ifdef CIPHER_CPU_STUB
#  include "may13/cipher_stubs.h"
#endif

#include "may13/cipher_lnn.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static uint64_t now_ns_lnn(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

static inline float lnn_tanh(float x) { return tanhf(x); }
static inline float lnn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float lnn_softplus(float x) {
    // softplus(x) = log(1 + e^x), numerically stable for large x
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

// Softmax in-place over n elements
static void lnn_softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= (s + 1e-8f);
}

// ---------------------------------------------------------------------------
// CfC forward pass — core math
//
// Given concatenated input [x; h] ∈ ℝ^{I+H}:
//   τ = softplus(W_τ · [x;h] + b_τ)   ∈ ℝ^H  (positive time constant)
//   g = tanh(W_g · [x;h] + b_g)       ∈ ℝ^H  (target state)
//   h' = g·(1 - e^{-τΔt}) + h·e^{-τΔt}       (closed-form ODE solution)
//
// This is the exact solution to the liquid state ODE:
//   dh/dt = -τ(h,x)·(h - g(h,x))
// at time t+Δt, which equals the CfC formulation.
// ---------------------------------------------------------------------------

static void cfc_step(const CipherLnnWeights* W,
                     const float*            x,      // I-dim input
                     float*                  h,      // H-dim hidden (in-place update)
                     float                   dt)
{
    const int I = CIPHER_LNN_INPUT_DIM;
    const int H = CIPHER_LNN_HIDDEN_DIM;

    // Concatenate [x; h] → xh (I+H dim)
    float xh[CIPHER_LNN_CONCAT_DIM];
    memcpy(xh,     x, I * sizeof(float));
    memcpy(xh + I, h, H * sizeof(float));

    // Compute τ and g
    float tau[CIPHER_LNN_HIDDEN_DIM];
    float g[CIPHER_LNN_HIDDEN_DIM];

    for (int i = 0; i < H; i++) {
        float s_tau = W->b_tau[i];
        float s_g   = W->b_g[i];
        for (int j = 0; j < I + H; j++) {
            s_tau += W->W_tau[i][j] * xh[j];
            s_g   += W->W_g[i][j]   * xh[j];
        }
        tau[i] = lnn_softplus(s_tau);  // τ > 0
        g[i]   = lnn_tanh(s_g);       // g ∈ (-1,1)
    }

    // Closed-form ODE: h' = g·(1-e^{-τΔt}) + h·e^{-τΔt}
    for (int i = 0; i < H; i++) {
        float decay = expf(-tau[i] * dt);
        h[i] = g[i] * (1.0f - decay) + h[i] * decay;
    }
}

// ---------------------------------------------------------------------------
// Analytical weight initialization
//
// Encodes Layer 3 rules in W_g and W_tau:
//
// W_g: substitute bias for substitutable op classes
//   - Units 0-6: respond to op class one-hot
//   - Unit 0 (GEMM-detector): W_g[0][0] = +4.0 → g→1 when GEMM
//   - Unit 1 (ATTN-passthrough): W_g[1][1] = +4.0, but output head
//     maps this to negative substitute logit
//   - Unit 7-8: respond to N≤4 counter (high counter → passthrough)
//   - Unit 9: responds to warmup phase (warmup → passthrough)
//
// W_tau: adaptive time constant
//   - During warmup: high tau → slow adaptation → conservative
//   - During convergence: low tau → fast adaptation → aggressive
//
// W_out: output projection
//   - out[0] (substitute logit): positive from GEMM/EW/REDUCE units,
//     negative from ATTN/last-layer/warmup units
//   - out[1-7] (recipe type): follows op class signal
//   - out[8] (confidence): follows magnitude of substitute signal
// ---------------------------------------------------------------------------

static void init_weights_analytical(CipherLnnWeights* W) {
    memset(W, 0, sizeof(*W));

    const int I = CIPHER_LNN_INPUT_DIM;

    // ── W_g: target state gates ─────────────────────────────────────────
    // Units 0-6: op class detectors (one-hot at input[0-6])
    for (int cls = 0; cls < 7; cls++) {
        // Strong positive response to their own op class
        W->W_g[cls][cls] = 3.5f;
        W->b_g[cls] = -1.5f;  // Threshold: only fires when op class present
    }

    // Unit 7: substitution suppressor (fires on N≤4 violation)
    // input[12] = substitution_counter / 4.0, so =1.0 at limit
    W->W_g[7][12] = 4.0f;    // input[12] = counter/4
    W->b_g[7]     = -1.0f;

    // Unit 8: warmup suppressor (fires when training_phase == 0)
    // input[14] = phase (0=warmup, 1=convergence)
    W->W_g[8][14] = -4.0f;   // Inverted: phase=0 (warmup) → fires
    W->b_g[8]     =  2.0f;   // Bias: active when phase < 0.5

    // Unit 9: gradient divergence detector
    // input[13] = ema_gradient_norm (spikes on divergence)
    W->W_g[9][13] = 3.0f;
    W->b_g[9]     = -1.0f;

    // Units 10-15: recipe parameter estimators
    // Unit 10: GEMM compute-bound detector (high shmem)
    W->W_g[10][0]  = 2.0f;   // GEMM class
    W->W_g[10][11] = 1.5f;   // shmem ratio
    W->b_g[10]     = -2.0f;

    // Unit 11: decode/vector mode (unit 11: low grid x/y)
    W->W_g[11][0] = 2.0f;    // GEMM class
    W->W_g[11][7] = -2.0f;   // log2(gx): low = decode mode
    W->b_g[11]    = 0.5f;

    // Units 16-63: hardware trajectory response (liquid state hw fields)
    // Unit 16: SM idle response
    W->W_g[16][16] = 3.0f;   // input[16] = sm_idle_fraction
    W->b_g[16]     = -0.5f;

    // Remaining units: small positive initialisation for stability
    for (int i = 17; i < CIPHER_LNN_HIDDEN_DIM; i++) {
        W->b_g[i] = 0.01f;
    }

    // ── W_tau: time constants ────────────────────────────────────────────
    // Warmup phase → high tau (slow, conservative)
    for (int i = 0; i < CIPHER_LNN_HIDDEN_DIM; i++) {
        W->W_tau[i][14] = -1.5f;  // input[14] = phase: convergence=1 → lower tau
        W->b_tau[i]     = 1.0f;   // Default tau ≈ softplus(1) ≈ 1.31
    }
    // Op class detectors: fast time constant (τ → respond immediately)
    for (int cls = 0; cls < 7; cls++) {
        W->W_tau[cls][cls] = 0.5f;  // Present op class → slightly faster
    }

    // ── W_out: output projection ─────────────────────────────────────────
    // Output 0: substitute logit
    // Positive from: GEMM(0), EW(4), REDUCE(5), CONV(6) detectors
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][0]  = +2.5f;  // GEMM
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][2]  = +2.0f;  // EW
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][3]  = +2.0f;  // REDUCE
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][4]  = +1.5f;  // CONV
    // Negative from: ATTN(1) suppressor, N≤4(7), warmup(8), divergence(9)
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][1]  = -4.0f;  // ATTN always FP
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][7]  = -8.0f;  // N≤4 violation (must dominate)
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][8]  = -8.0f;  // Warmup (must dominate)
    W->W_out[CIPHER_LNN_OUT_SUBSTITUTE][9]  = -4.0f;  // Divergence
    W->b_out[CIPHER_LNN_OUT_SUBSTITUTE]     = -0.5f;  // Default: passthrough

    // Outputs 1-7: recipe type logits (matches op class)
    for (int cls = 0; cls < 7; cls++) {
        W->W_out[CIPHER_LNN_OUT_RECIPE_BASE + cls][cls] = 3.0f;
        W->b_out[CIPHER_LNN_OUT_RECIPE_BASE + cls] = -0.5f;
    }

    // Output 8: confidence (from magnitude of substitute signal + op class clarity)
    for (int cls = 0; cls < 7; cls++) {
        W->W_out[CIPHER_LNN_OUT_CONFIDENCE][cls] = 1.0f;
    }
    W->b_out[CIPHER_LNN_OUT_CONFIDENCE] = 0.0f;

    // Outputs 9-11: recipe params
    // Param 0 (tile log2): GEMM compute-bound unit
    W->W_out[CIPHER_LNN_OUT_PARAM_0][10] = 2.0f;
    W->b_out[CIPHER_LNN_OUT_PARAM_0]     = 4.0f;  // Default: 2^4 = 16 tile

    // Param 1 (stages): decode mode unit
    W->W_out[CIPHER_LNN_OUT_PARAM_1][11] = 1.0f;
    W->b_out[CIPHER_LNN_OUT_PARAM_1]     = 3.0f;  // Default: 3 stages

    // Param 2 (efficiency): HBM utilisation
    W->W_out[CIPHER_LNN_OUT_PARAM_2][16] = -1.5f; // Idle SMs → higher efficiency
    W->b_out[CIPHER_LNN_OUT_PARAM_2]     = 0.85f; // Default: 85% efficiency
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void cipher_lnn_init(CipherLnnState* state) {
    memset(state, 0, sizeof(*state));
    init_weights_analytical(&state->weights);
    state->delta_t      = 1.0f;
    state->initialized  = true;

    // Size in KB
    size_t weight_bytes = sizeof(CipherLnnWeights);
    fprintf(stderr,
        "[CIPHER L3.10] CfC LNN initialized.\n"
        "  Architecture: %d → %d → %d (CfC)\n"
        "  Weight init:  analytical (Layer 3 rules encoded)\n"
        "  Weight size:  %zu KB\n"
        "  Δt:           %.1f\n",
        CIPHER_LNN_INPUT_DIM, CIPHER_LNN_HIDDEN_DIM, CIPHER_LNN_OUTPUT_DIM,
        weight_bytes / 1024, state->delta_t);
}

CipherLnnInput cipher_lnn_build_input(
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid)
{
    CipherLnnInput inp = {};

    // [0-6] Op class one-hot
    if (op_class < 7) inp.x[op_class] = 1.0f;

    // [7-9] Grid dims (log2, normalised to [0,1] assuming max 2^16)
    auto safe_log2 = [](uint32_t v) -> float {
        return v > 0 ? logf((float)v) / logf(2.0f) / 16.0f : 0.0f;
    };
    inp.x[7]  = safe_log2(grid_x);
    inp.x[8]  = safe_log2(grid_y);
    inp.x[9]  = safe_log2(grid_z);

    // [10] Block size
    inp.x[10] = safe_log2(block_size);

    // [11] Shared memory ratio
    inp.x[11] = (float)shmem_bytes / 65536.0f;
    if (inp.x[11] > 1.0f) inp.x[11] = 1.0f;

    // [12-15] Liquid state fields
    if (liquid && liquid->initialized && liquid->device) {
        const CipherLiquidState* ls = liquid->device;
        inp.x[12] = (float)ls->layer[0].sub_counter / 4.0f;
        inp.x[13] = ls->global_grad_ema > 10.0f ? 1.0f
                    : ls->global_grad_ema / 10.0f;
        inp.x[14] = (float)ls->phase;          // 0=warmup, 1=conv, 2=fine
        inp.x[15] = ls->error.total_accumulated > 1.0f ? 1.0f
                    : ls->error.total_accumulated;

        // [16-19] Hardware telemetry
        inp.x[16] = ls->hw.sm_idle_fraction;
        inp.x[17] = ls->hw.hbm_bw_utilized;
        inp.x[18] = ls->hw.l2_hit_rate;
        inp.x[19] = ls->hw.nvlink_utilization;

        // [20-27] Workload rhythm (last 8 ops, one-hot each)
        for (int i = 0; i < 8; i++) {
            int idx = ((int)ls->rhythm.write_head - 1 - i + CIPHER_WORKLOAD_HIST_LEN)
                      % CIPHER_WORKLOAD_HIST_LEN;
            uint8_t cls = ls->rhythm.op_class[idx];
            inp.x[20 + i] = (float)cls / 7.0f;  // Normalised op class
        }

        // [28-31] NCCL congestion (last 4, normalised)
        float nccl_ref = ls->nccl.ema_duration_ns > 0
                         ? ls->nccl.ema_duration_ns : 50000000.0f;
        for (int i = 0; i < 4 && i < CIPHER_NCCL_HIST_LEN; i++) {
            inp.x[28 + i] = (float)ls->nccl.duration_ns[i] / nccl_ref;
            if (inp.x[28+i] > 2.0f) inp.x[28+i] = 2.0f;
        }
    } else {
        // No liquid state: phase=convergence, everything else zero
        inp.x[14] = 1.0f;
    }
    // [32-47] reserved — zeros

    return inp;
}

CipherLnnDecision cipher_lnn_forward(CipherLnnState*      state,
                                     const CipherLnnInput* input)
{
    uint64_t t0 = now_ns_lnn();
    CipherLnnDecision dec = {};

    // Save h before update (for Koopman loss)
    float h_prev[CIPHER_LNN_HIDDEN_DIM];
    memcpy(h_prev, state->h, sizeof(h_prev));

    // CfC step: updates state->h in place
    cfc_step(&state->weights, input->x, state->h, state->delta_t);

    // Output projection: y = W_out · h + b_out
    float y[CIPHER_LNN_OUTPUT_DIM];
    for (int o = 0; o < CIPHER_LNN_OUTPUT_DIM; o++) {
        y[o] = state->weights.b_out[o];
        for (int i = 0; i < CIPHER_LNN_HIDDEN_DIM; i++)
            y[o] += state->weights.W_out[o][i] * state->h[i];
    }

    // Parse outputs
    dec.substitute_logit = y[CIPHER_LNN_OUT_SUBSTITUTE];
    dec.should_substitute = (dec.substitute_logit > 0.0f);
    dec.confidence        = lnn_sigmoid(y[CIPHER_LNN_OUT_CONFIDENCE]);

    // Recipe type: argmax over recipe logits [1-7]
    float* recipe_logits = y + CIPHER_LNN_OUT_RECIPE_BASE;
    lnn_softmax(recipe_logits, 7);
    int best_recipe = 0;
    for (int i = 1; i < 7; i++)
        if (recipe_logits[i] > recipe_logits[best_recipe])
            best_recipe = i;
    dec.recipe_type = (uint8_t)best_recipe;

    // Recipe params
    dec.recipe_params[0] = y[CIPHER_LNN_OUT_PARAM_0];
    dec.recipe_params[1] = y[CIPHER_LNN_OUT_PARAM_1];
    dec.recipe_params[2] = lnn_sigmoid(y[CIPHER_LNN_OUT_PARAM_2]);

    // Timing + stats
    dec.forward_pass_ns = now_ns_lnn() - t0;
    state->forward_pass_count++;
    float n = (float)state->forward_pass_count;
    state->avg_forward_ns = state->avg_forward_ns * (n-1)/n
                           + (float)dec.forward_pass_ns / n;

    if (dec.should_substitute) state->substitutions++;
    else                        state->passthroughs++;

    if (state->forward_pass_count > 0)
        state->substitution_rate = (float)state->substitutions
                                   / (float)state->forward_pass_count;

    return dec;
}

void cipher_lnn_koopman_update(CipherLnnState* state,
                               const float*    h_before,
                               const float*    h_after,
                               float           lr)
{
    // Koopman linearity loss: L = ||K·ψ(h_before) - ψ(h_after)||²
    // Simple gradient step on W_g (which shapes h trajectories).
    // ψ(h) = h (identity observable for the hidden state).
    // K approximated by computing h_after - h_before direction.
    //
    // Gradient: ∂L/∂W_g ≈ -lr · (h_after - K·h_before) · h_before^T
    // We implement a simplified version: nudge W_g toward the direction
    // that makes CfC trajectories more linear (Koopman property).

    float diff[CIPHER_LNN_HIDDEN_DIM];
    for (int i = 0; i < CIPHER_LNN_HIDDEN_DIM; i++)
        diff[i] = h_after[i] - h_before[i];

    float loss = 0.0f;
    for (int i = 0; i < CIPHER_LNN_HIDDEN_DIM; i++)
        loss += diff[i] * diff[i];
    loss = sqrtf(loss) / CIPHER_LNN_HIDDEN_DIM;

    // EMA track Koopman loss
    state->koopman_loss_ema = 0.999f * state->koopman_loss_ema + 0.001f * loss;

    // Small gradient step: nudge b_g to make trajectories smoother
    for (int i = 0; i < CIPHER_LNN_HIDDEN_DIM; i++) {
        state->weights.b_g[i] -= lr * diff[i] * 0.01f;
        // Clip to prevent divergence
        if (state->weights.b_g[i] >  5.0f) state->weights.b_g[i] =  5.0f;
        if (state->weights.b_g[i] < -5.0f) state->weights.b_g[i] = -5.0f;
    }
}

CipherLnnDecision cipher_lnn_decide(
    CipherLnnState*             state,
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid,
    float                       confidence_threshold)
{
    CipherLnnInput inp = cipher_lnn_build_input(
        op_class, grid_x, grid_y, grid_z, block_size, shmem_bytes, liquid);
    CipherLnnDecision dec = cipher_lnn_forward(state, &inp);

    // ── Hard oracle gates (authoritative — override LNN) ────────────────
    // These mirror the Layer 3 oracle rules. The LNN can NEVER substitute
    // when any of these fire, regardless of its logit.

    // Gate 1: Confidence threshold
    if (dec.confidence < confidence_threshold) {
        dec.should_substitute = false;
        return dec;
    }

    // Gate 2: Warmup phase — substitution always blocked
    if (liquid && liquid->initialized && liquid->device) {
        const CipherLiquidState* ls = liquid->device;

        if (ls->phase == 0) {  // WARMUP
            dec.should_substitute = false;
            return dec;
        }

        // Gate 3: N≤4 rule — layer 0 as global counter proxy
        if (ls->layer[0].sub_counter >= 4) {
            dec.should_substitute = false;
            return dec;
        }

        // Gate 4: ATTN always full precision (structural rule)
        if (op_class == (uint8_t)cipher::OpClass::ATTENTION) {
            dec.should_substitute = false;
            return dec;
        }

        // Gate 5: Gradient divergence detected
        if (ls->layer[0].perm_passthrough) {
            dec.should_substitute = false;
            return dec;
        }
    }

    return dec;
}

void cipher_lnn_reset_hidden(CipherLnnState* state) {
    memset(state->h, 0, sizeof(state->h));
}

void cipher_lnn_report(const CipherLnnState* state) {
    uint64_t total = state->substitutions + state->passthroughs;
    fprintf(stderr,
        "[CIPHER L3.10] CfC LNN Report\n"
        "  Forward passes:    %lu\n"
        "  Avg latency:       %.0f ns  (target <2000ns)\n"
        "  Substitutions:     %lu  (%.1f%%)\n"
        "  Passthroughs:      %lu  (%.1f%%)\n"
        "  Koopman loss EMA:  %.4f\n",
        state->forward_pass_count,
        state->avg_forward_ns,
        state->substitutions,
        total > 0 ? (double)state->substitutions*100.0/total : 0.0,
        state->passthroughs,
        total > 0 ? (double)state->passthroughs*100.0/total : 0.0,
        state->koopman_loss_ema);
}
