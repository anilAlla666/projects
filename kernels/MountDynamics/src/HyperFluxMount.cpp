/*
 * ╔══════════════════════════════════════════════════════════════════════════════════╗
 * ║              HYPERFLUX MOUNT DYNAMICS - IMPLEMENTATION                           ║
 * ║                                                                                  ║
 * ║  ⚠️  CONFIDENTIAL - DO NOT DISTRIBUTE SOURCE CODE                                ║
 * ║  This file is compiled into the static library.                                  ║
 * ║                                                                                  ║
 * ║  Architecture: Compact-192                                                       ║
 * ║  Input(20) → Dense(192) → LN → SiLU → Dense(192) → LN → SiLU → Heads → Out(3)   ║
 * ║                                                                                  ║
 * ║  Copyright (c) 2026 HyperFlux Neural Dynamics. All Rights Reserved.              ║
 * ╚══════════════════════════════════════════════════════════════════════════════════╝
 */

#include "HyperFluxMount.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// Include weights
#include "weights_data.inc"

// ════════════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ════════════════════════════════════════════════════════════════════════════════════

#define INPUT_DIM 20
#define HIDDEN_DIM 192
#define H_HEAD_DIM 48
#define V_HEAD_DIM 24
#define OUTPUT_DIM 3

static const float MOUNT_PARAMS[4][5] = {
    {15.0f, 8.0f, 12.0f, 2.5f, 6.0f},   // Horse
    {20.0f, 12.0f, 15.0f, 4.0f, 8.0f},  // Panther
    {12.0f, 5.0f, 8.0f, 1.8f, 4.0f},    // Bear
    {25.0f, 10.0f, 12.0f, 3.0f, 12.0f}, // Gryphon
};

// ════════════════════════════════════════════════════════════════════════════════════
// SIMD DETECTION
// ════════════════════════════════════════════════════════════════════════════════════

#if defined(__AVX2__)
    #include <immintrin.h>
    #define USE_AVX2 1
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
    #define USE_SSE4 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define USE_NEON 1
#endif

// ════════════════════════════════════════════════════════════════════════════════════
// MATH FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════════

static inline float fast_exp(float x) {
    // Clamp to prevent overflow
    x = fmaxf(-88.0f, fminf(88.0f, x));
    return expf(x);
}

static inline float silu(float x) {
    return x / (1.0f + fast_exp(-x));
}

// Matrix-vector multiply: out = mat * vec + bias
static void matmul(const float* mat, const float* vec, const float* bias,
                   float* out, int rows, int cols) {
#ifdef USE_AVX2
    for (int i = 0; i < rows; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        
        for (; j + 8 <= cols; j += 8) {
            __m256 m = _mm256_loadu_ps(&mat[i * cols + j]);
            __m256 v = _mm256_loadu_ps(&vec[j]);
            sum = _mm256_fmadd_ps(m, v, sum);
        }
        
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);
        
        // Remainder
        for (; j < cols; j++) {
            result += mat[i * cols + j] * vec[j];
        }
        
        out[i] = result + bias[i];
    }
#else
    // Scalar fallback
    for (int i = 0; i < rows; i++) {
        float sum = bias[i];
        for (int j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * vec[j];
        }
        out[i] = sum;
    }
#endif
}

// LayerNorm: out = gamma * (x - mean) / sqrt(var + eps) + beta
static void layernorm(float* data, const float* gamma, const float* beta, int size) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;
    
    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    // Normalize
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < size; i++) {
        data[i] = gamma[i] * (data[i] - mean) * inv_std + beta[i];
    }
}

// Apply SiLU activation in-place
static void apply_silu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = silu(data[i]);
    }
}

// ════════════════════════════════════════════════════════════════════════════════════
// NEURAL NETWORK FORWARD PASS
// ════════════════════════════════════════════════════════════════════════════════════

static int g_initialized = 0;

// Buffers for intermediate activations
static float g_hidden1[HIDDEN_DIM];
static float g_hidden2[HIDDEN_DIM];
static float g_h_hidden[H_HEAD_DIM];
static float g_v_hidden[V_HEAD_DIM];
static float g_h_out[2];
static float g_v_out[1];

static void forward(const float* input, float* output) {
    // Layer 1: fc1 -> ln1 -> silu
    matmul(fc1_weight, input, fc1_bias, g_hidden1, HIDDEN_DIM, INPUT_DIM);
    layernorm(g_hidden1, ln1_gamma, ln1_beta, HIDDEN_DIM);
    apply_silu(g_hidden1, HIDDEN_DIM);
    
    // Layer 2: fc2 -> ln2 -> silu
    matmul(fc2_weight, g_hidden1, fc2_bias, g_hidden2, HIDDEN_DIM, HIDDEN_DIM);
    layernorm(g_hidden2, ln2_gamma, ln2_beta, HIDDEN_DIM);
    apply_silu(g_hidden2, HIDDEN_DIM);
    
    // Horizontal head: h_fc -> h_ln -> silu -> h_out
    matmul(h_fc_weight, g_hidden2, h_fc_bias, g_h_hidden, H_HEAD_DIM, HIDDEN_DIM);
    layernorm(g_h_hidden, h_ln_gamma, h_ln_beta, H_HEAD_DIM);
    apply_silu(g_h_hidden, H_HEAD_DIM);
    matmul(h_out_weight, g_h_hidden, h_out_bias, g_h_out, 2, H_HEAD_DIM);
    
    // Vertical head: v_fc -> v_ln -> silu -> v_out
    matmul(v_fc_weight, g_hidden2, v_fc_bias, g_v_hidden, V_HEAD_DIM, HIDDEN_DIM);
    layernorm(g_v_hidden, v_ln_gamma, v_ln_beta, V_HEAD_DIM);
    apply_silu(g_v_hidden, V_HEAD_DIM);
    matmul(v_out_weight, g_v_hidden, v_out_bias, g_v_out, 1, V_HEAD_DIM);
    
    // Combine outputs: [dvx, dvy, dvz]
    output[0] = g_h_out[0];  // dvx
    output[1] = g_v_out[0];  // dvy
    output[2] = g_h_out[1];  // dvz
}

// ════════════════════════════════════════════════════════════════════════════════════
// INPUT PREPROCESSING
// ════════════════════════════════════════════════════════════════════════════════════

static void prepare_input(const HyperFluxMountInput* input, float* features) {
    int idx = (int)input->type;
    if (idx < 0 || idx > 3) idx = 0;
    
    const float* params = MOUNT_PARAMS[idx];
    float maxSpeed = params[0];
    float turnRate = params[3];
    
    // Mount one-hot encoding
    features[0] = (idx == 0) ? 1.0f : 0.0f;
    features[1] = (idx == 1) ? 1.0f : 0.0f;
    features[2] = (idx == 2) ? 1.0f : 0.0f;
    features[3] = (idx == 3) ? 1.0f : 0.0f;
    
    // Current velocity
    features[4] = input->velocityX;
    features[5] = input->velocityY;
    features[6] = input->velocityZ;
    features[7] = input->currentSpeed;
    
    // Controls
    features[8] = input->throttle;
    features[9] = input->steering;
    features[10] = input->jumpInput;
    
    // Terrain
    features[11] = input->terrainSlope;
    features[12] = input->terrainFacing;
    
    // State
    features[13] = input->isGrounded ? 1.0f : 0.0f;
    features[14] = input->stamina;
    
    // Physics-derived features
    float speedRatio = input->currentSpeed / (maxSpeed + 1e-6f);
    float slopeFactor = 1.0f - input->terrainSlope * input->terrainFacing * 0.5f;
    slopeFactor = fmaxf(0.5f, fminf(1.5f, slopeFactor));
    float targetSpeed = input->throttle * maxSpeed * slopeFactor;
    float targetSpeedRatio = targetSpeed / (maxSpeed + 1e-6f);
    float speedError = (targetSpeed - input->currentSpeed) / (maxSpeed + 1e-6f);
    int canJump = input->isGrounded && (input->stamina > 0.15f) && (input->jumpInput > 0.5f);
    float effectiveTurnRate = turnRate * (1.0f - speedRatio * 0.3f);
    
    features[15] = speedRatio;
    features[16] = targetSpeedRatio;
    features[17] = speedError;
    features[18] = canJump ? 1.0f : 0.0f;
    features[19] = effectiveTurnRate / 5.0f;
}

// ════════════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ════════════════════════════════════════════════════════════════════════════════════

int HyperFlux_Initialize(void) {
    g_initialized = 1;
    return 1;
}

void HyperFlux_Shutdown(void) {
    g_initialized = 0;
}

int HyperFlux_IsInitialized(void) {
    return g_initialized;
}

int HyperFlux_Predict(const HyperFluxMountInput* input, HyperFluxMountOutput* output) {
    if (!g_initialized || !input || !output) {
        return 0;
    }
    
    float features[INPUT_DIM];
    float result[OUTPUT_DIM];
    
    prepare_input(input, features);
    forward(features, result);
    
    output->deltaVelocityX = result[0];
    output->deltaVelocityY = result[1];
    output->deltaVelocityZ = result[2];
    
    return 1;
}

int HyperFlux_PredictBatch(const HyperFluxMountInput* inputs,
                           HyperFluxMountOutput* outputs,
                           int count) {
    if (!g_initialized || !inputs || !outputs || count <= 0) {
        return 0;
    }
    
    int maxCount = (count > 64) ? 64 : count;
    
    for (int i = 0; i < maxCount; i++) {
        HyperFlux_Predict(&inputs[i], &outputs[i]);
    }
    
    return maxCount;
}

const char* HyperFlux_GetVersion(void) {
    return "1.0.0";
}

void HyperFlux_GetMountParams(HyperFluxMountType type,
                              float* maxSpeed, float* accel,
                              float* turnRate, float* jumpForce) {
    int idx = (int)type;
    if (idx < 0 || idx > 3) idx = 0;
    
    const float* params = MOUNT_PARAMS[idx];
    if (maxSpeed) *maxSpeed = params[0];
    if (accel) *accel = params[1];
    if (turnRate) *turnRate = params[3];
    if (jumpForce) *jumpForce = params[4];
}
