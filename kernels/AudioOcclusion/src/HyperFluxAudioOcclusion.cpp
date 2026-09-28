/*
 * HYPERFLUX AUDIOOCCLUSION - FIXED IMPLEMENTATION
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 * Fix: proj.1 is LayerNorm not Linear; added output heads
 */
#include "HGAudioOcclusion.h"
#include <cmath>
#include <cstring>
#include "weights.inc"

static constexpr int IN = 26;
static constexpr int H = 256;
static constexpr int OUT = 4;

alignas(64) static float h1[H], h2[H], res[H];
alignas(64) static float head_buf_1[1];
alignas(64) static float head_buf_64[64];
static int g_init = 0;

static inline float gelu_f(float x) {
    return x * 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void mm_gelu(const float* __restrict__ W, const float* __restrict__ x,
                     const float* __restrict__ b, float* __restrict__ y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = b[i];
        const float* Wi = &W[i * cols];
        for (int j = 0; j < cols; j++) s += Wi[j] * x[j];
        y[i] = gelu_f(s);
    }
}

static void mm(const float* __restrict__ W, const float* __restrict__ x,
                const float* __restrict__ b, float* __restrict__ y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = b[i];
        const float* Wi = &W[i * cols];
        for (int j = 0; j < cols; j++) s += Wi[j] * x[j];
        y[i] = s;
    }
}

static void ln(float* x, const float* g, const float* b, int n) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float inv = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = g[i] * (x[i] - mean) * inv + b[i];
}

static void forward(const float* input, float* output) {
    // Input projection: Linear(IN→H) + GELU
    mm_gelu(HyperFlux::AudioOcclusion::Weights::proj_0_weight, input, HyperFlux::AudioOcclusion::Weights::proj_0_bias, h1, H, IN);

    // FIX: proj.1 is LayerNorm, NOT Linear
    ln(h1, HyperFlux::AudioOcclusion::Weights::proj_1_weight, HyperFlux::AudioOcclusion::Weights::proj_1_bias, H);

    // ResBlock 0
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_0_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_0_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_0_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_0_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_0_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_0_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_0_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_0_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 1
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_1_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_1_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_1_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_1_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_1_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_1_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_1_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_1_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 2
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_2_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_2_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_2_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_2_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_2_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_2_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_2_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_2_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 3
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_3_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_3_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_3_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_3_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_3_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_3_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_3_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_3_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 4
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_4_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_4_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_4_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_4_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_4_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_4_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_4_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_4_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 5
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_5_net_0_weight, h1, HyperFlux::AudioOcclusion::Weights::blocks_5_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_5_net_1_weight, HyperFlux::AudioOcclusion::Weights::blocks_5_net_1_bias, H);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::blocks_5_net_3_weight, h2, HyperFlux::AudioOcclusion::Weights::blocks_5_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::AudioOcclusion::Weights::blocks_5_net_4_weight, HyperFlux::AudioOcclusion::Weights::blocks_5_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // Output heads
    mm_gelu(HyperFlux::AudioOcclusion::Weights::attenuation_head_0_weight, h1, HyperFlux::AudioOcclusion::Weights::attenuation_head_0_bias, head_buf_64, 64, H);
    mm(HyperFlux::AudioOcclusion::Weights::attenuation_head_2_weight, head_buf_64, HyperFlux::AudioOcclusion::Weights::attenuation_head_2_bias, &output[0], 1, 64);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::lowpass_head_0_weight, h1, HyperFlux::AudioOcclusion::Weights::lowpass_head_0_bias, head_buf_64, 64, H);
    mm(HyperFlux::AudioOcclusion::Weights::lowpass_head_2_weight, head_buf_64, HyperFlux::AudioOcclusion::Weights::lowpass_head_2_bias, &output[1], 1, 64);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::reverb_head_0_weight, h1, HyperFlux::AudioOcclusion::Weights::reverb_head_0_bias, head_buf_64, 64, H);
    mm(HyperFlux::AudioOcclusion::Weights::reverb_head_2_weight, head_buf_64, HyperFlux::AudioOcclusion::Weights::reverb_head_2_bias, &output[2], 1, 64);
    mm_gelu(HyperFlux::AudioOcclusion::Weights::delay_head_0_weight, h1, HyperFlux::AudioOcclusion::Weights::delay_head_0_bias, head_buf_64, 64, H);
    mm(HyperFlux::AudioOcclusion::Weights::delay_head_2_weight, head_buf_64, HyperFlux::AudioOcclusion::Weights::delay_head_2_bias, &output[3], 1, 64);
}

static void encode_input(const HG_AudioInput* in, float* input) {
    float dx = in->sourceX - in->listenerX;
    float dy = in->sourceY - in->listenerY;
    float dz = in->sourceZ - in->listenerZ;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    
    input[0] = in->listenerX / 1000.0f;
    input[1] = in->listenerY / 1000.0f;
    input[2] = in->listenerZ / 1000.0f;
    input[3] = in->sourceX / 1000.0f;
    input[4] = in->sourceY / 1000.0f;
    input[5] = in->sourceZ / 1000.0f;
    input[6] = (float)in->soundType / 7.0f;
    input[7] = (float)in->envType / 5.0f;
    input[8] = (float)in->wallConfig / 4.0f;
    input[9] = in->wallDistance / 100.0f;
    input[10] = in->wallThickness / 10.0f;
    input[11] = dx / 1000.0f;
    input[12] = dy / 1000.0f;
    input[13] = dz / 1000.0f;
    input[14] = dist / 1000.0f;
    input[15] = dx / (dist + 1e-6f);
    input[16] = dy / (dist + 1e-6f);
    input[17] = dz / (dist + 1e-6f);
    input[18] = (float)(in->soundType == SOUND_GUNFIRE);
    input[19] = (float)(in->soundType == SOUND_EXPLOSION);
    input[20] = (float)(in->envType == ENV_CAVE_MINE);
    input[21] = (float)(in->envType == ENV_BASE_INTERIOR);
    input[22] = (float)(in->wallConfig != WALL_NONE);
    input[23] = in->wallDistance * in->wallThickness / 1000.0f;
    input[24] = dist * (float)(in->wallConfig != WALL_NONE) / 1000.0f;
    input[25] = logf(dist + 1.0f) / 10.0f;
}

int HG_Audio_Init(void) { g_init = 1; return 0; }
void HG_Audio_Shutdown(void) { g_init = 0; }

int HG_Audio_Compute(const HG_AudioInput* in, HG_AudioOutput* out) {
    float input[IN], output[OUT];
    encode_input(in, input);
    forward(input, output);
    out->attenuation = 1.0f / (1.0f + expf(-output[0]));
    out->lowpassCutoff = 200.0f + 19800.0f / (1.0f + expf(-output[1]));
    out->reverbWet = 1.0f / (1.0f + expf(-output[2]));
    out->delayMs = 100.0f / (1.0f + expf(-output[3]));
    return 0;
}

int HG_Audio_ComputeBatch(const HG_AudioInput* inputs, HG_AudioOutput* outputs, int count) {
    for (int i = 0; i < count; i++) HG_Audio_Compute(&inputs[i], &outputs[i]);
    return 0;
}