/*
 * HYPERFLUX LUMENGI - FIXED IMPLEMENTATION
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 * Fix: proj.1 is LayerNorm not Linear; added output heads
 */
#include "HGLumenGI.h"
#include <cmath>
#include <cstring>
#include "weights.inc"

static constexpr int IN = 22;
static constexpr int H = 256;
static constexpr int OUT = 6;

alignas(64) static float h1[H], h2[H], res[H];
alignas(64) static float head_buf_1[1];
alignas(64) static float head_buf_3[3];
alignas(64) static float head_buf_32[32];
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
    mm_gelu(HyperFlux::LumenGI::Weights::proj_0_weight, input, HyperFlux::LumenGI::Weights::proj_0_bias, h1, H, IN);

    // FIX: proj.1 is LayerNorm, NOT Linear
    ln(h1, HyperFlux::LumenGI::Weights::proj_1_weight, HyperFlux::LumenGI::Weights::proj_1_bias, H);

    // ResBlock 0
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_0_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_0_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_0_net_1_weight, HyperFlux::LumenGI::Weights::blocks_0_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_0_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_0_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_0_net_4_weight, HyperFlux::LumenGI::Weights::blocks_0_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 1
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_1_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_1_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_1_net_1_weight, HyperFlux::LumenGI::Weights::blocks_1_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_1_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_1_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_1_net_4_weight, HyperFlux::LumenGI::Weights::blocks_1_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 2
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_2_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_2_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_2_net_1_weight, HyperFlux::LumenGI::Weights::blocks_2_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_2_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_2_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_2_net_4_weight, HyperFlux::LumenGI::Weights::blocks_2_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 3
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_3_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_3_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_3_net_1_weight, HyperFlux::LumenGI::Weights::blocks_3_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_3_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_3_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_3_net_4_weight, HyperFlux::LumenGI::Weights::blocks_3_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 4
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_4_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_4_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_4_net_1_weight, HyperFlux::LumenGI::Weights::blocks_4_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_4_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_4_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_4_net_4_weight, HyperFlux::LumenGI::Weights::blocks_4_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 5
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_5_net_0_weight, h1, HyperFlux::LumenGI::Weights::blocks_5_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_5_net_1_weight, HyperFlux::LumenGI::Weights::blocks_5_net_1_bias, H);
    mm_gelu(HyperFlux::LumenGI::Weights::blocks_5_net_3_weight, h2, HyperFlux::LumenGI::Weights::blocks_5_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LumenGI::Weights::blocks_5_net_4_weight, HyperFlux::LumenGI::Weights::blocks_5_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // Output heads
    mm_gelu(HyperFlux::LumenGI::Weights::color_head_0_weight, h1, HyperFlux::LumenGI::Weights::color_head_0_bias, head_buf_64, 64, H);
    mm(HyperFlux::LumenGI::Weights::color_head_2_weight, head_buf_64, HyperFlux::LumenGI::Weights::color_head_2_bias, &output[0], 3, 64);
    mm_gelu(HyperFlux::LumenGI::Weights::intensity_head_0_weight, h1, HyperFlux::LumenGI::Weights::intensity_head_0_bias, head_buf_32, 32, H);
    mm(HyperFlux::LumenGI::Weights::intensity_head_2_weight, head_buf_32, HyperFlux::LumenGI::Weights::intensity_head_2_bias, &output[3], 1, 32);
    mm_gelu(HyperFlux::LumenGI::Weights::ao_head_0_weight, h1, HyperFlux::LumenGI::Weights::ao_head_0_bias, head_buf_32, 32, H);
    mm(HyperFlux::LumenGI::Weights::ao_head_2_weight, head_buf_32, HyperFlux::LumenGI::Weights::ao_head_2_bias, &output[4], 1, 32);
    mm_gelu(HyperFlux::LumenGI::Weights::skyvis_head_0_weight, h1, HyperFlux::LumenGI::Weights::skyvis_head_0_bias, head_buf_32, 32, H);
    mm(HyperFlux::LumenGI::Weights::skyvis_head_2_weight, head_buf_32, HyperFlux::LumenGI::Weights::skyvis_head_2_bias, &output[5], 1, 32);
}

static void encode_input(const HG_LumenInput* in, float* input) {
    float ndotl = in->normalX*in->sunDirX + in->normalY*in->sunDirY + in->normalZ*in->sunDirZ;
    
    input[0] = in->posX / 1000.0f;
    input[1] = in->posY / 1000.0f;
    input[2] = in->posZ / 1000.0f;
    input[3] = in->normalX;
    input[4] = in->normalY;
    input[5] = in->normalZ;
    input[6] = in->sunDirX;
    input[7] = in->sunDirY;
    input[8] = in->sunDirZ;
    input[9] = in->sunIntensity;
    input[10] = in->timeOfDay;
    input[11] = (float)in->isIndoor;
    input[12] = (float)in->surfaceType / 5.0f;
    input[13] = ndotl;
    input[14] = sinf(in->timeOfDay * 6.2831853f);
    input[15] = cosf(in->timeOfDay * 6.2831853f);
    input[16] = (float)(in->surfaceType == SURFACE_METAL);
    input[17] = (float)(in->surfaceType == SURFACE_WATER);
    input[18] = (float)(in->surfaceType == SURFACE_STONE);
    input[19] = in->sunIntensity * (1.0f - (float)in->isIndoor);
    input[20] = in->posY / 100.0f;
    input[21] = ndotl * in->sunIntensity;
}

int HG_Lumen_Init(void) { g_init = 1; return 0; }
void HG_Lumen_Shutdown(void) { g_init = 0; }

int HG_Lumen_Compute(const HG_LumenInput* in, HG_LumenOutput* out) {
    float input[IN], output[OUT];
    encode_input(in, input);
    forward(input, output);
    out->indirectR = 1.0f / (1.0f + expf(-output[0]));
    out->indirectG = 1.0f / (1.0f + expf(-output[1]));
    out->indirectB = 1.0f / (1.0f + expf(-output[2]));
    out->indirectIntensity = 2.0f / (1.0f + expf(-output[3]));
    out->ambientOcclusion = 1.0f / (1.0f + expf(-output[4]));
    out->skyVisibility = 1.0f / (1.0f + expf(-output[5]));
    return 0;
}

int HG_Lumen_ComputeBatch(const HG_LumenInput* inputs, HG_LumenOutput* outputs, int count) {
    for (int i = 0; i < count; i++) HG_Lumen_Compute(&inputs[i], &outputs[i]);
    return 0;
}