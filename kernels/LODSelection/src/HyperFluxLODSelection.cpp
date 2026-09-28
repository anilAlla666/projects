/*
 * HYPERFLUX LODSELECTION - FIXED IMPLEMENTATION
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 * Fix: proj.1 is LayerNorm not Linear; added output heads
 */
#include "HGLODSelection.h"
#include <cmath>
#include <cstring>
#include "weights.inc"

static constexpr int IN = 16;
static constexpr int H = 192;
static constexpr int OUT = 3;

alignas(64) static float h1[H], h2[H], res[H];
alignas(64) static float head_buf_1[1];
alignas(64) static float head_buf_5[5];
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
    mm_gelu(HyperFlux::LODSelection::Weights::proj_0_weight, input, HyperFlux::LODSelection::Weights::proj_0_bias, h1, H, IN);

    // FIX: proj.1 is LayerNorm, NOT Linear
    ln(h1, HyperFlux::LODSelection::Weights::proj_1_weight, HyperFlux::LODSelection::Weights::proj_1_bias, H);

    // ResBlock 0
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_0_net_0_weight, h1, HyperFlux::LODSelection::Weights::blocks_0_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_0_net_1_weight, HyperFlux::LODSelection::Weights::blocks_0_net_1_bias, H);
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_0_net_3_weight, h2, HyperFlux::LODSelection::Weights::blocks_0_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_0_net_4_weight, HyperFlux::LODSelection::Weights::blocks_0_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 1
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_1_net_0_weight, h1, HyperFlux::LODSelection::Weights::blocks_1_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_1_net_1_weight, HyperFlux::LODSelection::Weights::blocks_1_net_1_bias, H);
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_1_net_3_weight, h2, HyperFlux::LODSelection::Weights::blocks_1_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_1_net_4_weight, HyperFlux::LODSelection::Weights::blocks_1_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 2
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_2_net_0_weight, h1, HyperFlux::LODSelection::Weights::blocks_2_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_2_net_1_weight, HyperFlux::LODSelection::Weights::blocks_2_net_1_bias, H);
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_2_net_3_weight, h2, HyperFlux::LODSelection::Weights::blocks_2_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_2_net_4_weight, HyperFlux::LODSelection::Weights::blocks_2_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 3
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_3_net_0_weight, h1, HyperFlux::LODSelection::Weights::blocks_3_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_3_net_1_weight, HyperFlux::LODSelection::Weights::blocks_3_net_1_bias, H);
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_3_net_3_weight, h2, HyperFlux::LODSelection::Weights::blocks_3_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_3_net_4_weight, HyperFlux::LODSelection::Weights::blocks_3_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 4
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_4_net_0_weight, h1, HyperFlux::LODSelection::Weights::blocks_4_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_4_net_1_weight, HyperFlux::LODSelection::Weights::blocks_4_net_1_bias, H);
    mm_gelu(HyperFlux::LODSelection::Weights::blocks_4_net_3_weight, h2, HyperFlux::LODSelection::Weights::blocks_4_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::LODSelection::Weights::blocks_4_net_4_weight, HyperFlux::LODSelection::Weights::blocks_4_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // Output heads
    { // LOD head (5-class)
        float lod_logits[5];
        mm_gelu(HyperFlux::LODSelection::Weights::lod_head_0_weight, h1, HyperFlux::LODSelection::Weights::lod_head_0_bias, head_buf_64, 64, H);
        mm(HyperFlux::LODSelection::Weights::lod_head_2_weight, head_buf_64, HyperFlux::LODSelection::Weights::lod_head_2_bias, lod_logits, 5, 64);
        output[0] = 0; float best = lod_logits[0];
        for (int i = 1; i < 5; i++) { if (lod_logits[i] > best) { best = lod_logits[i]; output[0] = (float)i; } }
    }
    mm_gelu(HyperFlux::LODSelection::Weights::blend_head_0_weight, h1, HyperFlux::LODSelection::Weights::blend_head_0_bias, head_buf_32, 32, H);
    mm(HyperFlux::LODSelection::Weights::blend_head_2_weight, head_buf_32, HyperFlux::LODSelection::Weights::blend_head_2_bias, &output[1], 1, 32);
    mm_gelu(HyperFlux::LODSelection::Weights::cull_head_0_weight, h1, HyperFlux::LODSelection::Weights::cull_head_0_bias, head_buf_32, 32, H);
    mm(HyperFlux::LODSelection::Weights::cull_head_2_weight, head_buf_32, HyperFlux::LODSelection::Weights::cull_head_2_bias, &output[2], 1, 32);
}

static void encode_input(const HG_CameraState* cam, const HG_ObjectInfo* obj, float* input) {
    float dx = obj->posX - cam->cameraPosX;
    float dy = obj->posY - cam->cameraPosY;
    float dz = obj->posZ - cam->cameraPosZ;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float ndot = (dx*cam->cameraForwardX + dy*cam->cameraForwardY + dz*cam->cameraForwardZ) / (dist + 1e-6f);
    float screenSize = obj->boundingRadius / (dist + 1e-6f);
    
    input[0] = dx / 1000.0f;
    input[1] = dy / 1000.0f;
    input[2] = dz / 1000.0f;
    input[3] = dist / 1000.0f;
    input[4] = ndot;
    input[5] = screenSize;
    input[6] = obj->boundingRadius / 100.0f;
    input[7] = cam->cameraFOV / 120.0f;
    input[8] = cam->cameraVelX / 100.0f;
    input[9] = cam->cameraVelY / 100.0f;
    input[10] = cam->cameraVelZ / 100.0f;
    input[11] = (float)obj->importance / 4.0f;
    input[12] = logf(dist + 1.0f) / 10.0f;
    input[13] = (float)(obj->importance == IMPORTANCE_HERO);
    input[14] = (float)(obj->importance == IMPORTANCE_CRITICAL);
    input[15] = screenSize * cam->cameraFOV / 90.0f;
}

int HG_LOD_Init(void) { g_init = 1; return 0; }
void HG_LOD_Shutdown(void) { g_init = 0; }

int HG_LOD_Select(const HG_CameraState* camera, const HG_ObjectInfo* obj, HG_LODResult* result) {
    float input[IN], output[OUT];
    encode_input(camera, obj, input);
    forward(input, output);
    result->lodLevel = (int)(output[0] + 0.5f);
    if (result->lodLevel < 0) result->lodLevel = 0;
    if (result->lodLevel > 4) result->lodLevel = 4;
    result->blendFactor = 1.0f / (1.0f + expf(-output[1]));
    result->shouldCull = output[2] > 0.0f ? 1 : 0;
    return 0;
}

int HG_LOD_SelectBatch(const HG_CameraState* camera, const HG_ObjectInfo* objects,
                        HG_LODResult* results, int count) {
    for (int i = 0; i < count; i++) HG_LOD_Select(camera, &objects[i], &results[i]);
    return 0;
}