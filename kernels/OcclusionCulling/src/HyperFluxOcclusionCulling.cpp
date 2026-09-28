/*
 * HYPERFLUX OCCLUSIONCULLING - FIXED IMPLEMENTATION
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 * Fix: proj.1 is LayerNorm not Linear; added output heads
 */
#include "HGOcclusionCulling.h"
#include <cmath>
#include <cstring>
#include "weights.inc"

static constexpr int IN = 24;
static constexpr int H = 320;
static constexpr int OUT = 2;

alignas(64) static float h1[H], h2[H], res[H];
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
    mm_gelu(HyperFlux::OcclusionCulling::Weights::proj_0_weight, input, HyperFlux::OcclusionCulling::Weights::proj_0_bias, h1, H, IN);

    // FIX: proj.1 is LayerNorm, NOT Linear
    ln(h1, HyperFlux::OcclusionCulling::Weights::proj_1_weight, HyperFlux::OcclusionCulling::Weights::proj_1_bias, H);

    // ResBlock 0
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_0_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_0_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_0_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_0_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_0_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_0_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_0_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_0_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 1
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_1_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_1_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_1_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_1_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_1_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_1_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_1_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_1_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 2
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_2_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_2_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_2_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_2_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_2_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_2_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_2_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_2_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 3
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_3_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_3_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_3_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_3_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_3_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_3_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_3_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_3_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 4
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_4_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_4_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_4_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_4_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_4_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_4_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_4_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_4_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 5
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_5_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_5_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_5_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_5_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_5_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_5_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_5_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_5_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 6
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_6_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_6_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_6_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_6_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_6_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_6_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_6_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_6_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // ResBlock 7
    memcpy(res, h1, H * sizeof(float));
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_7_net_0_weight, h1, HyperFlux::OcclusionCulling::Weights::blocks_7_net_0_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_7_net_1_weight, HyperFlux::OcclusionCulling::Weights::blocks_7_net_1_bias, H);
    mm_gelu(HyperFlux::OcclusionCulling::Weights::blocks_7_net_3_weight, h2, HyperFlux::OcclusionCulling::Weights::blocks_7_net_3_bias, h2, H, H);
    ln(h2, HyperFlux::OcclusionCulling::Weights::blocks_7_net_4_weight, HyperFlux::OcclusionCulling::Weights::blocks_7_net_4_bias, H);
    for (int i = 0; i < H; i++) h1[i] = h2[i] + res[i];

    // Output heads
    { // Occlusion head: 320→128→64→1
        float hd128[128], hd64[64];
        mm_gelu(HyperFlux::OcclusionCulling::Weights::head_0_weight, h1, HyperFlux::OcclusionCulling::Weights::head_0_bias, hd128, 128, H);
        mm_gelu(HyperFlux::OcclusionCulling::Weights::head_3_weight, hd128, HyperFlux::OcclusionCulling::Weights::head_3_bias, hd64, 64, 128);
        mm(HyperFlux::OcclusionCulling::Weights::head_5_weight, hd64, HyperFlux::OcclusionCulling::Weights::head_5_bias, &output[0], 1, 64);
        output[1] = 1.0f / (1.0f + expf(-output[0]));
        output[0] = output[0];
    }
}

static void encode_input(const HG_CameraInfo* cam, const HG_ObjectBounds* obj,
                          const HG_OccluderInfo* occ, float* input) {
    float dx_obj = obj->posX - cam->cameraPosX;
    float dy_obj = obj->posY - cam->cameraPosY;
    float dz_obj = obj->posZ - cam->cameraPosZ;
    float dist_obj = sqrtf(dx_obj*dx_obj + dy_obj*dy_obj + dz_obj*dz_obj);
    
    float dx_occ = occ->posX - cam->cameraPosX;
    float dy_occ = occ->posY - cam->cameraPosY;
    float dz_occ = occ->posZ - cam->cameraPosZ;
    float dist_occ = sqrtf(dx_occ*dx_occ + dy_occ*dy_occ + dz_occ*dz_occ);
    
    float ndot_obj = (dx_obj*cam->cameraForwardX + dy_obj*cam->cameraForwardY + dz_obj*cam->cameraForwardZ) / (dist_obj + 1e-6f);
    float ndot_occ = (dx_occ*cam->cameraForwardX + dy_occ*cam->cameraForwardY + dz_occ*cam->cameraForwardZ) / (dist_occ + 1e-6f);
    
    input[0] = dx_obj / 1000.0f;
    input[1] = dy_obj / 1000.0f;
    input[2] = dz_obj / 1000.0f;
    input[3] = dist_obj / 1000.0f;
    input[4] = ndot_obj;
    input[5] = obj->boundingRadius / 100.0f;
    input[6] = obj->boundingRadius / (dist_obj + 1e-6f);
    input[7] = cam->cameraFOV / 120.0f;
    input[8] = dx_occ / 1000.0f;
    input[9] = dy_occ / 1000.0f;
    input[10] = dz_occ / 1000.0f;
    input[11] = dist_occ / 1000.0f;
    input[12] = ndot_occ;
    input[13] = occ->size / 100.0f;
    input[14] = occ->size / (dist_occ + 1e-6f);
    input[15] = (float)occ->type / 5.0f;
    input[16] = (dist_occ < dist_obj) ? 1.0f : 0.0f;
    input[17] = (dist_obj - dist_occ) / 1000.0f;
    input[18] = logf(dist_obj + 1.0f) / 10.0f;
    input[19] = logf(dist_occ + 1.0f) / 10.0f;
    input[20] = (float)(occ->type == OCCLUDER_STRUCTURE);
    input[21] = (float)(occ->type == OCCLUDER_TERRAIN_LARGE);
    input[22] = (float)(occ->type == OCCLUDER_WALL_THICK);
    input[23] = occ->size * ndot_occ;
}

int HG_Occlusion_Init(void) { g_init = 1; return 0; }
void HG_Occlusion_Shutdown(void) { g_init = 0; }

int HG_Occlusion_Test(const HG_CameraInfo* camera, const HG_ObjectBounds* object,
                       const HG_OccluderInfo* occluder, HG_VisibilityResult* result) {
    float input[IN], output[OUT];
    encode_input(camera, object, occluder, input);
    forward(input, output);
    result->isVisible = output[0] > 0.5f ? 1 : 0;
    result->confidence = output[1];
    return 0;
}

int HG_Occlusion_TestBatch(const HG_CameraInfo* camera, const HG_ObjectBounds* objects,
                            const HG_OccluderInfo* occluders, HG_VisibilityResult* results, int count) {
    for (int i = 0; i < count; i++) HG_Occlusion_Test(camera, &objects[i], &occluders[i], &results[i]);
    return 0;
}