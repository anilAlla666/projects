// ==========================================================================
// HyperFlux SDK - SIMD Runtime Detection & Dispatch
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// INTERNAL - Not part of public API.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"
#include <cstdint>

namespace hyperflux {
namespace internal {

// Detected at Init() time
struct SimdCaps {
    uint32_t flags = HF_SIMD_SCALAR;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_avx512vnni = false;
    bool has_neon = false;
};

// Detect hardware SIMD capabilities
SimdCaps DetectSimd();

// Global caps (set once during Init)
extern SimdCaps g_simd_caps;
extern uint32_t g_active_backend;

// ============================================================================
// SIMD Matmul function pointer type
// Computes: output[out_dim] = W[out_dim x in_dim] @ input[in_dim] * scales + biases
// with INT8 weights, INT8 input, FP32 output
// ============================================================================
typedef void (*MatVecInt8Fn)(
    const int8_t* weights,      // [out_dim × in_dim], row-major
    const int8_t* input,        // [in_dim], quantized
    float* output,              // [out_dim], dequantized
    const float* scales,        // [out_dim], per-output-channel scale
    const float* biases,        // [out_dim]
    float input_scale,          // input quantization scale
    int in_dim,
    int out_dim
);

// Active dispatch function (set at Init)
extern MatVecInt8Fn g_matvec_int8;

// Select best matvec implementation based on caps
void SelectMatVec(const SimdCaps& caps, uint32_t preferred);

}} // namespace hyperflux::internal
