// ==========================================================================
// HyperFlux SDK - Scalar Fallback (no SIMD)
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"
#include <cstdint>

namespace hyperflux {
namespace internal {

HF_INLINE void MatVecInt8_Scalar(
    const int8_t* HF_RESTRICT weights,
    const int8_t* HF_RESTRICT input,
    float* HF_RESTRICT output,
    const float* HF_RESTRICT scales,
    const float* HF_RESTRICT biases,
    float input_scale,
    int in_dim,
    int out_dim)
{
    for (int o = 0; o < out_dim; o++) {
        const int8_t* w_row = weights + o * in_dim;
        int32_t acc = 0;
        for (int i = 0; i < in_dim; i++) {
            acc += (int32_t)w_row[i] * (int32_t)input[i];
        }
        output[o] = (float)acc * scales[o] * input_scale + biases[o];
    }
}

}} // namespace hyperflux::internal
