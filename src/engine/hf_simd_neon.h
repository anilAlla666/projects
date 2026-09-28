// ==========================================================================
// HyperFlux SDK - ARM NEON INT8 SIMD Kernels
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"

#if HF_ARCH_ARM64
#include <arm_neon.h>

namespace hyperflux {
namespace internal {

HF_INLINE void MatVecInt8_NEON(
    const int8_t* HF_RESTRICT weights,
    const int8_t* HF_RESTRICT input,
    float* HF_RESTRICT output,
    const float* HF_RESTRICT scales,
    const float* HF_RESTRICT biases,
    float input_scale,
    int in_dim,
    int out_dim)
{
    const int in_dim_padded = (in_dim + 15) & ~15;

    for (int o = 0; o < out_dim; o++) {
        const int8_t* w_row = weights + o * in_dim_padded;

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);

        for (int i = 0; i < in_dim_padded; i += 16) {
            // Load 8 INT8 values
            int8x8_t w_lo = vld1_s8(w_row + i);
            int8x8_t x_lo = vld1_s8(input + i);
            int8x8_t w_hi = vld1_s8(w_row + i + 8);
            int8x8_t x_hi = vld1_s8(input + i + 8);

            // Widen multiply: INT8 × INT8 → INT16
            int16x8_t prod_lo = vmull_s8(w_lo, x_lo);
            int16x8_t prod_hi = vmull_s8(w_hi, x_hi);

            // Pairwise add and accumulate to INT32
            acc0 = vpadalq_s16(acc0, prod_lo);
            acc1 = vpadalq_s16(acc1, prod_hi);
        }

        // Horizontal sum
        int32x4_t total = vaddq_s32(acc0, acc1);
        int32_t dot = vaddvq_s32(total);

        output[o] = (float)dot * scales[o] * input_scale + biases[o];
    }
}

}} // namespace hyperflux::internal

#endif // HF_ARCH_ARM64
