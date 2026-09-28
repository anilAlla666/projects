// ==========================================================================
// HyperFlux SDK - AVX2 INT8 SIMD Kernels
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"

#if HF_ARCH_X86_64
#include <immintrin.h>

namespace hyperflux {
namespace internal {

HF_INLINE void MatVecInt8_AVX2(
    const int8_t* HF_RESTRICT weights,
    const int8_t* HF_RESTRICT input,
    float* HF_RESTRICT output,
    const float* HF_RESTRICT scales,
    const float* HF_RESTRICT biases,
    float input_scale,
    int in_dim,
    int out_dim)
{
    // Pad in_dim to multiple of 16 for aligned processing
    const int in_dim_padded = (in_dim + 15) & ~15;

    for (int o = 0; o < out_dim; o++) {
        const int8_t* w_row = weights + o * in_dim_padded;

        // Accumulate INT8 × INT8 → INT32 using AVX2
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();

        int i = 0;
        for (; i + 32 <= in_dim_padded; i += 32) {
            // Load 16 weights and 16 inputs, sign-extend to INT16
            __m128i w_lo = _mm_load_si128((const __m128i*)(w_row + i));
            __m128i x_lo = _mm_load_si128((const __m128i*)(input + i));
            __m256i w16_lo = _mm256_cvtepi8_epi16(w_lo);
            __m256i x16_lo = _mm256_cvtepi8_epi16(x_lo);

            // Multiply pairs and accumulate adjacent pairs to INT32
            __m256i prod_lo = _mm256_mullo_epi16(w16_lo, x16_lo);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod_lo, _mm256_set1_epi16(1)));

            // Second half
            __m128i w_hi = _mm_load_si128((const __m128i*)(w_row + i + 16));
            __m128i x_hi = _mm_load_si128((const __m128i*)(input + i + 16));
            __m256i w16_hi = _mm256_cvtepi8_epi16(w_hi);
            __m256i x16_hi = _mm256_cvtepi8_epi16(x_hi);

            __m256i prod_hi = _mm256_mullo_epi16(w16_hi, x16_hi);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod_hi, _mm256_set1_epi16(1)));
        }

        // Handle remaining 16 elements
        for (; i + 16 <= in_dim_padded; i += 16) {
            __m128i w_v = _mm_load_si128((const __m128i*)(w_row + i));
            __m128i x_v = _mm_load_si128((const __m128i*)(input + i));
            __m256i w16 = _mm256_cvtepi8_epi16(w_v);
            __m256i x16 = _mm256_cvtepi8_epi16(x_v);
            __m256i prod = _mm256_mullo_epi16(w16, x16);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
        }

        // Horizontal sum: acc0 + acc1
        __m256i total = _mm256_add_epi32(acc0, acc1);
        __m128i lo128 = _mm256_castsi256_si128(total);
        __m128i hi128 = _mm256_extracti128_si256(total, 1);
        __m128i sum128 = _mm_add_epi32(lo128, hi128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int32_t dot = _mm_extract_epi32(sum128, 0);

        // Dequantize: float_result = int32_dot * weight_scale * input_scale + bias
        output[o] = (float)dot * scales[o] * input_scale + biases[o];
    }
}

}} // namespace hyperflux::internal

#endif // HF_ARCH_X86_64
