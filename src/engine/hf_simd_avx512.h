// ==========================================================================
// HyperFlux SDK - AVX-512 / VNNI INT8 SIMD Kernels
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"

#if HF_ARCH_X86_64
#include <immintrin.h>

namespace hyperflux {
namespace internal {

#ifdef __AVX512F__
HF_INLINE void MatVecInt8_AVX512(
    const int8_t* HF_RESTRICT weights,
    const int8_t* HF_RESTRICT input,
    float* HF_RESTRICT output,
    const float* HF_RESTRICT scales,
    const float* HF_RESTRICT biases,
    float input_scale,
    int in_dim,
    int out_dim)
{
    const int in_dim_padded = (in_dim + 63) & ~63;

    for (int o = 0; o < out_dim; o++) {
        const int8_t* w_row = weights + o * in_dim_padded;

        __m512i acc = _mm512_setzero_si512();

        for (int i = 0; i < in_dim_padded; i += 64) {
            // Load 32 bytes at a time, process as INT16
            __m256i w_lo = _mm256_load_si256((const __m256i*)(w_row + i));
            __m256i x_lo = _mm256_load_si256((const __m256i*)(input + i));
            __m512i w16_lo = _mm512_cvtepi8_epi16(w_lo);
            __m512i x16_lo = _mm512_cvtepi8_epi16(x_lo);
            __m512i prod_lo = _mm512_mullo_epi16(w16_lo, x16_lo);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(prod_lo, _mm512_set1_epi16(1)));

            __m256i w_hi = _mm256_load_si256((const __m256i*)(w_row + i + 32));
            __m256i x_hi = _mm256_load_si256((const __m256i*)(input + i + 32));
            __m512i w16_hi = _mm512_cvtepi8_epi16(w_hi);
            __m512i x16_hi = _mm512_cvtepi8_epi16(x_hi);
            __m512i prod_hi = _mm512_mullo_epi16(w16_hi, x16_hi);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(prod_hi, _mm512_set1_epi16(1)));
        }

        int32_t dot = _mm512_reduce_add_epi32(acc);
        output[o] = (float)dot * scales[o] * input_scale + biases[o];
    }
}
#endif // __AVX512F__

#ifdef __AVX512VNNI__
HF_INLINE void MatVecInt8_VNNI(
    const int8_t* HF_RESTRICT weights,
    const int8_t* HF_RESTRICT input,
    float* HF_RESTRICT output,
    const float* HF_RESTRICT scales,
    const float* HF_RESTRICT biases,
    float input_scale,
    int in_dim,
    int out_dim)
{
    // VNNI: _mm512_dpbusd_epi32 does 4×INT8 dot product directly
    // Requires first operand unsigned, second signed
    // We offset weights by +128 and correct after
    const int in_dim_padded = (in_dim + 63) & ~63;
    const __m512i offset = _mm512_set1_epi8((char)128);

    for (int o = 0; o < out_dim; o++) {
        const int8_t* w_row = weights + o * in_dim_padded;
        __m512i acc = _mm512_setzero_si512();
        __m512i x_sum = _mm512_setzero_si512(); // For correction

        for (int i = 0; i < in_dim_padded; i += 64) {
            __m512i w = _mm512_load_si512((const __m512i*)(w_row + i));
            __m512i x = _mm512_load_si512((const __m512i*)(input + i));

            // Make weights unsigned: w_u = w + 128
            __m512i w_u = _mm512_add_epi8(w, offset);

            // VNNI: acc += dot4(w_u, x) per 32-bit lane
            acc = _mm512_dpbusd_epi32(acc, w_u, x);

            // Track sum of x for offset correction
            __m512i ones = _mm512_set1_epi8(1);
            x_sum = _mm512_dpbusd_epi32(x_sum, (const __m512i)ones, x);
        }

        // Correction: subtract 128 * sum(x) per group
        int32_t dot = _mm512_reduce_add_epi32(acc);
        int32_t x_total = _mm512_reduce_add_epi32(x_sum);
        dot -= 128 * x_total;

        output[o] = (float)dot * scales[o] * input_scale + biases[o];
    }
}
#endif // __AVX512VNNI__

}} // namespace hyperflux::internal

#endif // HF_ARCH_X86_64
