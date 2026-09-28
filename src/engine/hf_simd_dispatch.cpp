// ==========================================================================
// HyperFlux SDK - SIMD Detection & Dispatch Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "hf_simd_dispatch.h"
#include "hf_simd_scalar.h"

#if HF_ARCH_X86_64
#include "hf_simd_avx2.h"
#include "hf_simd_avx512.h"
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif
#endif

#if HF_ARCH_ARM64
#include "hf_simd_neon.h"
#endif

namespace hyperflux {
namespace internal {

SimdCaps g_simd_caps;
uint32_t g_active_backend = HF_SIMD_SCALAR;
MatVecInt8Fn g_matvec_int8 = nullptr;

#if HF_ARCH_X86_64
static void cpuid(int info[4], int leaf) {
#ifdef _MSC_VER
    __cpuid(info, leaf);
#else
    __cpuid_count(leaf, 0, info[0], info[1], info[2], info[3]);
#endif
}
#endif

SimdCaps DetectSimd() {
    SimdCaps caps;

#if HF_ARCH_X86_64
    int info[4];
    cpuid(info, 0);
    int max_leaf = info[0];

    if (max_leaf >= 7) {
        cpuid(info, 7);
        caps.has_avx2 = (info[1] & (1 << 5)) != 0;
        caps.has_avx512f = (info[1] & (1 << 16)) != 0;
        caps.has_avx512vnni = (info[2] & (1 << 11)) != 0;
    }

    if (caps.has_avx512vnni) caps.flags |= HF_SIMD_VNNI | HF_SIMD_AVX512 | HF_SIMD_AVX2;
    else if (caps.has_avx512f) caps.flags |= HF_SIMD_AVX512 | HF_SIMD_AVX2;
    else if (caps.has_avx2) caps.flags |= HF_SIMD_AVX2;
#endif

#if HF_ARCH_ARM64
    caps.has_neon = true;  // Always available on AArch64
    caps.flags |= HF_SIMD_NEON;
#endif

    return caps;
}

void SelectMatVec(const SimdCaps& caps, uint32_t preferred) {
    // Default: scalar
    g_matvec_int8 = MatVecInt8_Scalar;
    g_active_backend = HF_SIMD_SCALAR;

#if HF_ARCH_ARM64
    if (caps.has_neon) {
        g_matvec_int8 = MatVecInt8_NEON;
        g_active_backend = HF_SIMD_NEON;
    }
#endif

#if HF_ARCH_X86_64
    if (caps.has_avx2) {
        g_matvec_int8 = MatVecInt8_AVX2;
        g_active_backend = HF_SIMD_AVX2;
    }

#ifdef __AVX512F__
    if (caps.has_avx512f && (preferred == 0 || preferred == HF_SIMD_AVX512)) {
        g_matvec_int8 = MatVecInt8_AVX512;
        g_active_backend = HF_SIMD_AVX512;
    }
#endif

#ifdef __AVX512VNNI__
    if (caps.has_avx512vnni && (preferred == 0 || preferred == HF_SIMD_VNNI)) {
        g_matvec_int8 = MatVecInt8_VNNI;
        g_active_backend = HF_SIMD_VNNI;
    }
#endif
#endif

    HF_LOG("SIMD backend: %s (flags=0x%02x)",
           g_active_backend == HF_SIMD_VNNI ? "AVX-512 VNNI" :
           g_active_backend == HF_SIMD_AVX512 ? "AVX-512" :
           g_active_backend == HF_SIMD_AVX2 ? "AVX2" :
           g_active_backend == HF_SIMD_NEON ? "NEON" : "Scalar",
           caps.flags);
}

}} // namespace hyperflux::internal
