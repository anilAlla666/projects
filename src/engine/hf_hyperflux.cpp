// ==========================================================================
// HyperFlux SDK - Library Initialization
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "../include/hyperflux/hyperflux.h"
#include "hf_simd_dispatch.h"

namespace hyperflux {

static bool s_initialized = false;

Status Init(const InitConfig& config) {
    if (s_initialized) return Status::OK;

    // Detect SIMD capabilities
    internal::g_simd_caps = internal::DetectSimd();

    // Select dispatch
    uint32_t preferred = 0;
    switch (config.preferred_backend) {
        case SimdBackend::AVX2:   preferred = HF_SIMD_AVX2; break;
        case SimdBackend::AVX512: preferred = HF_SIMD_AVX512; break;
        case SimdBackend::VNNI:   preferred = HF_SIMD_VNNI; break;
        case SimdBackend::NEON:   preferred = HF_SIMD_NEON; break;
        case SimdBackend::SCALAR: preferred = HF_SIMD_SCALAR; break;
        default: preferred = 0; break;
    }

    internal::SelectMatVec(internal::g_simd_caps, preferred);

    s_initialized = true;
    HF_LOG("HyperFlux SDK v%s initialized", HF_VERSION_STRING);
    return Status::OK;
}

void Shutdown() {
    if (!s_initialized) return;
    internal::g_matvec_int8 = nullptr;
    s_initialized = false;
    HF_LOG("HyperFlux SDK shutdown");
}

const char* GetBackendName() {
    switch (internal::g_active_backend) {
        case HF_SIMD_VNNI:   return "AVX-512 VNNI";
        case HF_SIMD_AVX512: return "AVX-512";
        case HF_SIMD_AVX2:   return "AVX2";
        case HF_SIMD_NEON:   return "ARM NEON";
        default:              return "Scalar";
    }
}

uint32_t GetSimdCapabilities() {
    return internal::g_simd_caps.flags;
}

const char* GetVersion() {
    return HF_VERSION_STRING;
}

} // namespace hyperflux
