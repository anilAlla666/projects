// ==========================================================================
// HyperFlux SDK - Platform Configuration
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// CONFIDENTIAL AND PROPRIETARY - Do not distribute.
// ==========================================================================
#pragma once

// Version
#define HF_VERSION_MAJOR 1
#define HF_VERSION_MINOR 0
#define HF_VERSION_PATCH 0
#define HF_VERSION_STRING "1.0.0"

// Platform detection
#if defined(_MSC_VER)
    #define HF_PLATFORM_WINDOWS 1
    #define HF_COMPILER_MSVC 1
    #define HF_ALIGN(n) __declspec(align(n))
    #define HF_INLINE __forceinline
    #define HF_RESTRICT __restrict
    #define HF_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(__APPLE__)
        #define HF_PLATFORM_APPLE 1
    #else
        #define HF_PLATFORM_LINUX 1
    #endif
    #define HF_COMPILER_GCC 1
    #define HF_ALIGN(n) __attribute__((aligned(n)))
    #define HF_INLINE __attribute__((always_inline)) inline
    #define HF_RESTRICT __restrict__
    #define HF_EXPORT __attribute__((visibility("default")))
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
    #define HF_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define HF_ARCH_ARM64 1
#endif

// SIMD capability flags (set at runtime)
#define HF_SIMD_SCALAR   0x00
#define HF_SIMD_AVX2     0x01
#define HF_SIMD_AVX512   0x02
#define HF_SIMD_VNNI     0x04
#define HF_SIMD_NEON     0x08

// Memory alignment for SIMD
#define HF_ALIGNMENT 64

// Maximum supported dimensions
#define HF_MAX_LAYERS 8
#define HF_MAX_DIM 512
#define HF_MAX_BATCH 64

// Debug mode
#ifdef HF_DEBUG
    #include <cstdio>
    #define HF_LOG(fmt, ...) fprintf(stderr, "[HyperFlux] " fmt "\n", ##__VA_ARGS__)
    #define HF_ASSERT(cond) do { if(!(cond)) { HF_LOG("ASSERT FAILED: %s at %s:%d", #cond, __FILE__, __LINE__); __builtin_trap(); } } while(0)
#else
    #define HF_LOG(fmt, ...) ((void)0)
    #define HF_ASSERT(cond) ((void)0)
#endif

// Profiling mode
#ifdef HF_PROFILE
    #include <chrono>
    #define HF_PROFILE_BEGIN(name) auto _hf_t_##name = std::chrono::high_resolution_clock::now()
    #define HF_PROFILE_END(name) do { \
        auto _hf_dt = std::chrono::high_resolution_clock::now() - _hf_t_##name; \
        HF_LOG("PROFILE %s: %.2f us", #name, std::chrono::duration<double, std::micro>(_hf_dt).count()); \
    } while(0)
#else
    #define HF_PROFILE_BEGIN(name) ((void)0)
    #define HF_PROFILE_END(name) ((void)0)
#endif
