// =============================================================================
// CIPHER — F1: cuLaunchKernel Intercept Layer
// Neural Dynamics | Anil Kumar Alla
//
// Hooks cuLaunchKernel via cuGetProcAddress before any GPU compute touches
// silicon. LD_PRELOAD on libcuda.so. Sub-100ns passthrough. Transparent to
// all frameworks: PyTorch, JAX, TensorFlow, cuBLAS, cuDNN — they all land here.
//
// MECHANISM: cuGetProcAddress is the stable NVIDIA-documented API for driver
// function resolution (same mechanism as Nsight Systems / NVProf). We
// intercept it, redirect cuLaunchKernel to cipher_dispatch(), and maintain
// a pointer to the real function for passthrough.
//
// DEPENDENCY: None. Ships standalone. Everything else in CIPHER depends on F1.
// =============================================================================

#pragma once

#include <cuda.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Dispatch result — returned by cipher_dispatch() to the intercept layer
// ---------------------------------------------------------------------------
typedef enum {
    CIPHER_PASS_THROUGH   = 0,   // Forward unchanged to real cuLaunchKernel
    CIPHER_SUBSTITUTED    = 1,   // O(1) neural equivalent dispatched
    CIPHER_DEFERRED       = 2,   // Queued for Layer 1 Koopman derivation
} CipherDispatchResult;

// ---------------------------------------------------------------------------
// Kernel launch descriptor — everything cuLaunchKernel exposes
// This struct is the boundary between F1 and the rest of CIPHER.
// ---------------------------------------------------------------------------
typedef struct {
    CUfunction  fn;
    uint32_t    grid_x, grid_y, grid_z;
    uint32_t    block_x, block_y, block_z;
    uint32_t    shared_bytes;
    CUstream    stream;
    void**      params;
    void**      extra;
    // Populated by Layer 3 classification
    uint8_t     op_class;          // 0-6 from cipher_classify.hpp
    uint8_t     confidence;        // 0-100
    uint64_t    intercept_ns;      // CLOCK_MONOTONIC_RAW timestamp at entry
} CipherKernelDesc;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Called at .so constructor (LD_PRELOAD). Installs the hook.
// Returns CUDA_SUCCESS on success.
CUresult cipher_intercept_init(void);

// Called at .so destructor. Removes hook, flushes state.
void cipher_intercept_teardown(void);

// The actual dispatch entry point — invoked inside our cuLaunchKernel shim.
// Classifies, routes, and either substitutes or passes through.
CipherDispatchResult cipher_dispatch(CipherKernelDesc* desc);

// Passthrough — calls the real cuLaunchKernel with original args.
// Always available even if CIPHER is mid-initialization.
CUresult cipher_passthrough(const CipherKernelDesc* desc);

// Live intercept stats — zeroed at init, updated atomically.
typedef struct {
    uint64_t total_intercepts;
    uint64_t substitutions;
    uint64_t passthroughs;
    uint64_t deferred;
    uint64_t overhead_ns_sum;    // sum of (post_dispatch - intercept_ns)
    uint64_t overhead_ns_max;
} CipherInterceptStats;

const CipherInterceptStats* cipher_intercept_stats(void);

#ifdef __cplusplus
}
#endif
