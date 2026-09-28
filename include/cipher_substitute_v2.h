// CIPHER Substitution Engine v2 — Stage 6
//
// Public API for the runtime substitute pipeline: NVRTC-driven compute_90a
// kernels, two-tier (PTX + CUBIN) compilation cache, TMA descriptor reuse,
// per-shape correctness gate. Default OFF: env CIPHER_SUBSTITUTE_V2=on.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherSubstituteV2Stats {
    int      enabled;
    uint64_t register_calls;        // shapes registered for substitution
    uint64_t compile_attempts;
    uint64_t compile_success;
    uint64_t compile_cache_hits;
    uint64_t substitutions;          // launches taken via substitute path
    uint64_t passthroughs;           // launches that reverted to fp16
    uint64_t correctness_failures;   // shapes evicted because gate failed
    uint64_t background_jobs;
} CipherSubstituteV2Stats;

int  cipher_substitute_v2_init(void);
int  cipher_substitute_v2_enabled(void);

// Register a (m,n,k,dtype) shape for the substitute pipeline. Returns a
// shape_id (≥0) or -1 on failure.
int  cipher_substitute_v2_register_shape(int m, int n, int k, int dtype);

// Returns 1 if `shape_id` has a compiled, gate-passed substitute kernel
// ready. Off hot path.
int  cipher_substitute_v2_ready(int shape_id);

// Mark a shape as failed the correctness gate; engine evicts and reverts to
// fp16 passthrough for that shape.
int  cipher_substitute_v2_invalidate(int shape_id);

int  cipher_substitute_v2_stats(CipherSubstituteV2Stats* out);
void cipher_substitute_v2_report(void);

// Stage 6 actuation: compile `source` (CUDA C source) for `kernel_name` at
// arch sm_90a. Returns a positive cubin_id on success, 0 on failure. The
// CUmodule + CUfunction are stored internally, retrievable via
// cipher_substitute_v2_get_function. dlopen-based; safe if NVRTC missing.
unsigned long cipher_substitute_v2_compile(const char* source, const char* kernel_name);
// Returns the CUfunction handle for `cubin_id`, or NULL.
void*         cipher_substitute_v2_get_function(unsigned long cubin_id);
// Release a compiled module.
int           cipher_substitute_v2_destroy(unsigned long cubin_id);

#ifdef __cplusplus
}
#endif
