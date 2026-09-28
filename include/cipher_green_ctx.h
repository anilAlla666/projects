// CIPHER — F2: Green Context Allocation
// cipher_green_ctx.h + cipher_green_ctx.cu
//
// Carves 8 SMs out of the H100's 132 SMs (or A100's 108) and reserves them
// permanently for CIPHER's three LNNs. The remaining 124 SMs handle all user
// workloads unmodified.
//
// API: CUDA Green Contexts (introduced in CUDA 12.4 / Hopper architecture).
//   - cudaDeviceGetDevResourceRange() — query available SM granularity
//   - cuDevSmResourceSplitByCount()   — partition SMs into two resource sets
//   - cuGreenCtxCreate()             — bind resource set to a Green Context
//   - cuCtxFromGreenCtx()            — get a usable CUcontext from green ctx
//
// Three contexts created:
//   CIPHER_CTX_LAYER3 — Layer 3 Substitutor LNN
//   CIPHER_CTX_LAYER2 — Layer 2 Orchestrator LNN
//   CIPHER_CTX_LAYER1 — Layer 1 Generator LNN
//
// Shared liquid state lives in pinned host memory + device global memory
// accessible from all three contexts via peer access (F4).
//
// SUCCESS CRITERION: 8 SMs isolated, 124 SMs free for user workload.
// DEPENDENCY: F1 must be initialized.
// =============================================================================

#pragma once
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#else
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// CIPHER SM allocation constants
// ---------------------------------------------------------------------------

#define CIPHER_SM_COUNT          8      // SMs reserved for CIPHER
#define CIPHER_CTX_COUNT         3      // One per LNN (L3, L2, L1)
#define CIPHER_MIN_SM_TOTAL     32      // Don't run on cards with <32 SMs

typedef enum {
    CIPHER_CTX_LAYER3 = 0,   // Koopman Substitutor
    CIPHER_CTX_LAYER2 = 1,   // Koopman Orchestrator
    CIPHER_CTX_LAYER1 = 2,   // Koopman Generator
} CipherCtxId;

// ---------------------------------------------------------------------------
// Green context state — one per CIPHER process
// ---------------------------------------------------------------------------

typedef struct {
    // CUDA Green Context handles
    CUgreenCtx    green_ctx[CIPHER_CTX_COUNT];
    CUcontext     ctx[CIPHER_CTX_COUNT];    // Usable CUcontext for each LNN

    // SM resource descriptors (CUDA 12.4+ CUdevResource)
#ifndef CIPHER_CPU_STUB
    CUdevResource    cipher_resource;       // 8-SM resource set for CIPHER
    CUdevResource    workload_resource;     // Remaining SMs for user workload
#else
    CUdevSmResource  cipher_resource;
    CUdevSmResource  workload_resource;
#endif

    // Streams for LNN forward passes — one per context
    cudaStream_t  stream[CIPHER_CTX_COUNT];

    // Device the Green Contexts were created on
    CUdevice      device;
    int           total_sms;       // Hardware total
    int           cipher_sms;      // Actual allocated (may differ from 8 if
                                   // hardware granularity forces rounding)
    int           workload_sms;    // Remaining for user

    bool          initialized;
    bool          fallback_mode;   // True if Green Ctx unavailable (<CUDA 12.4)
} CipherGreenCtxState;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize Green Contexts. Must be called after cuInit() / CUDA runtime init.
// On pre-12.4 CUDA: falls back to shared-context mode (fallback_mode=true).
CUresult cipher_green_ctx_init(CipherGreenCtxState* state, int device_ordinal);

// Destroy all Green Contexts and free resources.
void cipher_green_ctx_destroy(CipherGreenCtxState* state);

// Get the CUcontext to use for a given LNN layer.
// In fallback mode, returns the current CUDA context.
CUcontext cipher_get_ctx(const CipherGreenCtxState* state, CipherCtxId id);

// Get the stream to submit LNN work on.
cudaStream_t cipher_get_stream(const CipherGreenCtxState* state, CipherCtxId id);

// Print allocation report to stderr.
void cipher_green_ctx_report(const CipherGreenCtxState* state);

// ---------------------------------------------------------------------------
// SM band hint table (Op 14 SHIELD support, v1 = log-only enforcement)
//
// SHIELD writes a per-session band hint via cipher_sm_set_priority(). Stage 2
// ARBITRATE reads via cipher_sm_get_priority() on its periodic loop. The
// table is a 256-slot atomic-packed (band:8 | sid_top:8) hash bucket, lossy
// on collisions, never wrong. No allocation, no syscall.
//
// Actual SM rebalancing (cuGreenCtxDestroy + recreate) is deferred — v1 just
// logs the hint so SHIELD's signal is observable.
// ---------------------------------------------------------------------------
typedef enum {
    CIPHER_BAND_DEFAULT    = 0,
    CIPHER_BAND_PROTECTED  = 1,   // HUMAN_INTERACTIVE TTFT priority
    CIPHER_BAND_BACKGROUND = 2,   // BATCH workloads can yield
} CipherSmBand;

void         cipher_sm_set_priority(uint64_t session_id, uint8_t band);
uint8_t      cipher_sm_get_priority(uint64_t session_id);
unsigned     cipher_sm_priority_count(void);  // tests/debug
void         cipher_sm_priority_scan_log(void); // ARBITRATE periodic logger

#ifdef __cplusplus
}
#endif
