// CIPHER — F5: Hardware Telemetry Pipeline
// cipher_telemetry.h
//
#ifdef CIPHER_CUPTI_AVAILABLE
// Async collection of 32 hardware metrics via CUPTI Performance Monitoring
// (PM) Sampling API. Feeds the hardware_trajectory field of the liquid state
// every 2ms. Adds zero latency to the kernel dispatch critical path.
//
// METRICS COLLECTED (→ CipherHwTrajectory):
//   SM utilization    — CUPTI_ACTIVITY_KIND_SM_ACTIVITY or PM counter
//   L2 hit rate       — l2_hit_rate_sector_pipe_lsu_miss_rate (complement)
//   HBM bandwidth     — dram_read_bytes + dram_write_bytes / Δt
//   NVLink utilization— nvlrx/nvltx bytes / Δt
//   GPU temperature   — NVML (non-CUPTI, simpler)
//   Power draw        — NVML
//
// IMPLEMENTATION:
//   PM Sampling requires CUDA 11.6+ and a dedicated context. We run it on
//   CIPHER_CTX_LAYER2's Green Context. Sampling rate: 500Hz (2ms period).
//   Background thread spins on cudaEventSynchronize + PM decode loop.
//   Writes atomically to the liquid state HW trajectory field.
//
// SUCCESS CRITERION: 32-dim context vector at <5µs total latency.
// DEPENDENCY: F2 (Green Contexts), F4 (Liquid state must be initialized).
// =============================================================================

#endif  // CIPHER_CUPTI_AVAILABLE
#pragma once
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#else
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include "cipher_liquid_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// Telemetry collection interval (2ms = 500Hz)
#define CIPHER_TELEMETRY_INTERVAL_NS  2000000ULL

// ---------------------------------------------------------------------------
// Telemetry state
// ---------------------------------------------------------------------------

typedef struct {
    // CUPTI state — stored as opaque bytes to avoid header dependency.
    // Actual CUPTI types are only used inside cipher_telemetry.cpp.
    // Size: max(sizeof(CUpti_Profiler_Initialize_Params),
    //           sizeof(CUpti_Profiler_BeginSession_Params)) = ~256 bytes.
    uint8_t             cupti_state[512];  // Opaque CUPTI params storage
    bool                cupti_initialized;
    void*               nvml_device;    // nvmlDevice_t — opaque to avoid nvml.h dep
    bool                nvml_available;

    // Background sampling thread
    pthread_t           sample_thread;
    volatile bool       thread_running;
    volatile bool       stop_requested;

    // Target: the liquid state HW trajectory to update
    CipherLiquidStateMgr* liquid_mgr;

    // Last sample values (double-buffer to avoid tearing)
    CipherHwTrajectory  buf[2];
    volatile int        write_buf;  // 0 or 1 — which buf is being written

    // Timing
    uint64_t            last_sample_ns;
    uint64_t            sample_count;

    // CUDA device
    int                 device_ordinal;
    bool                initialized;
} CipherTelemetryState;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#ifdef CIPHER_CUPTI_AVAILABLE
// Initialize CUPTI PM sampling + NVML. Starts background thread.
// liquid_mgr: the liquid state to update every 2ms.
#endif  // CIPHER_CUPTI_AVAILABLE
int cipher_telemetry_init(CipherTelemetryState* state,
                          int                   device_ordinal,
                          CipherLiquidStateMgr* liquid_mgr);

#ifdef CIPHER_CUPTI_AVAILABLE
// Stop background thread and clean up CUPTI/NVML.
#endif  // CIPHER_CUPTI_AVAILABLE
void cipher_telemetry_destroy(CipherTelemetryState* state);

// Force a synchronous sample (for testing, not for runtime use).
// Writes directly to provided hw_out buffer.
int cipher_telemetry_sample_sync(CipherTelemetryState* state,
                                 CipherHwTrajectory*   hw_out);

// Report telemetry stats.
void cipher_telemetry_report(const CipherTelemetryState* state);

#ifdef __cplusplus
}
#endif
