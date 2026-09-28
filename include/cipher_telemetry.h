// =============================================================================
// CIPHER — F5: Hardware Telemetry Pipeline
// cipher_telemetry.h
//
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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
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
    // CUPTI context + metric IDs
    CUpti_Profiler_Initialize_Params profiler_params;
    CUpti_Profiler_BeginSession_Params session_params;
    bool                cupti_initialized;

    // NVML handles for temperature + power (simpler than CUPTI for these)
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

// Initialize CUPTI PM sampling + NVML. Starts background thread.
// liquid_mgr: the liquid state to update every 2ms.
int cipher_telemetry_init(CipherTelemetryState* state,
                          int                   device_ordinal,
                          CipherLiquidStateMgr* liquid_mgr);

// Stop background thread and clean up CUPTI/NVML.
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
