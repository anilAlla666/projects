// CIPHER Silicon Model — Stage 2
//
// Single read-only data structure populated once at init from CUDA + NVML
// queries. Every layer / op references it instead of carrying its own
// constants. Dynamic fields (clock_sustained_mhz, thermal_headroom,
// power_draw_watts) are written exclusively by THERMOSTAT and read with
// atomic-relaxed loads from any thread.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifndef __cplusplus
#  include <stdatomic.h>
#  define _CIPHER_ATOMIC(T) _Atomic T
#else
#  include <atomic>
#  define _CIPHER_ATOMIC(T) std::atomic<T>
   extern "C" {
#endif

typedef struct CipherSiliconModel {
    // ── Static (immutable after init) ────────────────────────────────────────
    int    sm_count;
    int    compute_major;
    int    compute_minor;
    int    clock_boost_mhz;
    int    memory_clock_mhz;
    int    memory_bus_width_bits;
    size_t l2_total;
    size_t l2_persist_max;
    size_t l2_window_max;
    size_t hbm_total_bytes;
    double hbm_bandwidth_bytes_per_sec;
    double peak_fp16_tflops;            // sparse FP16 TC peak, derived from arch ref scaled by queried sm/clock
    double fp16_intensity_threshold;    // FLOPS/byte = peak_fp16_tflops*1e12 / hbm_bandwidth
    double power_cap_watts;             // current NVML power-management limit; 0.0 if NVML unavailable
    int    mig_active;                  // 1 iff persistingL2CacheMaxSize == 0
    int    mps_active;                  // 1 iff CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set

    // ── Dynamic (atomic; THERMOSTAT writes, observers read) ──────────────────
    _CIPHER_ATOMIC(int)    clock_sustained_mhz;   // last observed sustained clock
    _CIPHER_ATOMIC(double) thermal_headroom;       // 0.0 = throttling, 1.0 = full headroom
    _CIPHER_ATOMIC(double) power_draw_watts;       // last observed power draw

    // ── Init bookkeeping ─────────────────────────────────────────────────────
    int initialized;                    // 1 after successful init
} CipherSiliconModel;

// Idempotent. Returns 1 on success, 0 if no CUDA device is available.
int cipher_silicon_init(void);

// Read-only handle. Returns NULL until init has succeeded.
const CipherSiliconModel* cipher_silicon_get(void);

// Atomic write of dynamic fields. Safe from any thread; intended to be called
// by THERMOSTAT on its periodic sample tick.
void cipher_silicon_update(int clock_mhz, double thermal_headroom, double power_w);

#ifdef __cplusplus
} // extern "C"
#endif
