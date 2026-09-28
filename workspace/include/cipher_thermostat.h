// Op 20 THERMOSTAT — Predictive thermal-throttle prevention.
//
// Default OFF: env var `CIPHER_THERMOSTAT=on` enables.
//
// Two-signal detector (both must be present simultaneously):
//   1. NVML temperature > 75 °C AND trending upward
//   2. Per-shape (M,K,N) Welford on inter-event ITL has drifted >10 % above
//      its long-window baseline (frozen after first 30 samples).
//
// When both signals fire, sets `g_thermostat_aggressive` (atomic, sticky).
// Cleared when temp < 72 °C AND no shape exceeds 5 % drift.
//
// v1: NVML temperature is read from g_cipher.liquid.device->hw.gpu_temp_c
// (already populated by the existing 500 Hz telemetry thread; THERMOSTAT
// adds zero new NVML calls). The aggressive flag is exported but NOT yet
// consumed by Op 3 SUBSTITUTE — wiring is a deferred Stage 0 plan.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_thermostat_init(void);
void     cipher_thermostat_observe(const CipherRingEntry* ev);   // Stage 1
void     cipher_thermostat_poll(void);                           // Stage 2
int      cipher_thermostat_aggressive_active(void);
unsigned cipher_thermostat_event_count(void);
unsigned cipher_thermostat_drift_shape_count(void);

// Test-only injection — bypass real NVML temp / real Welford drift so the
// gate runs without baking the GPU. Production never calls these.
void     cipher_thermostat_inject_temp(float celsius);            // < 0 = clear
void     cipher_thermostat_inject_drift_for_shape(uint32_t M, uint32_t K, uint32_t N,
                                                   double mean_inflation_factor);
void     cipher_thermostat_clear_injection(void);
void     cipher_thermostat_force_poll(void);   // bypasses 500 ms cadence (tests)

void     cipher_thermostat_report(void);   // /tmp/cipher_thermostat_report.json

#ifdef __cplusplus
}
#endif
