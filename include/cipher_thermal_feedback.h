// CIPHER Thermal-Substitution Feedback Loop — Stage 11
//
// THERMOSTAT (telemetry) writes (clock_sustained, headroom, power) into the
// silicon model. This module reads those, computes an adaptive substitution
// aggressiveness, and exposes it to SUBSTITUTE/Op 3 callers via an
// observable score in [0.0, 1.0].
//
// At low headroom (high power, low clock): aggressiveness ↑ — substitute
// more iterative ops to relieve power and let DVFS recover the clock.
// At high headroom (low power, high clock): aggressiveness ↓ — less need
// to substitute, let cuBLAS run.
//
// Default OFF. Env: CIPHER_THERMAL_FEEDBACK=on.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherThermalFeedbackStats {
    int      enabled;
    double   substitute_aggressiveness;     // 0..1, current operating point
    double   last_clock_mhz;
    double   last_headroom;
    double   last_power_w;
    uint64_t tick_calls;
    uint64_t aggressiveness_increased;
    uint64_t aggressiveness_decreased;
    uint64_t oscillation_clamps;
} CipherThermalFeedbackStats;

int  cipher_thermal_feedback_init(void);
int  cipher_thermal_feedback_enabled(void);

// Read silicon-model dynamic fields and update aggressiveness.
// Caller (THERMOSTAT periodic worker) invokes once per sample tick.
void cipher_thermal_feedback_tick(void);

// Read the current aggressiveness score (0..1). Cheap.
double cipher_thermal_feedback_aggressiveness(void);

int  cipher_thermal_feedback_stats(CipherThermalFeedbackStats* out);
void cipher_thermal_feedback_report(void);

// OP 32 — DVFS graceful privilege handling.
//
// status:  0=unprobed, 1=supported, 2=not-supported, 3=no-permission, 4=disabled-by-env
// last_rc: NVML return code from last attempt (0=success, 4=NOT_SUPPORTED, 7=NO_PERMISSION)
typedef struct CipherDvfsStatus {
    int      enabled_request;     // CIPHER_DVFS=on requested
    int      status;              // 0..4 above
    int      last_rc;
    int      min_mhz;
    int      max_mhz;
    int      cur_target_mhz;
    uint64_t set_attempts;
    uint64_t set_successes;
    uint64_t set_failures_nopriv;
    uint64_t set_failures_unsupp;
    uint64_t set_failures_other;
    uint64_t reset_on_exit;
} CipherDvfsStatus;

int  cipher_dvfs_status(CipherDvfsStatus* out);

#ifdef __cplusplus
}
#endif
