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

#ifdef __cplusplus
}
#endif
