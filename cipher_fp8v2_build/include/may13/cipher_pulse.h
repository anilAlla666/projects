// Op 22 PULSE — Hardware fault early warning (Stage 1 collector + Stage 2 evaluator).
//
// Default OFF: env var `CIPHER_PULSE=on` enables.
//
// v1 has TWO of three signals — score is therefore capped at 2 (WARNING)
// in v1, and CRITICAL (score == 3) is UNREACHABLE.
//
// Signals:
//   1. Per-shape execution-time drift (own Welford, long window):
//        triggers when (mean - long_baseline_mean) > 2σ AND mean/baseline > 1.10
//   2. Per-substitution max_diff degradation                          ⏸ DEFERRED
//        Requires sentinel comparison in Op 3 SUBSTITUTE — Stage 0 hook,
//        same family as SHIELD/SUSTAIN/THERMOSTAT deferrals. Until that
//        hook is approved and built, Signal 2 contributes 0 to the score.
//   3. NVML correctable ECC error rate:
//        triggers when ECC delta vs init baseline > 10 in any 1-hour window.
//
// Score thresholds (v1):
//   0 → no alert
//   1 → INFO     → append /tmp/cipher_pulse.log
//   2 → WARNING  → write /tmp/cipher_pulse_alert.json (severity=MEDIUM)
//   3 → CRITICAL → unreachable in v1; would also signal Op 19 CONTINUITY
//
// Alert JSON includes "signal_2_deferred": true so operators reading the
// file know the ceiling and why.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int         cipher_pulse_init(void);
void        cipher_pulse_observe(const CipherRingEntry* ev);  // Stage 1
void        cipher_pulse_evaluate(void);                       // Stage 2
int         cipher_pulse_score(void);                          // 0..2 in v1
unsigned    cipher_pulse_alert_count(void);
const char* cipher_pulse_severity(void);                       // "NONE"|"INFO"|"WARNING"

// Test-only injection
void        cipher_pulse_inject_drift_for_shape(uint32_t M, uint32_t K, uint32_t N,
                                                 double mean_inflation_factor);
void        cipher_pulse_inject_ecc_delta(uint64_t additional_correctable);
void        cipher_pulse_clear_injection(void);
void        cipher_pulse_force_evaluate(void);                 // bypasses dispatch-count gate

void        cipher_pulse_report(void);   // /tmp/cipher_pulse_report.json

#ifdef __cplusplus
}
#endif
