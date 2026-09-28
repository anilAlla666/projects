// Op 31 HIBERNATE — Execution-idle SM power gating.
//
// Detects 5–50 ms idle gaps between dispatches (where DVFS at 100 ms
// granularity cannot react). When an idle window is open, the
// idle_gate_flag is raised; on a permissioned host, ARBITRATE issues
// nvmlDeviceSetPowerManagementLimit() to drop the floor power for the
// idle session's SM partition.
//
// Path B: production-ready code path with pod-degraded fallback. On this
// pod nvmlDeviceSetPowerManagementLimit returns NOT_SUPPORTED — same probe
// pattern as VOLT.
//
// Default OFF: env var `CIPHER_HIBERNATE=on` enables.
// `CIPHER_HIBERNATE_THRESHOLD_MS` (default 5) sets the idle gate.
//
// Same signal-handler discipline as VOLT: all NVML pointers cached at init.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_hibernate_init(void);
void     cipher_hibernate_observe(const CipherRingEntry* ev);   // Stage 1
void     cipher_hibernate_poll(void);                           // Stage 2
int      cipher_hibernate_actuation_supported(void);
int      cipher_hibernate_idle_gate_flag(void);                 // 0 / 1
unsigned cipher_hibernate_idle_event_count(void);               // # times gate raised
unsigned cipher_hibernate_pre_engage_count(void);               // # times pre-engaged
const char* cipher_hibernate_status_string(void);

// Test-only injection
void     cipher_hibernate_inject_idle_gap_ms(unsigned ms);
void     cipher_hibernate_clear_injection(void);
void     cipher_hibernate_force_poll(void);

void     cipher_hibernate_report(void);   // /tmp/cipher_hibernate_report.json

#ifdef __cplusplus
}
#endif
