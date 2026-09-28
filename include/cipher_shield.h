// Op 14 SHIELD — Human-session latency protection (Stage 1 monitor).
//
// Default OFF: env var `CIPHER_SHIELD=on` enables (also requires
// CIPHER_SENSE=on, since SHIELD reads SENSE's session classification).
//
// v1 scope (per the approved plan):
//   Protection 1 — TTFT Guard: when SENSE classifies a session as
//     HUMAN_INTERACTIVE, SHIELD writes CIPHER_BAND_PROTECTED into the
//     cipher_sm_set_priority hint table. ARBITRATE (Stage 2) logs the hint;
//     actual SM rebalancing is deferred to a v2 plan.
//   Protection 3 — Latency Smoothing: SHIELD tracks per-session ITL P50/P95
//     over a 20-event rolling window. When P95/P50 > 3.0, sets a
//     per-session oracle_aggressive flag (NOT yet read by Stage 0 oracle —
//     wiring is a separate Stage-0 hook plan).
//
// Deferred to v2 (explicit user approval required for the Stage 0 hooks):
//   Protection 2 — ITL jitter cache_aggressive_flag → Op 3 SUBSTITUTE
//   Oracle aggressiveness wired into Gate 2 of cipher_oracle.cpp
//   ARBITRATE actually changing SM split based on band hints
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_shield_init(void);
void     cipher_shield_observe(const CipherRingEntry* ev);
void     cipher_shield_arbitrate_scan(void);
unsigned cipher_shield_jitter_event_count(void);
unsigned cipher_shield_smoothed_session_count(void);
unsigned cipher_shield_priority_assignments(void);
void     cipher_shield_report(void);

#ifdef __cplusplus
}
#endif
