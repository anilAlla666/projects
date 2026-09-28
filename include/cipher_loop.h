// Op 26 LOOP — Agentic runaway detection (Stage 1 monitor).
//
// Default OFF: env var `CIPHER_LOOP=on` enables. Also benefits from
// `CIPHER_SENSE=on` — LOOP only scores sessions that SENSE has classified as
// AGENT_AUTONOMOUS (if SENSE is off, LOOP scores every session anyway,
// since runaway patterns are self-evident from shape repetition + no prefill
// refresh).
//
// Detection (three signals, per-session, no content access):
//   S1 Shape-cycle repetition — rolling 64-event window of shape-proxy tuples.
//      A repeating period ≤ 8 sustained for ≥ 3 full cycles contributes 1.
//   S2 Burn rate — decodes per wall-second exceeds the session's own
//      first-100-event baseline by ≥ 4× contributes 1.
//   S3 Prefill drought — events_since_last_prefill > 500 AND S1 active
//      contributes 1.
//
// Score = S1 + S2 + S3 (0..3). Score ≥ 2 ⇒ CIPHER_LOOP_RUNAWAY.
//
// v1 action: writes a CIPHER_BAND_DEMOTED hint (mirrors SHIELD's
// BAND_PROTECTED). ARBITRATE logs only; real SM throttle is v2.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Idempotent. Reads CIPHER_LOOP env. Returns 1 if LOOP is enabled, 0 otherwise.
int      cipher_loop_init(void);

// Stage 1 per-entry hook. Cheap when disabled (single relaxed load + branch).
void     cipher_loop_observe(const CipherRingEntry* ev);

// 0..3 runaway score for a session fingerprint. Returns -1 if unknown/off.
int      cipher_loop_get_score(uint64_t fingerprint);

// Writes a JSON snapshot to /tmp/cipher_loop_report.json. No-op when off.
void     cipher_loop_report(void);

// Test/debug counters.
unsigned cipher_loop_session_count(void);
unsigned cipher_loop_runaway_count(void);

// OP 29 — aggregate runaway-detection telemetry.
typedef struct CipherLoopStats {
    int      enabled;
    unsigned session_count;
    unsigned runaway_count;
    unsigned active_sessions;
    unsigned max_score_seen;
    unsigned demoted_sessions;
    uint64_t current_fingerprint;
} CipherLoopStats;

int  cipher_loop_stats(CipherLoopStats* out);

#ifdef __cplusplus
}
#endif
