// Op 19 CONTINUITY — Incremental KV checkpoint tracking (Stage 1 observer).
//
// Default OFF: env var `CIPHER_CONTINUITY=on`. Conforms to OP_CONTRACT.md
// invariants I1–I6 — no memcpy, no pinned memory, no CUDA calls.
//
// v1 bookkeeping: for each AGENT_AUTONOMOUS session (SENSE-gated), track
// which attention regions are active. Every SNAPSHOT_EVERY attention events,
// bump a manifest counter (would-be-checkpointed marker). Emit a JSON
// manifest via cipher_continuity_report().
//
// v2 (Tier A, separate plan + approval): actual KV page capture to host
// pinned arena in a Stage 3 worker. Not in this op.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_continuity_init(void);
void     cipher_continuity_observe(const CipherRingEntry* ev);
unsigned cipher_continuity_session_count(void);
unsigned cipher_continuity_manifest_count(void);
void     cipher_continuity_report(void);

#ifdef __cplusplus
}
#endif
