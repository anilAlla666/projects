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
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_continuity_init(void);
void     cipher_continuity_observe(const CipherRingEntry* ev);
unsigned cipher_continuity_session_count(void);
unsigned cipher_continuity_manifest_count(void);
void     cipher_continuity_report(void);

// OP 28 — checkpoint / restore for rolling restarts.
//
// save(path):    writes a binary snapshot of the session manifest table.
// restore(path): reads a binary snapshot, populates the session table.
//
// Auto-save on shared-lib unload when CIPHER_CONTINUITY_SAVE_PATH=<path>.
// Auto-restore on init when CIPHER_CONTINUITY_RESTORE_PATH=<path>.
int  cipher_continuity_save(const char* path);
int  cipher_continuity_restore(const char* path);

typedef struct CipherContinuityStats {
    int      enabled;
    unsigned saves_done;
    unsigned restores_done;
    uint64_t bytes_written;
    uint64_t bytes_read;
    unsigned sessions_saved;
    unsigned sessions_restored;
} CipherContinuityStats;

int  cipher_continuity_stats(CipherContinuityStats* out);

#ifdef __cplusplus
}
#endif
