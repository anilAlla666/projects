// Op 13 SENSE — Session classification from kernel timing patterns.
//
// Classifies the active session into HUMAN_INTERACTIVE / AGENT_AUTONOMOUS /
// BATCH_BACKGROUND / UNKNOWN using only ring-buffer timing and shape
// sequences. No content access. Stage 1 hook, default OFF.
//
// Default OFF: env var `CIPHER_SENSE=on` enables. When unset, the observe
// hook returns at a single relaxed atomic load (~3 cycles); no allocations,
// no threads, no syscalls.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_SESSION_UNKNOWN           = 0,
    CIPHER_SESSION_HUMAN_INTERACTIVE = 1,
    CIPHER_SESSION_AGENT_AUTONOMOUS  = 2,
    CIPHER_SESSION_BATCH_BACKGROUND  = 3,
} CipherSessionType;

// Idempotent. Reads CIPHER_SENSE env. Returns 1 if SENSE is enabled, 0 otherwise.
int  cipher_sense_init(void);

// Stage 1 per-entry hook. Cheap when disabled (single relaxed load + branch).
void cipher_sense_observe(const CipherRingEntry* ev);

// Lookup current classification for a fingerprint. Returns UNKNOWN if absent.
CipherSessionType cipher_sense_get_type(uint64_t fingerprint);

// Returns the fingerprint of the most recently active session. 0 if none.
uint64_t cipher_sense_current_session(void);

// Writes a JSON snapshot to /tmp/cipher_sense_report.json. No-op when off.
void cipher_sense_report(void);

// Test/debug — returns the number of allocated session slots.
unsigned cipher_sense_session_count(void);

#ifdef __cplusplus
}
#endif
