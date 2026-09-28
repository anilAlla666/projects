// Op 16 GUARD — KV cache privacy enforcement (Stage 1 observer).
//
// Detects cross-session reuse of identical kernel param fingerprints within a
// bounded residency window. Flags such events as potential KV / activation
// leakage between tenants.
//
// Default OFF: env var `CIPHER_GUARD=on`. Conforms to OP_CONTRACT.md I1–I6.
// v1 observer only — no masking / eviction.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_guard_init(void);
void     cipher_guard_observe(const CipherRingEntry* ev);
unsigned cipher_guard_leak_count(void);
unsigned cipher_guard_session_count(void);
void     cipher_guard_report(void);

#ifdef __cplusplus
}
#endif
