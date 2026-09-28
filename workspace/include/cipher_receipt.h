// Op 18 RECEIPT — per-session signed proof of compute.
//
// Per-session FNV-64 dispatch-chain + launch counter, HMAC-SHA256 signed
// at report time with an ephemeral (or env-provided) key. Observer is
// lock-free and does no crypto. Crypto runs in report (single-threaded).
//
// Default OFF: env var `CIPHER_RECEIPT=on`. Requires SENSE.
//   CIPHER_RECEIPT_KEY=<64-hex-chars> — optional fixed key for repro tests.
// Contract I1–I6 (crypto is off the hot path).
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_receipt_init(void);
void     cipher_receipt_observe(const CipherRingEntry* ev);
unsigned cipher_receipt_session_count(void);
void     cipher_receipt_report(void);

#ifdef __cplusplus
}
#endif
