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

// OP 30 — per-tenant billing. Hot-path adders called from cudaMemcpy /
// cudaMalloc shims. tenant_id is the SENSE current_session() fingerprint.
void     cipher_receipt_observe_h2d(uint64_t bytes);
void     cipher_receipt_observe_d2h(uint64_t bytes);
void     cipher_receipt_observe_alloc_bytes(uint64_t bytes);

// Emit a billing-shaped JSON to `path`. One row per active tenant with
// {tenant_fp, launches, bytes_in, bytes_out, alloc_bytes, wall_ns,
//  hmac}. Returns rows written. Auto-emits to CIPHER_RECEIPT_PATH on
// shared-lib unload when set.
int      cipher_receipt_emit(const char* path);

typedef struct CipherReceiptBillingStats {
    int      enabled;
    uint64_t tenant_count;
    uint64_t total_launches;
    uint64_t total_h2d_bytes;
    uint64_t total_d2h_bytes;
    uint64_t total_alloc_bytes;
} CipherReceiptBillingStats;

int      cipher_receipt_billing_stats(CipherReceiptBillingStats* out);

#ifdef __cplusplus
}
#endif
