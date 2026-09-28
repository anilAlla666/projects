// Op 29 COMPLY — regulatory compliance artifact bundler.
//
// Aggregates live state from RECEIPT, CARBON, GUARD, DETERMINISM,
// FAIRNESS, TOPOLOGY via their C APIs and emits a single JSON
// "compliance pack" at report time. Observer is a no-op; all work runs
// in report.
//
// Default OFF: env var `CIPHER_COMPLY=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int  cipher_comply_init(void);
void cipher_comply_observe(const CipherRingEntry* ev);
int  cipher_comply_ok(void);
void cipher_comply_report(void);

// OP 31 — streaming audit JSONL.
//
// Append-only JSON-lines log of privileged events.
// Activated by env: CIPHER_COMPLY_AUDIT_PATH=<path>. One event per line,
// flushed per line. Hot-path-safe (single write() syscall per event).
//
// Event kinds (suggested, but the kind string is free-form):
//   alloc          {bytes, ptr}
//   alloc_blocked  {bytes, cap, in_flight}
//   free           {ptr}
//   dvfs_set       {target_mhz, rc}
//   dvfs_blocked   {rc, reason}
//   fp8_disabled   {torch_native}
//   kv_redirect    {layer, ctx_len, bytes}
//   nccl_tuner     {algo, proto, nvlink, ib}
//
// Returns 1 if written, 0 if disabled / open failure.
int cipher_comply_audit(const char* kind, const char* json_kv);

typedef struct CipherComplyAuditStats {
    int      enabled;
    int      armed;          // CIPHER_COMPLY_AUDIT_PATH set
    uint64_t events_written;
    uint64_t bytes_written;
    uint64_t open_failures;
} CipherComplyAuditStats;

int cipher_comply_audit_stats(CipherComplyAuditStats* out);

#ifdef __cplusplus
}
#endif
