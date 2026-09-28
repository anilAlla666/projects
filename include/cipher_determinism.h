// Op 21 DETERMINISM — reproducible dispatch-sequence fingerprint.
//
// Stage 1 observer — folds the ordered `params_hash` stream into a running
// 64-bit FNV-mix hash. For identical deterministic workloads (same seed,
// same shapes, same call order) the reported hash matches across reruns.
//
// Default OFF: env var `CIPHER_DETERMINISM=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_determinism_init(void);
void     cipher_determinism_observe(const CipherRingEntry* ev);
uint64_t cipher_determinism_hash(void);
uint64_t cipher_determinism_count(void);
void     cipher_determinism_report(void);

// OP 17 — record / verify the dispatch fingerprint to / against a file.
//
// record(path): writes JSON {hash, count, ts} to path. Returns 1 on success.
// verify(path): reads JSON, compares hash + count to current values.
//               Returns 1 if match, 0 if mismatch, -1 if read error.
//
// Auto-record / auto-verify env knobs:
//   CIPHER_DETERMINISM_RECORD=<path>  -- snapshot on shared-lib unload.
//   CIPHER_DETERMINISM_VERIFY=<path>  -- compare on unload; if mismatch,
//                                        sets exit status via _exit(EXIT_VERIFY_FAIL).
//   CIPHER_DETERMINISM_VERIFY_FATAL   -- if set, mismatch aborts with non-zero
//                                        exit; otherwise the failure is logged.
int      cipher_determinism_record(const char* path);
int      cipher_determinism_verify(const char* path);

typedef struct CipherDeterminismVerifyStats {
    int      enabled;
    int      record_armed;        // CIPHER_DETERMINISM_RECORD set
    int      verify_armed;        // CIPHER_DETERMINISM_VERIFY set
    int      fatal_on_fail;
    uint64_t hash;                // current
    uint64_t count;
    uint64_t expected_hash;       // 0 if not loaded
    uint64_t expected_count;
    int      verify_runs;
    int      verify_passes;
    int      verify_fails;
} CipherDeterminismVerifyStats;

int cipher_determinism_verify_stats(CipherDeterminismVerifyStats* out);

#ifdef __cplusplus
}
#endif
