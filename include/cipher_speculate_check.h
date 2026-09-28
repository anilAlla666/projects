// OP 15 — SPECULATE_CHECK kernel output cache.
//
// Caches (params_hash → predicted output_hash) from kernels that have been
// observed once. On a subsequent invocation with the same params_hash, the
// cached output is the speculative answer; downstream consumers may skip
// re-execution and verify lazily.
//
// Default OFF: env CIPHER_SPECULATE_CHECK=on.
//   record(params_hash, output_hash) — call AFTER kernel completes.
//   lookup(params_hash) -> output_hash (0 = miss).
//
// Lock-free open-address probe; no allocs on hot path.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_speculate_check_init(void);
void     cipher_speculate_check_record(uint64_t params_hash, uint64_t output_hash);
uint64_t cipher_speculate_check_lookup(uint64_t params_hash);

typedef struct CipherSpeculateCheckStats {
    int      enabled;
    uint32_t entries;
    uint64_t records;
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t mismatches;     // record() with same params, different output
    uint32_t capacity;
} CipherSpeculateCheckStats;

int  cipher_speculate_check_stats(CipherSpeculateCheckStats* out);

#ifdef __cplusplus
}
#endif
