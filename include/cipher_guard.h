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
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_guard_init(void);
void     cipher_guard_observe(const CipherRingEntry* ev);
unsigned cipher_guard_leak_count(void);
unsigned cipher_guard_session_count(void);
void     cipher_guard_report(void);

// OP 16 — memory bounds enforcement.
//
// Called from the cudaMalloc / cudaFree intercepts. Tracks cumulative
// bytes-in-flight and (if CIPHER_GUARD_BYTES_CAP=N is set) refuses
// allocations that would push current+size over the cap.
//
// alloc_admit() returns 1 if the alloc may proceed, 0 if it must be
// denied (caller returns cudaErrorMemoryAllocation). Callers MUST call
// alloc_record(ptr,size) AFTER a successful alloc, and free_record(ptr)
// BEFORE freeing — otherwise stats will drift.
//
// All entry points are no-ops when GUARD is disabled, so the cudaMalloc
// hot-path overhead is one branch.
int      cipher_guard_alloc_admit(uint64_t size);
void     cipher_guard_alloc_record(void* ptr, uint64_t size);
void     cipher_guard_free_record(void* ptr);

typedef struct CipherGuardMemoryStats {
    int      enabled;
    int      enforce;             // 1 if CIPHER_GUARD_BYTES_CAP set
    uint64_t bytes_cap;
    uint64_t allocs_total;
    uint64_t frees_total;
    uint64_t bytes_in_flight;
    uint64_t peak_bytes;
    uint64_t allocs_blocked;
    uint64_t frees_unmatched;
} CipherGuardMemoryStats;

int      cipher_guard_memory_stats(CipherGuardMemoryStats* out);

#ifdef __cplusplus
}
#endif
