// CIPHER Persistence Engine — Stage 3
//
// L2 cache persistence policy management. Tracks "hot regions" (pointer +
// size + score) and admits as many as fit in the silicon-reported
// `l2_persist_max` budget via fractional knapsack on freq-score density.
// Provides per-region CUaccessPolicyWindow generation and stream-level
// application for callers that hold a CUDA stream.
//
// Hot-path safety: the get_window lookup is bounded (≤ MAX_REGIONS linear
// scan) and lock-free against register/unregister/recompute, which take a
// mutex internally. Caller is responsible for not calling those mutating
// functions from the launch hot path.
//
// Default OFF (env CIPHER_PERSIST_ENGINE=on/1/ON). All entry points are
// no-ops until enabled, conforming to OP_CONTRACT I1.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#  include <cuda_runtime.h>
   extern "C" {
#endif

#define CIPHER_PERSIST_MAX_REGIONS 64

typedef struct CipherPersistEngineStats {
    int      registered_count;
    int      admitted_count;
    size_t   admitted_bytes;
    size_t   budget_bytes;
    uint64_t register_calls;
    uint64_t unregister_calls;
    uint64_t window_lookups;
    uint64_t window_hits;
    uint64_t recompute_calls;
    uint64_t apply_to_stream_calls;
    uint64_t l2_resets;
} CipherPersistEngineStats;

// Idempotent. Returns 1 on success, 0 if disabled / silicon not ready.
int cipher_persist_engine_init(void);

// 1 iff env-enabled and initialized.
int cipher_persist_engine_enabled(void);

// Register / unregister a hot region. freq_score is caller-defined (PREDICT
// will use observed reuse rate; tests use uniform 1.0). Re-registering an
// existing pointer updates bytes and freq_score in place. Returns 1 on
// success, 0 if table is full / engine disabled / inputs invalid.
int cipher_persist_engine_register(void* ptr, size_t bytes, double freq_score);
int cipher_persist_engine_unregister(void* ptr);

// Recompute admit_fraction across all registered regions using fractional
// knapsack on (freq_score / bytes). Bounded by CIPHER_PERSIST_MAX_REGIONS.
void cipher_persist_engine_recompute_budget(void);

#ifdef __cplusplus
// Build a cudaAccessPolicyWindow for `ptr` if it is a registered + admitted
// region. Returns 1 on hit, 0 on miss.
int cipher_persist_engine_get_window(void* ptr, cudaAccessPolicyWindow* out);

// Pick the top admitted region (by freq_score × admit_fraction) and fill
// `out` with its window. Cheap on the hot path — atomic snapshot, no mutex.
// Returns 1 on hit, 0 if no admitted region available.
int cipher_persist_engine_get_top_window(cudaAccessPolicyWindow* out);
#endif

// ABI-stable variant for callers in the hook DSO that don't want to pull in
// cuda_runtime.h. Fills out a flat 32-byte buffer with the same layout as
// CUaccessPolicyWindow / cudaAccessPolicyWindow.
//   bytes 0..7:  base_ptr (void*)
//   bytes 8..15: num_bytes (size_t)
//   bytes 16..19: hitRatio (float)
//   bytes 20..23: hitProp  (int = 2 = persisting)
//   bytes 24..27: missProp (int = 1 = streaming)
//   bytes 28..31: padding to 32
// Returns 1 on hit, 0 on miss.
int cipher_persist_engine_get_top_window_raw(void* out_32bytes);

// Apply the top admitted region as the stream-level access-policy window
// (cudaStreamSetAttribute / cudaStreamAttributeAccessPolicyWindow). Returns
// 1 if applied, 0 otherwise. Off the hot path; safe to call once per
// stream lifetime.
int cipher_persist_engine_apply_to_stream(void* stream_handle);

// Force eviction via cudaCtxResetPersistingL2Cache. Off hot path.
void cipher_persist_engine_reset_l2(void);

// Snapshot stats. Returns 1 on success.
int cipher_persist_engine_stats(CipherPersistEngineStats* out);

// Emit JSON to /tmp/cipher_persist_engine_report.json. Off hot path.
void cipher_persist_engine_report(void);

#ifdef __cplusplus
} // extern "C"
#endif
