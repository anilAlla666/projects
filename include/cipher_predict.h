// Op 17 PREDICT — Proactive L2 preloading candidate tracker (Stage 1 observer).
//
// Default OFF: env var `CIPHER_PREDICT=on`. Conforms to OP_CONTRACT.md
// invariants I1–I6 — no CUDA calls, no cudaAccessPolicyWindow.
//
// v1 bookkeeping: per-shape (params_hash) open-address hash table counts
// dispatches and short-gap reuses. At report time, emits shapes with
// hot-reuse signatures as preload-candidate JSON.
//
// v2 (Tier A, separate plan): feed candidates to cipher_l2_persist
// actuation in Stage 3. Not in this op.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_predict_init(void);
void     cipher_predict_observe(const CipherRingEntry* ev);
unsigned cipher_predict_candidate_count(void);
unsigned cipher_predict_shape_count(void);
void     cipher_predict_report(void);

// Stage 3 instrumentation — track (pointer, bytes) reuse so the persistence
// engine can be told which regions are hot. Called from the hook (cuBLAS
// shim path). Cheap: open-address probe on a fixed-size table, no allocs.
// When a pointer crosses HOT_PTR_COUNT, it is registered with the persist
// engine via dlsym. Default OFF unless CIPHER_PREDICT=on.
void     cipher_predict_observe_ptr(void* ptr, unsigned long bytes);
unsigned cipher_predict_hot_ptr_count(void);

#ifdef __cplusplus
}
#endif
