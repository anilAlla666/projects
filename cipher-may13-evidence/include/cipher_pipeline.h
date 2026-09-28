// Op 27 PIPELINE — Multi-agent session correlation (Stage 1 observer).
//
// Default OFF: env var `CIPHER_PIPELINE=on`. Conforms to OP_CONTRACT.md
// invariants I1–I6 — no memcpy, no pinned memory, no CUDA calls.
//
// v1 bookkeeping: per-session bounded shape set (first 32 distinct
// params_hash values). At report time, computes pairwise Jaccard
// similarity across sessions and emits upstream/downstream edges
// (sim >= 0.5) into a pipeline-graph JSON. Reads SENSE classification
// read-only at report time for edge tagging.
//
// v2 (separate plan + approval): cross-session priority inheritance
// via ARBITRATE v2. Not in this op.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_pipeline_init(void);
void     cipher_pipeline_observe(const CipherRingEntry* ev);
unsigned cipher_pipeline_session_count(void);
unsigned cipher_pipeline_edge_count(void);
void     cipher_pipeline_report(void);

#ifdef __cplusplus
}
#endif
