// =============================================================================
// CIPHER — L2.5 + L2.6: Neural NCCL Policy + Compute-Communication Overlap
// cipher_nccl_neural.h
//
// L2.5 — Neural NCCL Policy:
//   Replaces static NCCLbpf thresholds with a learned policy.
//   Inputs (from liquid state): message_size, NVLink utilization,
//     PortXmitWait, recent latency distribution.
//   Output: Ring/Tree/NVLS/LL128 selection + channel count.
//   Implemented as a lightweight CfC cell (32 inputs → 16 hidden → 4 outputs).
//   Runs on CPU in <5µs. No GPU needed for this inference.
//
// L2.6 — Compute-Communication Overlap:
//   Identifies AllReduce windows from liquid state NCCL history.
//   Schedules backward-pass compute to overlap with gradient communication.
//   Eliminates NCCL blocking by pre-staging AllReduce during last FFN layer.
//
//   KEY INSIGHT: In DDP training, AllReduce of gradient bucket N can overlap
//   with backward pass compute for layer N-1. CIPHER predicts when each
//   AllReduce will complete (from NCCL EMA) and schedules accordingly.
//
// SUCCESS CRITERIA:
//   L2.5: >27% AllReduce improvement. Exceeds rule-based NCCLbpf.
//   L2.6: NCCL blocking time <5% of training iteration.
// =============================================================================

#pragma once

#include "cipher_nccl_bpf.h"
#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// L2.5: Neural NCCL Policy — lightweight CfC inference
// ---------------------------------------------------------------------------

// Weight dimensions: 32 inputs → 16 hidden → 4 outputs
#define CIPHER_NCCL_CFC_INPUT_DIM    32
#define CIPHER_NCCL_CFC_HIDDEN_DIM   16
#define CIPHER_NCCL_CFC_OUTPUT_DIM    4   // [ring, tree, nvls, ll128] logits

typedef struct {
    // CfC weights (initialized from analytical rules, refined via online learning)
    float W_in[CIPHER_NCCL_CFC_HIDDEN_DIM][CIPHER_NCCL_CFC_INPUT_DIM];
    float W_out[CIPHER_NCCL_CFC_OUTPUT_DIM][CIPHER_NCCL_CFC_HIDDEN_DIM];
    float b_in[CIPHER_NCCL_CFC_HIDDEN_DIM];
    float b_out[CIPHER_NCCL_CFC_OUTPUT_DIM];
    float hidden[CIPHER_NCCL_CFC_HIDDEN_DIM];  // CfC liquid state (persists)

    // Stats
    uint64_t inference_count;
    float    avg_inference_us;
    float    improvement_vs_static;   // vs L2.4 rule-based baseline
    bool     initialized;
} CipherNcclNeuralState;

// Initialize neural policy. Weights set from analytical solution.
void cipher_nccl_neural_init(CipherNcclNeuralState* state);

// Neural policy inference — runs on CPU, <5µs.
// Returns refined CipherNcclPolicy (overrides L2.4 static decision).
CipherNcclPolicy cipher_nccl_neural_decide(
    CipherNcclNeuralState*     state,
    const CipherNcclMsgCtx*    ctx,
    const CipherLiquidStateMgr* liquid);

// Online learning: update weights from observed AllReduce outcome.
void cipher_nccl_neural_update(CipherNcclNeuralState* state,
                                CipherNcclAlgo         chosen_algo,
                                float                  actual_duration_ns,
                                float                  baseline_duration_ns);

void cipher_nccl_neural_report(const CipherNcclNeuralState* state);

// ---------------------------------------------------------------------------
// L2.6: Compute-Communication Overlap Scheduler
// ---------------------------------------------------------------------------

#define CIPHER_OVERLAP_MAX_BUCKETS  32   // Max gradient buckets tracked

typedef struct {
    uint64_t bucket_id;
    size_t   size_bytes;
    uint64_t allreduce_start_ns;
    float    predicted_duration_ns;  // From NCCL EMA in liquid state
    bool     overlap_scheduled;      // True if compute was overlapped
    bool     completed;
} CipherGradBucket;

typedef struct {
    CipherGradBucket  buckets[CIPHER_OVERLAP_MAX_BUCKETS];
    uint32_t          num_buckets;
    uint32_t          write_head;

    // Overlap stats
    uint64_t total_allreduces;
    uint64_t overlapped_allreduces;
    float    avg_blocking_fraction;   // Fraction of iteration spent blocking on NCCL
    float    blocking_target;         // 0.05 = 5% threshold from build plan
    bool     initialized;
} CipherOverlapState;

void cipher_overlap_init(CipherOverlapState* state);

// Called at start of each AllReduce — predicts completion time,
// signals dispatch layer to schedule next compute op.
// Returns predicted AllReduce duration in nanoseconds.
float cipher_overlap_schedule(CipherOverlapState*        state,
                              uint64_t                   bucket_id,
                              size_t                     msg_size_bytes,
                              const CipherLiquidStateMgr* liquid);

// Called when AllReduce completes — records actual duration.
void cipher_overlap_complete(CipherOverlapState* state,
                             uint64_t            bucket_id,
                             uint64_t            actual_duration_ns,
                             uint64_t            iteration_duration_ns);

void cipher_overlap_report(const CipherOverlapState* state);

#ifdef __cplusplus
}
#endif
