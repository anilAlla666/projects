// =============================================================================
// CIPHER — L2.4 + L2.5 + L2.6: NCCL Orchestration
// cipher_nccl.h
//
// L2.4 NCCLbpf eBPF Hook:
//   Integrates userspace eBPF runtime into the NCCL plugin interface.
//   LNN 2 writes policy decisions to a shared eBPF map. eBPF reads in <20ns.
//   Atomic hot-reload: 1.07µs. Zero dropped calls.
//   Proven floor: NCCLbpf +27% AllReduce on 4–128MiB messages.
//
// L2.5 Neural NCCL Policy:
//   Replaces static NCCLbpf rule thresholds with learned policy derived from
//   the liquid state. Inputs: message_size, nvlink_utilization, PortXmitWait,
//   recent_latency. Output: algorithm selection (Ring/Tree/NVLS/LL128).
//
// L2.6 Compute-Communication Overlap:
//   Identifies AllReduce windows from liquid state NCCL history.
//   Schedules backward-pass compute to overlap with gradient communication.
//   Target: NCCL blocking time <5% of training iteration.
//
// SUCCESS CRITERIA:
//   L2.4: 27%+ AllReduce improvement on 4–128MiB messages.
//   L2.5: >27%, exceeds rule-based NCCLbpf.
//   L2.6: NCCL blocking <5% of training iteration.
// =============================================================================
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "cipher_liquid_state.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── L2.4: Algorithm selection ─────────────────────────────────────────────────
typedef enum {
    NCCL_ALGO_RING   = 0,   // Best for small messages, high latency links
    NCCL_ALGO_TREE   = 1,   // Best for large messages, bandwidth-bound
    NCCL_ALGO_NVLS   = 2,   // NVLink SHARP — best on NVSwitch systems
    NCCL_ALGO_LL128  = 3,   // Low-latency 128B protocol
    NCCL_ALGO_AUTO   = 4,   // Let NCCL decide (passthrough)
} NcclAlgo;

// ── L2.4: eBPF policy map entry ───────────────────────────────────────────────
// Written by LNN 2, read by eBPF hook in <20ns
typedef struct {
    NcclAlgo  algo;
    uint32_t  chunk_size;     // Bytes per pipeline chunk
    uint8_t   pipeline_depth; // Number of in-flight chunks
    uint8_t   _pad[3];
    uint64_t  written_ns;     // Timestamp of last write
} NcclPolicyEntry;

// ── L2.5: Policy input features ───────────────────────────────────────────────
typedef struct {
    uint64_t  msg_size_bytes;
    float     nvlink_utilization;   // From liquid state hw trajectory
    float     nvlink_tx_gbps;
    float     recent_latency_us;    // EMA of AllReduce duration
    uint32_t  num_ranks;
    uint32_t  num_nodes;
} NcclPolicyInput;

// ── L2.6: Overlap window ─────────────────────────────────────────────────────
typedef struct {
    bool     overlap_possible;
    uint64_t allreduce_est_ns;    // Predicted AllReduce duration
    uint64_t compute_budget_ns;   // How much backward compute can be hidden
    float    overlap_fraction;    // Fraction of AllReduce that can be hidden
} OverlapWindow;

// ── Combined NCCL orchestrator ────────────────────────────────────────────────
typedef struct {
    CipherLiquidStateMgr* liquid;

    // L2.4 eBPF policy map (simulated in CPU mode)
    NcclPolicyEntry  policy_map;
    bool             ebpf_active;

    // L2.5 policy stats
    uint64_t  policy_decisions;
    uint64_t  algo_counts[5];    // Per-algorithm selection count

    // L2.6 overlap stats
    uint64_t  overlap_opportunities;
    uint64_t  overlap_applied;
    float     avg_overlap_fraction;

    bool initialized;
} CipherNcclOrchestrator;

// ── Public API ────────────────────────────────────────────────────────────────

// L2.4: Init + eBPF map setup
void cipher_nccl_init(CipherNcclOrchestrator* n, CipherLiquidStateMgr* liquid);

// L2.5: Select algorithm for a given AllReduce
NcclPolicyEntry cipher_nccl_select_policy(CipherNcclOrchestrator* n,
                                           const NcclPolicyInput*  input);

// L2.6: Compute overlap window for next AllReduce
OverlapWindow   cipher_nccl_overlap_window(CipherNcclOrchestrator* n,
                                            uint64_t msg_size_bytes);

// Record completed AllReduce (feeds L2.5 learning + L2.6 prediction)
void cipher_nccl_record_completion(CipherNcclOrchestrator* n,
                                   uint64_t duration_ns,
                                   uint64_t msg_size_bytes,
                                   NcclAlgo algo_used);

void cipher_nccl_report(const CipherNcclOrchestrator* n);

#ifdef __cplusplus
}
#endif
