// =============================================================================
// CIPHER — L2.4: NCCLbpf eBPF Hook
// cipher_nccl_bpf.h
//
// Integrates userspace eBPF runtime into NCCL's plugin interface.
// LNN 2 writes its policy decision to a shared eBPF map.
// eBPF program reads the map and selects the AllReduce algorithm in <20ns.
//
// MECHANISM (from NCCLbpf paper, proven +27% floor):
//   1. CIPHER installs an eBPF program into NCCL's plugin hook at startup
//   2. For each AllReduce call, NCCL invokes the plugin before selecting algo
//   3. Plugin reads from shared BPF_MAP_TYPE_ARRAY (key=0: policy decision)
//   4. Returns: RING / TREE / NVLS / LL128 selection
//   5. CIPHER updates the map atomically: hot-reload latency 1.07µs
//
// IN CPU STUB MODE:
//   eBPF is Linux kernel feature — not available in CPU-only test.
//   Entire eBPF path is stubbed. Logic, policy selection, and stats are real.
//   Only the actual kernel eBPF load/attach is a no-op.
//
// SUCCESS CRITERION: 27%+ AllReduce improvement on 4-128MiB messages.
// DEPENDENCY: F4 (Liquid state for NCCL history), F5 (NVLink telemetry).
// =============================================================================

#pragma once

#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// NCCL algorithm selection (matches NCCL internal enum)
typedef enum {
    CIPHER_NCCL_ALGO_AUTO   = 0,   // Let NCCL decide (passthrough)
    CIPHER_NCCL_ALGO_RING   = 1,   // Ring AllReduce — best for large messages
    CIPHER_NCCL_ALGO_TREE   = 2,   // Tree AllReduce — best for small messages
    CIPHER_NCCL_ALGO_NVLS   = 3,   // NVLink SHARP — best for NVSwitch clusters
    CIPHER_NCCL_ALGO_LL128  = 4,   // Low-latency 128B protocol — <256KB msgs
} CipherNcclAlgo;

// Protocol selection (inner loop of the algorithm)
typedef enum {
    CIPHER_NCCL_PROTO_AUTO  = 0,
    CIPHER_NCCL_PROTO_LL    = 1,   // Low-latency
    CIPHER_NCCL_PROTO_LL128 = 2,   // Low-latency 128B
    CIPHER_NCCL_PROTO_SIMPLE= 3,   // Simple (large messages)
} CipherNcclProto;

// Policy decision written to eBPF map
typedef struct {
    CipherNcclAlgo   algo;
    CipherNcclProto  proto;
    uint32_t         nchannels;       // Number of NCCL channels to use
    float            confidence;      // 0-1 confidence in this selection
} CipherNcclPolicy;

// NCCLbpf state
typedef struct {
    bool             bpf_loaded;      // True if eBPF prog successfully loaded
    bool             initialized;
    bool             stub_mode;       // True in CPU-only builds

    // Policy decision stats
    uint64_t         total_decisions;
    uint64_t         algo_counts[5];  // Per-algorithm selection counts
    float            avg_msg_size_mb; // Running average message size

    // Performance tracking
    uint64_t         allreduce_count;
    float            estimated_improvement; // vs NCCL default
} CipherNcclBpfState;

// Message metadata passed to policy decision
typedef struct {
    size_t   msg_size_bytes;
    uint32_t num_ranks;
    float    nvlink_util;       // From liquid state hw trajectory
    float    nccl_ema_ns;       // Recent AllReduce duration EMA
    bool     is_gradient_sync;  // True for DDP gradient allreduce
} CipherNcclMsgCtx;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize NCCLbpf. Attempts eBPF load; falls back to policy-only on failure.
int cipher_nccl_bpf_init(CipherNcclBpfState* state);

// Core policy decision — given message context, return algorithm selection.
// In full mode: also writes to eBPF map (<20ns for NCCL to read).
// In stub mode: returns the decision, skips the map write.
CipherNcclPolicy cipher_nccl_bpf_decide(CipherNcclBpfState*        state,
                                         const CipherNcclMsgCtx*    ctx,
                                         const CipherLiquidStateMgr* liquid);

// Called after AllReduce completes — records actual duration for feedback.
void cipher_nccl_bpf_feedback(CipherNcclBpfState* state,
                               uint64_t            actual_duration_ns,
                               size_t              msg_size_bytes);

void cipher_nccl_bpf_destroy(CipherNcclBpfState* state);

const char* cipher_nccl_algo_name(CipherNcclAlgo algo);

void cipher_nccl_bpf_report(const CipherNcclBpfState* state);

#ifdef __cplusplus
}
#endif
