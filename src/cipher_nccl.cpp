// CIPHER — L2.4 + L2.5 + L2.6: NCCL Orchestration Implementation
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_nccl.h"
#include "cipher.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

// NCCLbpf proven thresholds (rule-based baseline, +27% floor)
#define NCCL_SMALL_MSG_BYTES   (4u   * 1024u * 1024u)   //  4MB threshold
#define NCCL_LARGE_MSG_BYTES   (128u * 1024u * 1024u)   // 128MB threshold
#define NCCL_NVLS_UTIL_THRESH  0.60f                     // NVLink > 60% → avoid Tree

void cipher_nccl_init(CipherNcclOrchestrator* n, CipherLiquidStateMgr* liquid) {
    memset(n, 0, sizeof(*n));
    n->liquid      = liquid;
    n->ebpf_active = false;   // eBPF requires root + kernel module; simulated here
    n->initialized = true;
    fprintf(stderr, "[CIPHER L2.4] NCCL Orchestrator init. "
                    "eBPF: %s (CPU stub mode)\n",
            n->ebpf_active ? "ACTIVE" : "simulated");
}

// L2.5: Neural NCCL policy — analytical rules derived from NCCLbpf paper
// On H100 with NVSwitch: NVLS dominates when NVLink is not congested.
// On multi-node: Tree wins for large messages; Ring for small.
NcclPolicyEntry cipher_nccl_select_policy(CipherNcclOrchestrator* n,
                                           const NcclPolicyInput* inp) {
    NcclPolicyEntry e = { NCCL_ALGO_RING, 512*1024, 4, 0, 0 };
    n->policy_decisions++;

    // Feature 1: message size determines base algorithm
    NcclAlgo algo;
    if (inp->msg_size_bytes < NCCL_SMALL_MSG_BYTES) {
        // Small: LL128 lowest latency
        algo = NCCL_ALGO_LL128;
        e.chunk_size      = 64 * 1024;
        e.pipeline_depth  = 8;
    } else if (inp->msg_size_bytes > NCCL_LARGE_MSG_BYTES) {
        // Large: Tree for bandwidth efficiency
        algo = NCCL_ALGO_TREE;
        e.chunk_size      = 8 * 1024 * 1024;
        e.pipeline_depth  = 2;
    } else {
        // Mid-range: NVLS on NVSwitch systems (single-node), Ring otherwise
        bool nvlink_congested = (inp->nvlink_utilization > NCCL_NVLS_UTIL_THRESH);
        algo = (!nvlink_congested && inp->num_nodes == 1)
                ? NCCL_ALGO_NVLS : NCCL_ALGO_RING;
        e.chunk_size      = 1 * 1024 * 1024;
        e.pipeline_depth  = 4;
    }

    // Feature 2: adjust if NVLink congestion detected from liquid state
    if (n->liquid && n->liquid->initialized) {
        float nvlink_util = n->liquid->device->hw.nvlink_utilization;
        if (nvlink_util > 0.85f && algo == NCCL_ALGO_NVLS) {
            algo = NCCL_ALGO_RING;  // Fallback under high congestion
        }
    }

    e.algo = algo;
    n->algo_counts[algo]++;

    // Write to eBPF policy map (simulated — on H100 this is a 20ns map write)
    n->policy_map = e;

    return e;
}

// L2.6: Compute-communication overlap window
// Predicts how much of the next AllReduce can be hidden behind backward compute
OverlapWindow cipher_nccl_overlap_window(CipherNcclOrchestrator* n,
                                          uint64_t msg_size_bytes) {
    OverlapWindow w = { false, 0, 0, 0.0f };
    n->overlap_opportunities++;

    // Get EMA of recent AllReduce duration from liquid state NCCL history
    float ema_ns = 0.0f;
    if (n->liquid && n->liquid->initialized) {
        ema_ns = n->liquid->device->nccl.ema_duration_ns;
    }
    // Fallback: estimate from message size + ring bandwidth
    // Ring AllReduce: 2*(N-1)/N * msg / bw. N=8 ranks, bw=200GB/s (NVLink)
    if (ema_ns < 1000.0f) {
        float bw_gbps = 200.0f;
        float bw_bps  = bw_gbps * 1e9f;
        ema_ns = (float)(2 * 7 * msg_size_bytes) / (8.0f * bw_bps) * 1e9f;
    }

    w.allreduce_est_ns = (uint64_t)ema_ns;

    // Conservative: assume 60% of AllReduce can be overlapped with compute
    // (the first 40% is pipeline setup + last-layer gradient availability)
    w.overlap_fraction  = 0.60f;
    w.compute_budget_ns = (uint64_t)(ema_ns * w.overlap_fraction);
    w.overlap_possible  = (w.compute_budget_ns > 50000ULL); // >50µs is worthwhile

    if (w.overlap_possible) {
        n->overlap_applied++;
        n->avg_overlap_fraction = 0.9f * n->avg_overlap_fraction
                                + 0.1f * w.overlap_fraction;
    }

    return w;
}

void cipher_nccl_record_completion(CipherNcclOrchestrator* n,
                                   uint64_t duration_ns,
                                   uint64_t msg_size_bytes,
                                   NcclAlgo algo_used) {
    (void)algo_used;
    if (n->liquid && n->liquid->initialized)
        cipher_liquid_record_nccl(n->liquid, duration_ns, msg_size_bytes);
}

// ---------------------------------------------------------------------------
// RT bridge — called from hook shim via dlsym("cipher_nccl_record")
// ---------------------------------------------------------------------------

static CipherNcclOrchestrator g_nccl_orch = {};
static bool g_nccl_orch_init = false;

extern "C"
void cipher_nccl_record(uint64_t dur_ns, uint64_t bytes) {
    if (!g_nccl_orch_init) {
        // Lazy init with liquid state from global cipher runtime
        extern CipherRuntime g_cipher;
        if (g_cipher.liquid.initialized) {
            cipher_nccl_init(&g_nccl_orch, &g_cipher.liquid);
        }
        g_nccl_orch_init = true;
    }

    cipher_nccl_record_completion(&g_nccl_orch, dur_ns, bytes, NCCL_ALGO_AUTO);

    static uint64_t s_count = 0;
    s_count++;
    if (s_count <= 5 || (s_count % 100) == 0) {
        fprintf(stderr, "[CIPHER NCCL] AllReduce: %llu us, %llu bytes (#%llu)\n",
                (unsigned long long)(dur_ns / 1000),
                (unsigned long long)bytes,
                (unsigned long long)s_count);
    }
}

void cipher_nccl_report(const CipherNcclOrchestrator* n) {
    const char* algo_names[] = {"Ring","Tree","NVLS","LL128","Auto"};
    fprintf(stderr,
        "[CIPHER L2.4/5/6] NCCL Orchestrator\n"
        "  Policy decisions: %lu\n"
        "  Algorithm mix:\n",
        n->policy_decisions);
    for (int i = 0; i < 5; i++)
        if (n->algo_counts[i])
            fprintf(stderr, "    %s: %lu\n", algo_names[i], n->algo_counts[i]);
    fprintf(stderr,
        "  Overlap opportunities: %lu\n"
        "  Overlap applied:       %lu\n"
        "  Avg overlap fraction:  %.1f%%\n",
        n->overlap_opportunities,
        n->overlap_applied,
        n->avg_overlap_fraction * 100.0f);
}
