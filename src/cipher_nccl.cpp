// CIPHER — L2.4 + L2.5 + L2.6: NCCL Orchestration Implementation
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_nccl.h"
#include "cipher_nccl_neural.h"
#include "cipher.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <atomic>
#include <mutex>

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
// RT bridge — called from hook shim via dlsym
// ---------------------------------------------------------------------------

static CipherNcclOrchestrator g_nccl_orch = {};
static bool                   g_nccl_orch_init = false;

// Change 4 — wire the CfC neural policy into the live intercept path.
// All state is process-global and protected by g_nccl_neural_mtx. Lazy-
// initialised on first decide call so the hot path doesn't pay init cost
// until an AllReduce actually fires.
static CipherNcclNeuralState  g_nccl_neural      = {};
static bool                   g_nccl_neural_init = false;
static std::mutex             g_nccl_neural_mtx;

// Per-bucket EMA of observed latency — used as "baseline" when the CfC
// recommends the same algo it recommended last time (lets us compute a
// stable improvement number over time).
struct BucketStats {
    double   ema_dur_ns;       // exponentially smoothed duration
    uint64_t calls;
    int      last_algo;        // -1 = never seen
};
static constexpr int   NUM_BUCKETS = 7;
// Bucket upper bounds (bytes):  <=1K, <=64K, <=1M, <=16M, <=128M, <=1G, >1G
static const uint64_t  BUCKET_UB[NUM_BUCKETS] = {
    1024ULL,
    64ULL * 1024,
    1024ULL * 1024,
    16ULL * 1024 * 1024,
    128ULL * 1024 * 1024,
    1024ULL * 1024 * 1024,
    UINT64_MAX
};
static BucketStats g_buckets[NUM_BUCKETS] = {};

// Global telemetry counters (lock-free, for fast reads from the test)
static std::atomic<uint64_t> g_decide_count   {0};
static std::atomic<uint64_t> g_feedback_count {0};
static std::atomic<uint64_t> g_last_algo_chosen {4};  // NCCL_ALGO_AUTO default
static std::atomic<uint64_t> g_last_bytes       {0};

static int bucket_index_for(uint64_t bytes) {
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (bytes <= BUCKET_UB[i]) return i;
    }
    return NUM_BUCKETS - 1;
}

static void ensure_nccl_state_init_locked() {
    if (!g_nccl_orch_init) {
        extern CipherRuntime g_cipher;
        if (g_cipher.liquid.initialized) {
            cipher_nccl_init(&g_nccl_orch, &g_cipher.liquid);
        }
        g_nccl_orch_init = true;
    }
    if (!g_nccl_neural_init) {
        cipher_nccl_neural_init(&g_nccl_neural);
        for (int i = 0; i < NUM_BUCKETS; i++) {
            g_buckets[i].ema_dur_ns = 0.0;
            g_buckets[i].calls      = 0;
            g_buckets[i].last_algo  = -1;
        }
        g_nccl_neural_init = true;
    }
}

// ---------------------------------------------------------------------------
// Part A — policy DECIDE: called from the ncclAllReduce shim BEFORE the real
// collective runs. Returns the CfC-recommended algorithm enum. The shim is
// free to ignore the return (observation mode) or — in future — pass it to
// the NCCL tuner plugin for actual algorithm substitution.
// ---------------------------------------------------------------------------

extern "C"
int cipher_nccl_record_decide(uint64_t bytes, uint32_t num_ranks) {
    std::lock_guard<std::mutex> lock(g_nccl_neural_mtx);
    ensure_nccl_state_init_locked();

    extern CipherRuntime g_cipher;
    CipherLiquidStateMgr* liquid = g_cipher.liquid.initialized
                                    ? &g_cipher.liquid : nullptr;

    CipherNcclMsgCtx ctx = {
        /* msg_size_bytes    */ (size_t)bytes,
        /* num_ranks         */ (num_ranks > 0 ? num_ranks : 8),
        /* nvlink_util       */ (liquid && liquid->device)
                                 ? liquid->device->hw.nvlink_utilization : 0.3f,
        /* nccl_ema_ns       */ (liquid && liquid->device)
                                 ? liquid->device->nccl.ema_duration_ns : 0.0f,
        /* is_gradient_sync  */ true
    };

    CipherNcclPolicy p = cipher_nccl_neural_decide(&g_nccl_neural, &ctx, liquid);

    g_decide_count.fetch_add(1, std::memory_order_relaxed);
    g_last_algo_chosen.store((uint64_t)p.algo, std::memory_order_relaxed);
    g_last_bytes.store(bytes, std::memory_order_relaxed);

    static uint64_t s_log = 0;
    s_log++;
    if (s_log <= 8 || (s_log % 200) == 0) {
        // Name table indexed by CipherNcclAlgo enum (from cipher_nccl_bpf.h):
        //   0=AUTO 1=RING 2=TREE 3=NVLS 4=LL128
        static const char* names[] = {"AUTO","RING","TREE","NVLS","LL128"};
        int a = (int)p.algo;
        if (a < 0 || a > 4) a = 0;
        fprintf(stderr,
            "[CIPHER NCCL-CfC] DECIDE bytes=%llu ranks=%u → algo=%s "
            "nchan=%d conf=%.3f (#%llu)\n",
            (unsigned long long)bytes, (unsigned)num_ranks,
            names[a], p.nchannels, p.confidence,
            (unsigned long long)s_log);
    }
    return (int)p.algo;
}

// ---------------------------------------------------------------------------
// Part A — policy FEEDBACK: called from the ncclAllReduce shim AFTER the
// real collective returns. Updates the per-bucket latency EMA and calls
// cipher_nccl_neural_update so the CfC's improvement_vs_static statistic
// reflects observed behaviour.
// ---------------------------------------------------------------------------

extern "C"
void cipher_nccl_record_feedback(uint64_t bytes, uint64_t dur_ns,
                                  int      chosen_algo)
{
    std::lock_guard<std::mutex> lock(g_nccl_neural_mtx);
    ensure_nccl_state_init_locked();

    int bi = bucket_index_for(bytes);
    BucketStats& b = g_buckets[bi];

    // Use the PREVIOUS EMA as the "baseline" passed to neural_update.
    // If no prior observation exists, seed the EMA with this call and
    // skip the update (no improvement to compute yet).
    double baseline_ns = b.ema_dur_ns;
    if (b.calls == 0) {
        b.ema_dur_ns = (double)dur_ns;
    } else {
        // 0.9 / 0.1 exponential smoothing
        b.ema_dur_ns = 0.9 * b.ema_dur_ns + 0.1 * (double)dur_ns;
    }
    b.calls++;
    b.last_algo = chosen_algo;

    if (baseline_ns > 0.0) {
        cipher_nccl_neural_update(&g_nccl_neural,
                                   (CipherNcclAlgo)chosen_algo,
                                   (float)dur_ns,
                                   (float)baseline_ns);
    }

    g_feedback_count.fetch_add(1, std::memory_order_relaxed);
}

// Backwards-compat record: still used by the ncclAllReduce shim when called
// without a preceding decide (e.g. Part A not yet wired in the shim path).
extern "C"
void cipher_nccl_record(uint64_t dur_ns, uint64_t bytes) {
    {
        std::lock_guard<std::mutex> lock(g_nccl_neural_mtx);
        ensure_nccl_state_init_locked();
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

// ---------------------------------------------------------------------------
// Part A — telemetry for tests
// ---------------------------------------------------------------------------

extern "C" uint64_t cipher_nccl_decide_count(void)    { return g_decide_count.load(); }
extern "C" uint64_t cipher_nccl_feedback_count(void)  { return g_feedback_count.load(); }
extern "C" uint64_t cipher_nccl_last_algo(void)       { return g_last_algo_chosen.load(); }
extern "C" uint64_t cipher_nccl_last_bytes(void)      { return g_last_bytes.load(); }

extern "C"
void cipher_nccl_neural_global_report(void) {
    std::lock_guard<std::mutex> lock(g_nccl_neural_mtx);
    if (!g_nccl_neural_init) {
        fprintf(stderr, "[CIPHER NCCL-CfC] global report: not initialised yet\n");
        return;
    }
    // Name table indexed by CipherNcclAlgo:
    //   0=AUTO 1=RING 2=TREE 3=NVLS 4=LL128
    static const char* names[] = {"AUTO","RING","TREE","NVLS","LL128"};
    fprintf(stderr,
        "[CIPHER NCCL-CfC] Global report\n"
        "  decide_count   = %llu\n"
        "  feedback_count = %llu\n",
        (unsigned long long)g_decide_count.load(),
        (unsigned long long)g_feedback_count.load());
    for (int i = 0; i < NUM_BUCKETS; i++) {
        const BucketStats& b = g_buckets[i];
        if (b.calls == 0) continue;
        const char* n = (b.last_algo >= 0 && b.last_algo <= 4)
                        ? names[b.last_algo] : "?";
        fprintf(stderr,
            "  bucket[%d] ≤ %llu B : calls=%llu ema=%.1fµs last_algo=%s\n",
            i, (unsigned long long)BUCKET_UB[i],
            (unsigned long long)b.calls, b.ema_dur_ns / 1000.0, n);
    }
    cipher_nccl_neural_report(&g_nccl_neural);
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
