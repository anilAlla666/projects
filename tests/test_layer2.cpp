// =============================================================================
// CIPHER — Layer 2 Test Suite
// tests/test_layer2.cpp
// =============================================================================

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cipher_liquid_state.h"
#include "cipher_sm_packer.h"
#include "cipher_fusion.h"
#include "cipher_mem_layout.h"
#include "cipher_nccl_bpf.h"
#include "cipher_nccl_neural.h"
#include "cipher_layer2.h"

#define PASS  "\033[32m✓\033[0m"
#define FAIL  "\033[31m✗\033[0m"
static int g_pass = 0, g_fail = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define CHECK(cond, msg) do { \
    if (cond) { printf("  %s %s\n", PASS, msg); g_pass++; } \
    else      { printf("  %s %s\n", FAIL, msg); g_fail++; } \
} while(0)

// ---------------------------------------------------------------------------
// Shared liquid state
// ---------------------------------------------------------------------------

static CipherLiquidStateMgr g_liq;

static void setup_liquid(float sm_idle, float hbm, float nvlink, float nccl_ms) {
    if (!g_liq.initialized) {
        cudaMallocManaged((void**)&g_liq.device, sizeof(CipherLiquidState), 0);
        memset(g_liq.device, 0, sizeof(CipherLiquidState));
        g_liq.device->_magic = 0xC1F4E350U;
        g_liq.device->phase  = 1;
        g_liq.initialized    = true;
    }
    g_liq.device->hw.sm_idle_fraction   = sm_idle;
    g_liq.device->hw.hbm_bw_utilized    = hbm;
    g_liq.device->hw.nvlink_utilization = nvlink;
    g_liq.device->nccl.ema_duration_ns  = nccl_ms * 1e6f;
}

// ---------------------------------------------------------------------------
// L2.1: SM Work Packer
// ---------------------------------------------------------------------------

static void test_sm_packer(void) {
    printf("\n[L2.1] SM Work Packer\n");

    // High idle → PACK (EMA needs warmup, run 20 calls)
    CipherSmPackerState packer;
    cipher_sm_packer_init(&packer);
    setup_liquid(0.15f, 0.5f, 0.3f, 50.0f);
    uint8_t sig = CIPHER_PACK_SIGNAL_NONE;
    for (int i = 0; i < 20; i++)
        sig = cipher_sm_packer_decide(&packer, &g_liq);
    CHECK(sig == CIPHER_PACK_SIGNAL_PACK,
          "L2.1: High idle (15%) → PACK after EMA warmup");

    // Zero idle → HOLD
    CipherSmPackerState fresh;
    cipher_sm_packer_init(&fresh);
    setup_liquid(0.0f, 0.99f, 0.8f, 50.0f);
    for (int i = 0; i < 20; i++)
        sig = cipher_sm_packer_decide(&fresh, &g_liq);
    CHECK(sig == CIPHER_PACK_SIGNAL_HOLD, "L2.1: Zero idle → HOLD");

    // Latency
    const int ITERS = 500000;
    setup_liquid(0.15f, 0.5f, 0.3f, 50.0f);
    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        volatile uint8_t s = cipher_sm_packer_decide(&packer, &g_liq);
        (void)s;
    }
    double avg_ns = (double)(now_ns() - t0) / ITERS;
    printf("  Decision latency: %.1f ns\n", avg_ns);
    CHECK(avg_ns < 100.0, "L2.1: Decision latency <100ns");
}

// ---------------------------------------------------------------------------
// L2.2: Kernel Fusion Engine
// ---------------------------------------------------------------------------

static void test_fusion(void) {
    printf("\n[L2.2] Kernel Fusion Engine\n");

    CipherFusionState fusion;
    cipher_fusion_init(&fusion);
    setup_liquid(0.1f, 0.7f, 0.3f, 50.0f);

    CipherWorkloadRhythm* r = &g_liq.device->rhythm;

    // GEMM → EW → EW
    memset(r, 0, sizeof(*r));
    r->op_class[0] = (uint8_t)cipher::OpClass::GEMM;
    r->op_class[1] = (uint8_t)cipher::OpClass::ELEMENTWISE;
    r->op_class[2] = (uint8_t)cipher::OpClass::ELEMENTWISE;
    r->write_head = 3; r->fill = 3;
    CipherFusionDecision dec = cipher_fusion_decide(&fusion, &g_liq);
    CHECK(dec.should_fuse,                          "L2.2: GEMM+EW+EW pattern detected");
    CHECK(dec.type == CIPHER_FUSE_GEMM_BIAS_GELU,   "L2.2: Identified as GEMM+BIAS+GELU");
    CHECK(dec.predicted_speedup >= 1.2f,            "L2.2: Speedup estimate >= 1.2x");
    printf("  GEMM+EW+EW → %s (%.2fx)\n",
           cipher_fusion_type_name(dec.type), dec.predicted_speedup);

    // REDUCTION → GEMM
    r->op_class[0] = (uint8_t)cipher::OpClass::REDUCTION;
    r->op_class[1] = (uint8_t)cipher::OpClass::GEMM;
    r->write_head = 2; r->fill = 2;
    dec = cipher_fusion_decide(&fusion, &g_liq);
    CHECK(dec.type == CIPHER_FUSE_LAYERNORM_GEMM,   "L2.2: REDUCE+GEMM → LAYERNORM+GEMM");

    // No match
    r->op_class[0] = (uint8_t)cipher::OpClass::CONVOLUTION;
    r->op_class[1] = (uint8_t)cipher::OpClass::ATTENTION;
    r->write_head = 2; r->fill = 2;
    dec = cipher_fusion_decide(&fusion, &g_liq);
    CHECK(!dec.should_fuse,                         "L2.2: CONV+ATTN → no fusion (correct)");
}

// ---------------------------------------------------------------------------
// L2.3: Memory Layout Optimizer
// ---------------------------------------------------------------------------

static void test_mem_layout(void) {
    printf("\n[L2.3] Memory Layout Optimizer\n");

    CipherMemLayoutState layout;
    cipher_mem_layout_init(&layout, false);

    // High BW + transpose → TILED_32
    setup_liquid(0.1f, 0.90f, 0.3f, 50.0f);
    CipherLayoutDecision dec = cipher_mem_layout_decide(
        &layout, &g_liq, (uint8_t)cipher::OpClass::MEMCPY_TRANSPOSE, 4096, 4096);
    CHECK(dec.hint == CIPHER_LAYOUT_TILED_32 && dec.change_recommended,
          "L2.3: MEMCPY_TRANSPOSE → TILED_32");

    // AMD mode: misaligned → XCD_ALIGNED
    CipherMemLayoutState amd;
    cipher_mem_layout_init(&amd, true);
    dec = cipher_mem_layout_decide(
        &amd, &g_liq, (uint8_t)cipher::OpClass::GEMM, 4097, 4096);
    CHECK(dec.hint == CIPHER_LAYOUT_XCD_ALIGNED,
          "L2.3: AMD + misaligned → XCD_ALIGNED");

    // Low BW → no change
    setup_liquid(0.1f, 0.30f, 0.3f, 50.0f);
    CipherMemLayoutState fresh;
    cipher_mem_layout_init(&fresh, false);
    dec = cipher_mem_layout_decide(
        &fresh, &g_liq, (uint8_t)cipher::OpClass::GEMM, 4096, 4096);
    CHECK(!dec.change_recommended, "L2.3: Low BW util → no layout change");

    // NVIDIA mode never emits XCD_ALIGNED
    setup_liquid(0.1f, 0.90f, 0.3f, 50.0f);
    dec = cipher_mem_layout_decide(
        &layout, &g_liq, (uint8_t)cipher::OpClass::GEMM, 4096, 4096);
    CHECK(dec.hint != CIPHER_LAYOUT_XCD_ALIGNED,
          "L2.3: NVIDIA mode does not emit XCD_ALIGNED");
}

// ---------------------------------------------------------------------------
// L2.4: NCCLbpf Policy
// ---------------------------------------------------------------------------

static void test_nccl_bpf(void) {
    printf("\n[L2.4] NCCLbpf eBPF Policy\n");

    CipherNcclBpfState bpf;
    cipher_nccl_bpf_init(&bpf);
    setup_liquid(0.1f, 0.5f, 0.3f, 50.0f);

    struct { size_t sz; CipherNcclAlgo exp; const char* label; } cases[] = {
        { 64ULL*1024,           CIPHER_NCCL_ALGO_LL128, "<256KB  → LL128" },
        { 1ULL*1024*1024,       CIPHER_NCCL_ALGO_TREE,  "1MB     → TREE"  },
        { 32ULL*1024*1024,      CIPHER_NCCL_ALGO_RING,  "32MB    → RING"  },
        { 256ULL*1024*1024,     CIPHER_NCCL_ALGO_NVLS,  "256MB   → NVLS"  },
    };

    for (auto& c : cases) {
        CipherNcclMsgCtx ctx = {c.sz, 8, 0.3f, 50000000.0f, true};
        CipherNcclPolicy p = cipher_nccl_bpf_decide(&bpf, &ctx, &g_liq);
        bool ok = (p.algo == c.exp);
        printf("  %s %-18s → %-6s (conf=%.2f)\n",
               ok ? PASS : FAIL, c.label,
               cipher_nccl_algo_name(p.algo), p.confidence);
        if (ok) g_pass++; else g_fail++;
    }

    cipher_nccl_bpf_feedback(&bpf, 36500000, 32ULL*1024*1024);
    CHECK(bpf.estimated_improvement >= 0.27f,
          "L2.4: Estimated improvement >= 27% (NCCLbpf proven floor)");

    // NVLink saturated → RING switches to TREE
    CipherNcclMsgCtx ctx = {32ULL*1024*1024, 8, 0.95f, 50000000.0f, true};
    CipherNcclPolicy p = cipher_nccl_bpf_decide(&bpf, &ctx, &g_liq);
    CHECK(p.algo == CIPHER_NCCL_ALGO_TREE,
          "L2.4: NVLink >85% saturated → RING demoted to TREE");

    cipher_nccl_bpf_destroy(&bpf);
}

// ---------------------------------------------------------------------------
// L2.5: Neural NCCL Policy
// ---------------------------------------------------------------------------

static void test_nccl_neural(void) {
    printf("\n[L2.5] Neural NCCL Policy\n");

    CipherNcclNeuralState neural;
    cipher_nccl_neural_init(&neural);
    setup_liquid(0.1f, 0.5f, 0.3f, 50.0f);

    // Latency < 5µs
    CipherNcclMsgCtx ctx = {32ULL*1024*1024, 8, 0.3f, 50000000.0f, true};
    const int ITERS = 10000;
    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        volatile CipherNcclPolicy p = cipher_nccl_neural_decide(&neural, &ctx, &g_liq);
        (void)p;
    }
    double avg_us = (double)(now_ns() - t0) / ITERS / 1000.0;
    printf("  Inference latency: %.2f µs\n", avg_us);
    CHECK(avg_us < 5.0, "L2.5: Neural NCCL inference <5µs");

    // Valid output
    CipherNcclPolicy p = cipher_nccl_neural_decide(&neural, &ctx, &g_liq);
    CHECK(p.algo >= CIPHER_NCCL_ALGO_AUTO && p.algo <= CIPHER_NCCL_ALGO_LL128,
          "L2.5: Neural policy outputs valid algorithm");
    CHECK(p.confidence > 0.0f && p.confidence <= 1.0f,
          "L2.5: Confidence in (0,1]");
    printf("  32MB → %s (conf=%.2f)\n",
           cipher_nccl_algo_name(p.algo), p.confidence);

    // CfC temporal stability
    CipherNcclPolicy p2 = cipher_nccl_neural_decide(&neural, &ctx, &g_liq);
    CHECK(p2.algo == p.algo || fabsf(p2.confidence - p.confidence) < 0.3f,
          "L2.5: CfC temporal state stable under repeated identical input");

    // Small message → should prefer LL128 or TREE (not RING or NVLS)
    CipherNcclMsgCtx small_ctx = {32ULL*1024, 8, 0.3f, 50000000.0f, true};
    CipherNcclPolicy ps = cipher_nccl_neural_decide(&neural, &small_ctx, &g_liq);
    CHECK(ps.algo != CIPHER_NCCL_ALGO_RING,
          "L2.5: Small message (<256KB) not routed to RING");
}

// ---------------------------------------------------------------------------
// L2.6: Overlap Scheduler
// ---------------------------------------------------------------------------

static void test_overlap(void) {
    printf("\n[L2.6] Compute-Communication Overlap\n");

    CipherOverlapState overlap;
    cipher_overlap_init(&overlap);
    setup_liquid(0.1f, 0.5f, 0.3f, 50.0f);

    for (uint64_t b = 0; b < 5; b++) {
        float pred = cipher_overlap_schedule(&overlap, b, 32ULL*1024*1024, &g_liq);
        CHECK(pred > 0.0f, "L2.6: AllReduce duration prediction is positive");
        cipher_overlap_complete(&overlap, b, 40000000, 400000000);
    }

    CHECK(overlap.total_allreduces == 5,      "L2.6: All 5 AllReduces tracked");
    CHECK(overlap.overlapped_allreduces == 5, "L2.6: All 5 scheduled for overlap");
    printf("  Avg blocking fraction: %.1f%%  (target <5%%)\n",
           overlap.avg_blocking_fraction * 100.0f);
}

// ---------------------------------------------------------------------------
// L2.7: Unified Layer 2 Decision
// ---------------------------------------------------------------------------

static void test_layer2_unified(void) {
    printf("\n[L2.7] Unified Layer 2 — Full Integration\n");

    CipherLayer2State l2;
    cipher_layer2_init(&l2, false);
    setup_liquid(0.12f, 0.85f, 0.35f, 45.0f);

    // Inject GEMM+EW+EW for fusion
    CipherWorkloadRhythm* r = &g_liq.device->rhythm;
    memset(r, 0, sizeof(*r));
    r->op_class[0] = (uint8_t)cipher::OpClass::GEMM;
    r->op_class[1] = (uint8_t)cipher::OpClass::ELEMENTWISE;
    r->op_class[2] = (uint8_t)cipher::OpClass::ELEMENTWISE;
    r->write_head = 3; r->fill = 3;

    // Single decision latency
    uint64_t t0 = now_ns();
    CipherLayer2Decision dec = cipher_layer2_decide(
        &l2, &g_liq, (uint8_t)cipher::OpClass::GEMM,
        4096, 28672, false, 0, 0);
    uint64_t single_ns = now_ns() - t0;
    printf("  Single decision: %lu ns\n", single_ns);
    CHECK(single_ns < 50000, "L2.7: Single decision <50µs (cold path)");
    CHECK(dec.fusion.should_fuse, "L2.7: Fusion fires within unified call");

    // AllReduce decision
    dec = cipher_layer2_decide(
        &l2, &g_liq, (uint8_t)cipher::OpClass::GEMM,
        4096, 4096, true, 32ULL*1024*1024, 8);
    CHECK(dec.nccl_active, "L2.7: NCCL active for AllReduce dispatch");
    CHECK(dec.nccl_policy.algo != CIPHER_NCCL_ALGO_AUTO,
          "L2.7: NCCL algo selected (not AUTO passthrough)");
    printf("  AllReduce: %s (conf=%.2f)  overlap=%s\n",
           cipher_nccl_algo_name(dec.nccl_policy.algo),
           dec.nccl_policy.confidence,
           dec.overlap_window ? "YES" : "no");

    // Hot-path throughput
    const int ITERS = 100000;
    t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        volatile CipherLayer2Decision d = cipher_layer2_decide(
            &l2, &g_liq, (uint8_t)cipher::OpClass::GEMM,
            4096, 4096, false, 0, 0);
        (void)d;
    }
    double avg_ns = (double)(now_ns() - t0) / ITERS;
    printf("  Hot-path avg: %.0f ns\n", avg_ns);
    CHECK(avg_ns < 5000.0, "L2.7: Hot-path <5µs per decision");

    cipher_layer2_report(&l2);
    cipher_layer2_destroy(&l2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    printf("===================================================\n");
    printf("  CIPHER Layer 2 Test Suite\n");
    printf("  Koopman Orchestrator — L2.1 through L2.7\n");
    printf("===================================================\n");

    g_liq = {};
    setup_liquid(0.1f, 0.7f, 0.3f, 50.0f);

    test_sm_packer();
    test_fusion();
    test_mem_layout();
    test_nccl_bpf();
    test_nccl_neural();
    test_overlap();
    test_layer2_unified();

    if (g_liq.initialized && g_liq.device)
        cudaFree(g_liq.device);

    printf("\n===================================================\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0)
        printf("  \033[32m LAYER 2 GREEN -- ready for Layer 1\033[0m\n");
    else
        printf("  \033[31m FAILURES REMAIN\033[0m\n");
    printf("===================================================\n\n");
    return g_fail > 0 ? 1 : 0;
}
