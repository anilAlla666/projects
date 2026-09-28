/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_koopman_engine.cpp -- W14 Step 2 B.2 Koopman engine.
 *
 * Second actuator on the cipher_rt_matmul_dispatch substrate. Registers a
 * maybe_handle() callback that checks FP16 dtype + .cu shape-registry
 * membership, then dispatches cipher_koopman_fp16_launch_shape() instead of
 * cublasGemmEx for calibrated narrow-domain shapes.
 *
 * Three guards collapse into the .cu shape registry check (see header):
 *   1. registry-hit (model-keyed)
 *   2. shape in seeded narrow-domain set
 *   3. model_uuid match
 *
 * Calibration is driven externally by the existing
 * edmd_live_post_relaunch_hook at cipher_dispatch.cpp:213 which feeds real
 * GEMM ptr_A/ptr_B/ptr_C into cipher_edmd_live_collect; once
 * cipher_edmd_live_collect accumulates TARGET_ROWS snapshots a background
 * thread runs randomized-SVD + calls cipher_koopman_fp16_register_shape into
 * the .cu shape registry; subsequent eligible GEMMs are routed here.
 *
 * Convention (mirrors cipher_dispatch.cpp:229):
 *   cuBLAS  (m, n, k)          weight=A, activation=B, output=C
 *   Koopman M_py = n           — batch dimension
 *   Koopman K_dim = k          — input feature dimension
 *   Koopman N_dim = m          — output feature dimension
 *   x_fp16 = call->B           — activation pointer
 *   out_fp16 = call->C         — output pointer
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <atomic>

extern "C" {
#include "cipher_rt_koopman.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_ring_write.h"
#include "cipher_workload_detect.h"  /* K.2: classifier-driven engagement */
}

/* W14 Step 3 S3.B0 — slot-3 RING_WRITE producer for REMEMBER consumer (W13-14
 * Step 2 sub-element residue per WEEK_14_STEP_2_KOOPMAN_TIER_ADDENDUM.md).
 * Weak-linked so the engine builds clean if the ring is absent in some
 * future link configuration; v1 cipher_rt_phase4 Makefile always pulls it in
 * (cipher_rt_ring_write.o at Makefile:70). */
extern "C" __attribute__((weak)) void cipher_rt_ring_write(
    uint32_t tenant_id,
    uint32_t event_type, uint32_t event_subtype,
    uint64_t commit_seq,
    const void *payload, size_t payload_len);

/* REMEMBER event payload. 40 bytes max per cipher_rt_ring_entry.payload.
 * The REMEMBER consumer (S3.B1) reads (M, K_dim, N_dim) shape + (ptr_B,
 * ptr_C) launch pointers and feeds them into cipher_lnn_decide gate logic
 * at cipher_lnn.cpp:428-458. */
struct cipher_rt_koopman_event {
    uint32_t M;
    uint32_t K_dim;
    uint32_t N_dim;
    uint32_t op_class;   /* 0 = GEMM (Koopman is matmul-class) */
    uint64_t ptr_B;
    uint64_t ptr_C;
    uint64_t reserved;
};
static_assert(sizeof(cipher_rt_koopman_event) == 40,
              "cipher_rt_koopman_event must fit in RING_WRITE payload (40 B)");

/* Strong symbol from src/may13/cipher_block_sub_kernel.cu (W14 Step 2 B.0
 * port-landmark resolves this; nm -D verified at landmark commit ce4c1b8). */
extern "C" int cipher_koopman_fp16_launch_shape(
    const void *x_fp16, void *out_fp16,
    int M, int K_dim, int N_dim);

/* W14 Step 2 E ζ-deep Day 5 β — OOD detector. Returns max residual_ratio
 * across input rows. -1.0f if shape not registered. */
extern "C" float cipher_koopman_fp16_ood_max_residual(
    const void *x_fp16, int M, int K_dim, int N_dim);

#define CUDA_R_16F  2

static std::atomic<int>   g_enabled       {0};
static std::atomic<int>   g_verbose       {0};
static std::atomic<unsigned long> g_calls_total       {0};
static std::atomic<unsigned long> g_calls_handled     {0};
static std::atomic<unsigned long> g_calls_skipped     {0};
static std::atomic<unsigned long> g_calls_ood         {0};
static std::atomic<unsigned long> g_calls_substituted {0};
static std::atomic<unsigned long> g_remember_emits    {0};   /* W14 S3.B0 */
/* Per-early-exit counters (Option-2 Step 1 α — disambiguate which
 * pre-counter early-exit returns PASSTHROUGH; surfaces FP16-vs-BF16 etc.).
 * The g_enabled early-exit deliberately has no counter — registration log
 * already proves g_enabled=1 when koopman_init succeeds. */
static std::atomic<unsigned long> g_skip_dtype       {0};
static std::atomic<unsigned long> g_skip_dim         {0};
static std::atomic<unsigned long> g_skip_nullptr     {0};
/* K.2 (2026-05-27): classifier-skip counter. Distinct from per-gate skips so
 * engagement evidence shows "skipped because workload doesn't want Koopman". */
static std::atomic<unsigned long> g_skip_by_classifier {0};
/* W.3 (2026-05-27) Sub-step 1 empirical-first: bf16 observation counter.
 * Increments when actuator sees a CUDA_R_16BF GEMM that passes classifier
 * + dimension gates. Per Anil empirical-first adjudication, the bf16 path
 * is observation-only here; substitution (cast wrapper + EDMD auto-cal +
 * \xCE\xB2-residual gate) defers to a follow-up after measured residual data
 * adjudicates the production-rank engagement question. */
static std::atomic<unsigned long> g_calls_koopman_bf16_observed {0};
static std::atomic<unsigned long> g_calls_koopman_bf16_max_m    {0};  /* track largest bf16 M seen */
static std::atomic<unsigned long> g_calls_koopman_bf16_max_k    {0};
static std::atomic<unsigned long> g_calls_koopman_bf16_max_n    {0};
static float g_ood_threshold = -1.0f;   /* -1 = no OOD check (default until env reads) */
static int   g_ood_init      = 0;

/* maybe_handle: returns HANDLED on successful Koopman substitution, else
 * PASSTHROUGH so the next actuator (or real cublasGemmEx) runs. */
static int maybe_handle_koopman(const struct cipher_rt_matmul_call *call,
                                int *out_status)
{
    if (!g_enabled.load(std::memory_order_relaxed))
        return CIPHER_RT_MATMUL_PASSTHROUGH;

    /* K.2 (2026-05-27): classifier-driven engagement gate. AFTER env-gate so
     * CIPHER_KOOPMAN=0 still forces off. UNKNOWN class falls through to the
     * legacy dtype/shape gates so warmup behavior is unchanged. Costs one
     * pointer load + uint8 read per call. */
    {
        const struct cipher_workload_profile *p = cipher_workload_profile_get();
        if (p && p->workload_class != CIPHER_WL_UNKNOWN && !p->koopman_engage) {
            g_skip_by_classifier.fetch_add(1, std::memory_order_relaxed);
            return CIPHER_RT_MATMUL_PASSTHROUGH;
        }
    }

    /* W.3 (2026-05-27) Sub-step 1: bf16 observation path. CUDA_R_16BF (14)
     * GEMMs are observed for shape statistics + passthrough. Substitution
     * (cast wrapper + EDMD auto-cal + \xCE\xB2-residual gate) defers per Anil
     * empirical-first adjudication; the data here drives the next-step
     * spec for whether \xCE\xB2-rank engagement on production LLM is feasible
     * at r\xE2\x89\xA464 (cited rank ~1773 conflict per advisor).
     *
     * fp16 path unchanged per Memory #16 ABI-additive (Memory #12 production
     * narrow-domain Koopman + \xCE\xB2=0.05 threshold preserved below). */
    if (call->Atype == 14 /* CUDA_R_16BF */) {
        if (call->Btype == 14 && call->Ctype == 14 &&
            call->m > 0 && call->n > 0 && call->k > 0) {
            unsigned long obs = g_calls_koopman_bf16_observed.fetch_add(1,
                std::memory_order_relaxed) + 1;
            /* Track max dims (lossy, fine for histogram purpose). */
            unsigned long cm = g_calls_koopman_bf16_max_m.load(std::memory_order_relaxed);
            if ((unsigned long)call->m > cm)
                g_calls_koopman_bf16_max_m.store(call->m, std::memory_order_relaxed);
            unsigned long ck = g_calls_koopman_bf16_max_k.load(std::memory_order_relaxed);
            if ((unsigned long)call->k > ck)
                g_calls_koopman_bf16_max_k.store(call->k, std::memory_order_relaxed);
            unsigned long cn = g_calls_koopman_bf16_max_n.load(std::memory_order_relaxed);
            if ((unsigned long)call->n > cn)
                g_calls_koopman_bf16_max_n.store(call->n, std::memory_order_relaxed);
            /* Log a few representative shapes for analysis. */
            if (obs == 1 || obs == 100 || obs == 10000 || obs == 100000) {
                fprintf(stderr,
                        "[CIPHER KOOPMAN-BF16-OBS] obs=%lu m=%d n=%d k=%d "
                        "(bf16 cuBLAS GEMM seen; substitution deferred)\n",
                        obs, call->m, call->n, call->k);
            }
        }
        g_skip_dtype.fetch_add(1, std::memory_order_relaxed);
        return CIPHER_RT_MATMUL_PASSTHROUGH;
    }
    /* FP16-only — the .cu kernel is FP16-in/FP16-out */
    if (call->Atype != CUDA_R_16F ||
        call->Btype != CUDA_R_16F ||
        call->Ctype != CUDA_R_16F) {
        g_skip_dtype.fetch_add(1, std::memory_order_relaxed);
        return CIPHER_RT_MATMUL_PASSTHROUGH;
    }

    /* Defensive: bail on degenerate dimensions */
    if (call->m <= 0 || call->n <= 0 || call->k <= 0) {
        g_skip_dim.fetch_add(1, std::memory_order_relaxed);
        return CIPHER_RT_MATMUL_PASSTHROUGH;
    }
    if (!call->B || !call->C) {
        g_skip_nullptr.fetch_add(1, std::memory_order_relaxed);
        return CIPHER_RT_MATMUL_PASSTHROUGH;
    }

    g_calls_total.fetch_add(1, std::memory_order_relaxed);

    /* Convention per cipher_dispatch.cpp:229 — cuBLAS (m,n,k) → Koopman
     * (N_dim=m, K_dim=k, M_py=n). */
    const int M     = call->n;
    const int K_dim = call->k;
    const int N_dim = call->m;

    /* W14 Step 2 E ζ-deep Day 5 β — OOD detector. Compute residual_ratio
     * ||x - V_x V_x^T x|| / ||x||; skip substitution if > threshold. */
    if (!g_ood_init) {
        const char *e = getenv("CIPHER_KOOPMAN_OOD_THRESHOLD");
        if (e) {
            float t = (float)atof(e);
            if (t > 0.0f && t < 1.0f) g_ood_threshold = t;
        } else {
            g_ood_threshold = 0.05f;   /* default 5% OOD tolerance */
        }
        g_ood_init = 1;
    }
    if (g_ood_threshold > 0.0f) {
        float resid = cipher_koopman_fp16_ood_max_residual(call->B, M, K_dim, N_dim);
        if (resid < 0.0f) {
            /* Shape not in registry — fall through to existing logic. */
        } else if (resid > g_ood_threshold) {
            g_calls_ood.fetch_add(1, std::memory_order_relaxed);
            return CIPHER_RT_MATMUL_PASSTHROUGH;
        }
    }

    /* Three guards collapse into this call: cipher_koopman_fp16_launch_shape
     * returns -1 if the (K_dim, N_dim) shape is not in the .cu shape registry.
     * Only calibrated shapes (via cipher_edmd_live_collect background SVD +
     * cipher_koopman_fp16_register_shape) are present in that registry, and
     * calibration is per-shape per-model. */
    int rc = cipher_koopman_fp16_launch_shape(call->B, call->C, M, K_dim, N_dim);
    if (rc != 0) {
        g_calls_skipped.fetch_add(1, std::memory_order_relaxed);
        return CIPHER_RT_MATMUL_PASSTHROUGH;
    }

    g_calls_handled.fetch_add(1, std::memory_order_relaxed);

    /* W14 Step 3 S3.B0 — emit REMEMBER event on each successful Koopman
     * substitution. tenant_id=0u single-tenant default (matches W10 Step 1
     * classify_observer pattern at cipher_rt_classify_observer.c:102 and
     * cipher_rt_kv_alloc.c:591). The S3.B1 REMEMBER consumer drains this
     * slot in batches of 256 entries per CfC LNN forward (R-W14.3
     * mitigation per scope-lock line 147). */
    if (cipher_rt_ring_write) {
        cipher_rt_koopman_event ev;
        ev.M        = (uint32_t)M;
        ev.K_dim    = (uint32_t)K_dim;
        ev.N_dim    = (uint32_t)N_dim;
        ev.op_class = 0u;
        ev.ptr_B    = (uint64_t)(uintptr_t)call->B;
        ev.ptr_C    = (uint64_t)(uintptr_t)call->C;
        ev.reserved = 0ull;
        cipher_rt_ring_write(0u,
                             CIPHER_RT_RING_EVENT_REMEMBER,
                             /* subtype = handled = 1 */ 1u,
                             /* commit_seq */ 0u,
                             &ev, sizeof(ev));
        g_remember_emits.fetch_add(1, std::memory_order_relaxed);
    }

    if (g_verbose.load(std::memory_order_relaxed)) {
        fprintf(stderr,
                "[CIPHER KOOPMAN] HANDLED M=%d K=%d N=%d "
                "(B=%p C=%p stream=%p)\n",
                M, K_dim, N_dim,
                call->B, call->C, call->stream);
    }
    *out_status = 0;  /* CUBLAS_STATUS_SUCCESS */
    return CIPHER_RT_MATMUL_HANDLED;
}

static const struct cipher_rt_matmul_actuator g_koopman_actuator = {
    "koopman",
    0,                         /* priority — substrate ignores; ordering by registration */
    maybe_handle_koopman,
};

extern "C" int cipher_rt_koopman_init(void)
{
    /* Idempotent on re-call: re-reading env is safe, registration with the
     * substrate dedupes by actuator address (or fails -1; we tolerate). */
    const char *env_on  = getenv("CIPHER_KOOPMAN");
    const char *env_vb  = getenv("CIPHER_KOOPMAN_VERBOSE");
    int enabled = (env_on && env_on[0] == '1') ? 1 : 0;
    int verbose = (env_vb && env_vb[0] == '1') ? 1 : 0;
    g_verbose.store(verbose, std::memory_order_relaxed);

    if (!enabled) {
        fprintf(stderr,
                "[CIPHER KOOPMAN] env-gated off (CIPHER_KOOPMAN!=1)\n");
        return 0;  /* no-op when env-gated off */
    }

    /* Substrate must be initialized first; cipher_inject.c calls
     * cipher_rt_matmul_dispatch_init() before us, but defensively re-call
     * (idempotent). */
    cipher_rt_matmul_dispatch_init();

    int rc = cipher_rt_matmul_register_actuator(&g_koopman_actuator);
    if (rc != 0) {
        fprintf(stderr,
                "[CIPHER KOOPMAN] register_actuator failed rc=%d "
                "(substrate full?)\n", rc);
        return -1;
    }

    g_enabled.store(1, std::memory_order_relaxed);
    fprintf(stderr,
            "[CIPHER KOOPMAN] engine registered with matmul-dispatch "
            "substrate (verbose=%d)\n", verbose);
    return 0;
}

extern "C" int cipher_rt_koopman_is_active(void)
{
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_calls_total(void)
{
    return g_calls_total.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_calls_handled(void)
{
    return g_calls_handled.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_calls_skipped(void)
{
    return g_calls_skipped.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_remember_emits(void)
{
    return g_remember_emits.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_skip_dtype(void)
{
    return g_skip_dtype.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_skip_dim(void)
{
    return g_skip_dim.load(std::memory_order_relaxed);
}

extern "C" unsigned long cipher_rt_koopman_skip_nullptr(void)
{
    return g_skip_nullptr.load(std::memory_order_relaxed);
}

/* W.3 (2026-05-27) Sub-step 1 bf16 observation accessors. */
extern "C" unsigned long cipher_rt_koopman_bf16_observed(void)
{
    return g_calls_koopman_bf16_observed.load(std::memory_order_relaxed);
}
extern "C" unsigned long cipher_rt_koopman_bf16_max_m(void)
{
    return g_calls_koopman_bf16_max_m.load(std::memory_order_relaxed);
}
extern "C" unsigned long cipher_rt_koopman_bf16_max_k(void)
{
    return g_calls_koopman_bf16_max_k.load(std::memory_order_relaxed);
}
extern "C" unsigned long cipher_rt_koopman_bf16_max_n(void)
{
    return g_calls_koopman_bf16_max_n.load(std::memory_order_relaxed);
}
