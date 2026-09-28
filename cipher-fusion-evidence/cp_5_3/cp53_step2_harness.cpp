/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cp53_step2_harness.cpp — CP 5.3 STEP 2 (Axis A) partition-wiring gate harness.
 *
 * Experimental side-build (NOT part of libcipher_rt.so). Exercises the shipped
 * STEP 2 fix — grid <- green-ctx SM count + scoped PrimaryCtxGuard — through a
 * REAL 8-SM green context and runs the §7 gates A-num / B / C.
 *
 * It links the side-build engine (cp53_marlin_engine_step2.cpp, verbatim copy
 * of the shipped fixed engine + a %smid 9th kernel arg + quant export) and the
 * %smid-instrumented kernel (cp53_marlin_kernel_src_step2.cpp), and the REAL
 * shipped green-ctx module object (cipher_rt_phase4/cipher_rt_green_ctx.o) so
 * the green context, its group selection, and the STEP 2 accessors
 * cipher_rt_green_ctx_sm_count()/_group_id() are the production code paths.
 *
 * Per process the harness runs two phases:
 *   CONTROL    — green ctx NOT yet created; dispatch on the default stream.
 *                cipher_rt_green_ctx_sm_count()==0 -> engine takes grid=132,
 *                PrimaryCtxGuard held: the shipped single-tenant path. The
 *                per-shape Marlin-vs-cuBLAS error here sets epsilon = 2x it
 *                (the STEP 1 tolerance rule).
 *   PARTITION  — cipher_rt_green_ctx_ensure() creates a real 8-SM green ctx;
 *                dispatch on a green-bound stream. sm_count()==8 -> engine
 *                takes grid=8, guard skipped, GEMM launches green-confined.
 *                Per-shape error must meet epsilon (A-num); per-block %smid
 *                set must be <= 8 distinct ids (gate B).
 *
 * Roles:
 *   solo   <tenant_handle> <out_path>
 *   multi  <tenant_handle> <out_path> <barrier_dir>   (gate C: 2 processes;
 *          a file barrier aligns the two partition-phase dispatches.)
 *
 * Watchdog: a split-K co-residency deadlock (KU1) hangs cuStreamSynchronize
 * forever; the external `timeout` wrapper kills the process and the runlog
 * shows no PARTITION_DONE line => DEADLOCK.
 *
 * Usage:  cp53_step2_harness solo  <th> <out>
 *         cp53_step2_harness multi <th> <out> <barrier_dir>
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <random>
#include <vector>
#include <set>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/* ── side-build engine API ─────────────────────────────────────────────── */
extern "C" {
int  cipher_rt_marlin_engine_init(void);
int  cipher_rt_marlin_engine_quantize_repack(const void *d_fp16_weight, int K, int N);
int  cipher_rt_marlin_engine_lookup(const void *w_ptr, void **out_B, void **out_S,
                                    int *out_K, int *out_N, int *out_G);
int  cipher_rt_marlin_engine_dispatch(const void *a_fp16, const void *marlin_B,
                                      const void *marlin_S, void *c_fp16,
                                      int M, int N, int K, int G, void *stream);
int  cipher_diag_quantize_export(const void *d_fp16_weight, int K, int N,
                                 unsigned char *h_int4, unsigned short *h_scales);
extern void *cipher_diag_smid_buf;
/* ── real shipped green-ctx module (cipher_rt_green_ctx.o) ─────────────── */
int          cipher_rt_green_ctx_ensure(void);
int          cipher_rt_green_ctx_make_current(void);
unsigned int cipher_rt_green_ctx_sm_count(void);
unsigned int cipher_rt_green_ctx_group_id(void);
int          cipher_rt_green_ctx_is_initialized(void);
void        *cipher_rt_green_ctx_handle_voidp(void);
}

/* green_ctx.o calls this to pick its partition group. The shipped build
 * resolves it in cipher_rt_tenant.*; the side-build supplies it from the
 * CP53_TENANT_HANDLE env var so the gate-C driver can place the two test
 * processes on chosen (distinct) groups deterministically. */
extern "C" unsigned int cipher_rt_tenant_handle_u32_for_self(void)
{
    const char *e = getenv("CP53_TENANT_HANDLE");
    return e ? (unsigned int)strtoul(e, nullptr, 0) : 0u;
}

struct Shape { const char *name; int M, N, K; };
static const Shape SHAPES[4] = {
    { "S1-largeN-smallK", 16, 4096,  512  },
    { "S2-smallN-largeK", 16, 128,   8192 },
    { "S3-cp24-hang",     16, 4096,  4096 },
    { "S4-small-x-small", 16, 128,   512  },
};
static const int NSHAPE = 4;
static const int G = 128;

#define CK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA-ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
    printf("RESULT status=CUDA_ERROR line=%d\n",__LINE__); fflush(stdout); exit(3);} } while(0)
#define DK(call) do { CUresult r_=(call); if(r_!=CUDA_SUCCESS){ \
    const char *m=nullptr; cuGetErrorString(r_,&m); \
    printf("RESULT status=DRIVER_ERROR line=%d call=%s msg=%s\n",__LINE__,#call,m?m:"?"); \
    fflush(stdout); exit(3);} } while(0)

static const char *ts(void)
{
    static char buf[32];
    time_t t = time(nullptr); struct tm tmv;
    gmtime_r(&t, &tmv);
    strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

/* per-shape harness state */
struct ShapeRun {
    /* device buffers */
    void *d_W = nullptr, *d_A = nullptr, *d_Cm = nullptr;
    void *mB = nullptr,  *mS = nullptr;
    /* host */
    std::vector<__half> h_A, h_Cref;
    int M, N, K;
    double ctl_err = -1.0, part_err = -1.0;
    std::set<int> part_smids;
};

/* run one Marlin dispatch, sync, and return max-abs error vs h_Cref.
 * smids (if non-null) receives the distinct %smid set. The launch+sync is
 * the deadlock watchdog point. */
static double run_dispatch(ShapeRun &r, void *stream, CUcontext primary,
                           int *d_smid, int smid_cap, std::set<int> *smids,
                           const char *tag)
{
    int M = r.M, N = r.N, K = r.K;
    CK(cudaMemset(r.d_Cm, 0, (size_t)M*N*sizeof(__half)));
    if (d_smid) CK(cudaMemset(d_smid, 0xff, smid_cap*sizeof(int)));
    cipher_diag_smid_buf = d_smid;

    printf("[%s] DISPATCH M=%d N=%d K=%d stream=%p green_sm=%u ...\n",
           tag, M, N, K, stream, cipher_rt_green_ctx_sm_count());
    fflush(stdout);

    int rc = cipher_rt_marlin_engine_dispatch(r.d_A, r.mB, r.mS, r.d_Cm,
                                              M, N, K, G, stream);
    if (rc != 0) { printf("RESULT status=DISPATCH_RC_FAIL rc=%d tag=%s\n", rc, tag);
                   fflush(stdout); exit(3); }

    /* the watchdog point: a split-K deadlock hangs here forever. */
    if (stream) DK(cuStreamSynchronize((CUstream)stream));
    else        CK(cudaDeviceSynchronize());

    /* read results back under the primary ctx (KU2: primary-allocated mem). */
    CUcontext saved = nullptr;
    cuCtxGetCurrent(&saved);
    DK(cuCtxSetCurrent(primary));
    std::vector<__half> h_Cm((size_t)M*N);
    CK(cudaMemcpy(h_Cm.data(), r.d_Cm, (size_t)M*N*sizeof(__half),
                  cudaMemcpyDeviceToHost));
    std::vector<int> h_smid;
    if (d_smid) {
        h_smid.resize(smid_cap);
        CK(cudaMemcpy(h_smid.data(), d_smid, smid_cap*sizeof(int),
                      cudaMemcpyDeviceToHost));
    }
    if (saved) cuCtxSetCurrent(saved);

    double max_abs = 0.0; int nan_count = 0;
    for (size_t i = 0; i < (size_t)M*N; ++i) {
        float cm = __half2float(h_Cm[i]), cr = __half2float(r.h_Cref[i]);
        if (std::isnan(cm) || std::isnan(cr)) { nan_count++; continue; }
        double d = std::fabs((double)cm - (double)cr);
        if (d > max_abs) max_abs = d;
    }
    if (smids && d_smid) {
        for (int i = 0; i < smid_cap; ++i)
            if (h_smid[i] >= 0) smids->insert(h_smid[i]);
    }
    printf("[%s] max_abs_err=%.6f nan=%d\n", tag, max_abs, nan_count);
    fflush(stdout);
    return max_abs;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <solo|multi> <tenant_handle> <out> [barrier_dir]\n",
                argv[0]);
        return 2;
    }
    const std::string role = argv[1];
    const std::string out_path = argv[3];
    const std::string barrier_dir = (argc > 4) ? argv[4] : "";
    const bool multi = (role == "multi");
    if (multi && barrier_dir.empty()) {
        fprintf(stderr, "multi role needs <barrier_dir>\n"); return 2;
    }
    setenv("CP53_TENANT_HANDLE", argv[2], 1);

    printf("CP 5.3 STEP 2 harness — role=%s tenant_handle=%s pid=%d %s\n",
           role.c_str(), argv[2], getpid(), ts());
    printf("======================================================\n");
    fflush(stdout);

    /* ── driver init + primary context ─────────────────────────────────── */
    DK(cuInit(0));
    CUdevice dev;  DK(cuDeviceGet(&dev, 0));
    CUcontext primary;
    DK(cuDevicePrimaryCtxRetain(&primary, dev));
    DK(cuCtxSetCurrent(primary));
    CK(cudaSetDevice(0));

    if (cipher_rt_marlin_engine_init() != 0) {
        printf("RESULT status=ENGINE_INIT_FAIL\n"); return 3;
    }

    /* ── per-shape setup: weights, quant, cuBLAS reference ─────────────── */
    ShapeRun runs[NSHAPE];
    cublasHandle_t cb;
    if (cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS) {
        printf("RESULT status=CUBLAS_INIT_FAIL\n"); return 3;
    }
    for (int si = 0; si < NSHAPE; ++si) {
        const Shape S = SHAPES[si];
        ShapeRun &r = runs[si];
        r.M = S.M; r.N = S.N; r.K = S.K;
        const int M = S.M, N = S.N, K = S.K;

        std::mt19937 rng(1000u + (unsigned)si);            /* STEP 1 seeds */
        std::normal_distribution<float> wdist(0.0f, 0.40f), adist(0.0f, 1.0f);
        std::vector<__half> h_W((size_t)N*K);
        r.h_A.resize((size_t)M*K);
        for (auto &x : h_W)   x = __float2half(wdist(rng));
        for (auto &x : r.h_A) x = __float2half(adist(rng));

        CK(cudaMalloc(&r.d_W,  (size_t)N*K*sizeof(__half)));
        CK(cudaMalloc(&r.d_A,  (size_t)M*K*sizeof(__half)));
        CK(cudaMalloc(&r.d_Cm, (size_t)M*N*sizeof(__half)));
        CK(cudaMemcpy(r.d_W, h_W.data(),  (size_t)N*K*sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(r.d_A, r.h_A.data(),(size_t)M*K*sizeof(__half), cudaMemcpyHostToDevice));

        if (cipher_rt_marlin_engine_quantize_repack(r.d_W, K, N) != 0) {
            printf("RESULT status=QUANT_REPACK_FAIL shape=%s\n", S.name); return 3;
        }
        int oK=0,oN=0,oG=0;
        if (!cipher_rt_marlin_engine_lookup(r.d_W, &r.mB, &r.mS, &oK, &oN, &oG)) {
            printf("RESULT status=LOOKUP_FAIL shape=%s\n", S.name); return 3;
        }

        /* cuBLAS reference from the engine's exact int4+scales */
        std::vector<uint8_t>  h_int4((size_t)K*(N/2));
        std::vector<uint16_t> h_scl ((size_t)(K/G)*N);
        if (cipher_diag_quantize_export(r.d_W, K, N, h_int4.data(), h_scl.data()) != 0) {
            printf("RESULT status=QUANT_EXPORT_FAIL shape=%s\n", S.name); return 3;
        }
        std::vector<__half> h_Wdeq((size_t)K*N);
        for (int k = 0; k < K; ++k) {
            int kg = k / G;
            for (int c = 0; c < N/2; ++c) {
                uint8_t b = h_int4[(size_t)k*(N/2)+c];
                int lo = b & 0xF, hi = (b>>4) & 0xF;
                int q0 = (lo & 0x8) ? lo-16 : lo;
                int q1 = (hi & 0x8) ? hi-16 : hi;
                __half s0h, s1h;
                uint16_t u0 = h_scl[(size_t)kg*N + 2*c], u1 = h_scl[(size_t)kg*N + 2*c+1];
                memcpy(&s0h,&u0,2); memcpy(&s1h,&u1,2);
                h_Wdeq[(size_t)k*N + 2*c]   = __float2half(q0 * __half2float(s0h));
                h_Wdeq[(size_t)k*N + 2*c+1] = __float2half(q1 * __half2float(s1h));
            }
        }
        void *d_Wdeq=nullptr, *d_Cref=nullptr;
        CK(cudaMalloc(&d_Wdeq, (size_t)K*N*sizeof(__half)));
        CK(cudaMalloc(&d_Cref, (size_t)M*N*sizeof(__half)));
        CK(cudaMemcpy(d_Wdeq, h_Wdeq.data(), (size_t)K*N*sizeof(__half), cudaMemcpyHostToDevice));
        float alpha=1.0f, beta=0.0f;
        cublasStatus_t cs = cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K, &alpha,
                                         d_Wdeq, CUDA_R_16F, N,
                                         r.d_A,  CUDA_R_16F, K,
                                         &beta,
                                         d_Cref, CUDA_R_16F, N,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (cs != CUBLAS_STATUS_SUCCESS) {
            printf("RESULT status=CUBLAS_GEMM_FAIL cs=%d shape=%s\n", cs, S.name); return 3;
        }
        CK(cudaDeviceSynchronize());
        r.h_Cref.resize((size_t)M*N);
        CK(cudaMemcpy(r.h_Cref.data(), d_Cref, (size_t)M*N*sizeof(__half), cudaMemcpyDeviceToHost));
        CK(cudaFree(d_Wdeq)); CK(cudaFree(d_Cref));
        printf("[setup] %-18s quantized + cuBLAS reference ready\n", S.name);
        fflush(stdout);
    }
    cublasDestroy(cb);

    /* smid capture buffer — sized for the full-device control grid. */
    const int SMID_CAP = 160;
    int *d_smid = nullptr;
    CK(cudaMalloc(&d_smid, SMID_CAP*sizeof(int)));

    /* ── CONTROL phase — green ctx NOT created -> grid=132, guarded ─────── */
    if (cipher_rt_green_ctx_sm_count() != 0) {
        printf("RESULT status=PRECONDITION_FAIL detail=green_active_before_control\n");
        return 3;
    }
    printf("\n-- CONTROL phase (single-tenant path, grid=device SM count) --\n");
    for (int si = 0; si < NSHAPE; ++si) {
        printf("[ctl] shape=%s\n", SHAPES[si].name);
        runs[si].ctl_err = run_dispatch(runs[si], nullptr, primary,
                                        d_smid, SMID_CAP, nullptr, "ctl");
    }
    printf("CONTROL_DONE %s\n", ts()); fflush(stdout);

    /* ── create the real 8-SM green context ────────────────────────────── */
    printf("\n-- creating real green context --\n");
    if (cipher_rt_green_ctx_ensure() != 0) {
        printf("RESULT status=GREEN_ENSURE_FAIL\n"); return 3;
    }
    unsigned int green_sm = cipher_rt_green_ctx_sm_count();
    unsigned int group_id = cipher_rt_green_ctx_group_id();
    printf("[green] ensured: sm_count=%u group_id=%u\n", green_sm, group_id);
    if (green_sm != 8) {
        printf("RESULT status=GREEN_SM_UNEXPECTED sm_count=%u (want 8)\n", green_sm);
        return 3;
    }

    /* a green-bound stream: created with the green ctx current. */
    DK((CUresult)(cipher_rt_green_ctx_make_current() == 0 ? CUDA_SUCCESS
                                                          : CUDA_ERROR_INVALID_CONTEXT));
    CUstream gstream;
    DK(cuStreamCreate(&gstream, CU_STREAM_NON_BLOCKING));
    CUcontext sctx = nullptr;
    DK(cuStreamGetCtx(gstream, &sctx));
    /* sanity: the stream must belong to the green ctx, not primary. */
    if (sctx == primary) {
        printf("RESULT status=STREAM_NOT_GREEN_BOUND\n"); return 3;
    }
    printf("[green] green-bound stream created (stream ctx != primary)\n");
    fflush(stdout);

    /* ── gate-C file barrier: align the two partition-phase dispatches ─── */
    if (multi) {
        mkdir(barrier_dir.c_str(), 0777);
        char rf[512];
        snprintf(rf, sizeof rf, "%s/ready.%d", barrier_dir.c_str(), getpid());
        FILE *f = fopen(rf, "w"); if (f) { fprintf(f, "%u\n", group_id); fclose(f); }
        printf("[barrier] %s posted; waiting for peer ...\n", rf); fflush(stdout);
        for (int spin = 0; spin < 600; ++spin) {  /* ~60 s cap */
            int n = 0;
            for (int p = 0; p < 400000; ++p) {
                char cand[512];
                snprintf(cand, sizeof cand, "%s/ready.%d", barrier_dir.c_str(), p);
                struct stat st;
                if (stat(cand, &st) == 0) n++;
            }
            if (n >= 2) break;
            struct timespec slp{0, 100*1000*1000}; nanosleep(&slp, nullptr);
        }
        printf("[barrier] released %s\n", ts()); fflush(stdout);
    }

    /* ── PARTITION phase — green stream -> grid=8, guard skipped ────────── */
    printf("\n-- PARTITION phase (8-SM green ctx, grid=partition SM count) --\n");
    for (int si = 0; si < NSHAPE; ++si) {
        printf("[part] shape=%s\n", SHAPES[si].name);
        runs[si].part_smids.clear();
        runs[si].part_err = run_dispatch(runs[si], gstream, primary,
                                         d_smid, SMID_CAP, &runs[si].part_smids,
                                         "part");
    }
    printf("PARTITION_DONE %s\n", ts()); fflush(stdout);

    /* ── verdict + result file ─────────────────────────────────────────── */
    FILE *out = fopen(out_path.c_str(), "w");
    if (!out) { printf("RESULT status=OUT_OPEN_FAIL path=%s\n", out_path.c_str()); return 3; }
    fprintf(out, "# CP 5.3 STEP 2 harness result — pid=%d role=%s %s\n",
            getpid(), role.c_str(), ts());
    fprintf(out, "GROUP_ID %u\n", group_id);
    fprintf(out, "GREEN_SM %u\n", green_sm);

    bool all_anum = true, all_B = true;
    std::set<int> union_smids;
    for (int si = 0; si < NSHAPE; ++si) {
        ShapeRun &r = runs[si];
        double eps = 2.0 * r.ctl_err;
        bool anum = (r.part_err <= eps) || (r.part_err <= r.ctl_err);
        bool bgate = (r.part_smids.size() <= 8);
        all_anum &= anum; all_B &= bgate;
        std::string smids;
        for (int s : r.part_smids) { smids += std::to_string(s); smids += ","; union_smids.insert(s); }
        if (!smids.empty()) smids.pop_back();
        fprintf(out, "SHAPE %s CTL_ERR %.6f PART_ERR %.6f EPS %.6f ANUM %s "
                     "NSMID %zu B %s SMIDS %s\n",
                SHAPES[si].name, r.ctl_err, r.part_err, eps,
                anum?"PASS":"FAIL", r.part_smids.size(), bgate?"PASS":"FAIL",
                smids.c_str());
        printf("RESULT shape=%s ctl_err=%.6f part_err=%.6f eps=%.6f anum=%s "
               "nsmid=%zu B=%s\n",
               SHAPES[si].name, r.ctl_err, r.part_err, eps, anum?"PASS":"FAIL",
               r.part_smids.size(), bgate?"PASS":"FAIL");
    }
    std::string us;
    for (int s : union_smids) { us += std::to_string(s); us += ","; }
    if (!us.empty()) us.pop_back();
    fprintf(out, "UNION_SMIDS %s\n", us.c_str());
    fprintf(out, "ANUM_ALL %s\n", all_anum?"PASS":"FAIL");
    fprintf(out, "B_ALL %s\n",    all_B?"PASS":"FAIL");
    fclose(out);

    printf("\nRESULT status=HARNESS_DONE anum_all=%s b_all=%s group_id=%u "
           "union_smids=%s out=%s\n",
           all_anum?"PASS":"FAIL", all_B?"PASS":"FAIL", group_id, us.c_str(),
           out_path.c_str());
    fflush(stdout);

    cuStreamDestroy(gstream);
    return (all_anum && all_B) ? 0 : 1;
}
