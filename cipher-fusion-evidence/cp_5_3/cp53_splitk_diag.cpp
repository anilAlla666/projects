/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cp53_splitk_diag.cpp — CP 5.3 STEP 1 split-K generalisation diagnostic.
 *
 * Experimental side-build (NOT part of libcipher_rt.so). Resolves the CP 5.3
 * load-bearing unknown: does the Marlin split-K / `locks` protocol stay
 * correct when the kernel is launched with grid < 132 (fewer CTAs than the
 * device SM count)?
 *
 * One process tests ONE (shape, grid) cell — the watchdog is the external
 * `timeout` wrapper: a deadlocked GEMM hangs cudaDeviceSynchronize forever;
 * `timeout` kills the process and the runlog shows no COMPLETED line ⇒ the
 * driver records DEADLOCK. A wrong-but-non-hanging result is caught by the
 * numerical comparison below.
 *
 * The Marlin GEMM is launched on the FULL 132-SM GPU with grid set small —
 * so all CTAs co-reside and any deadlock/error is the *protocol* at small
 * grid, not the green-context occupancy problem (CP 5.3 memo §6: STEP 1
 * isolates the kernel/grid question; green-context routing is a later STEP).
 *
 * Reference oracle: cuBLAS FP16 GEMM (FP32 accumulate) on the weight
 * dequantized from the SAME int4+scales Marlin consumes (cipher_diag_quantize_
 * export) — so the only thing measured is the GEMM, not the quantizer.
 *
 * Usage:  cp53_splitk_diag <shape_idx 0..3> <grid>
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/* ── engine API (from cp53_marlin_engine_diag.cpp) ─────────────────────── */
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
extern int cipher_diag_grid_override;
}

struct Shape { const char *name; int M, N, K; };
static const Shape SHAPES[4] = {
    { "S1-largeN-smallK", 16, 4096,  512  },
    { "S2-smallN-largeK", 16, 128,   8192 },
    { "S3-cp24-hang",     16, 4096,  4096 },
    { "S4-small-x-small", 16, 128,   512  },
};

#define CK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA-ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
    printf("RESULT status=CUDA_ERROR\n"); fflush(stdout); return 3; } } while(0)

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s <shape 0..3> <grid>\n", argv[0]); return 2; }
    int shape_idx = atoi(argv[1]);
    int grid      = atoi(argv[2]);
    if (shape_idx < 0 || shape_idx > 3) { fprintf(stderr, "bad shape\n"); return 2; }
    const Shape S = SHAPES[shape_idx];
    const int M = S.M, N = S.N, K = S.K, G = 128;

    printf("CELL shape=%s M=%d N=%d K=%d G=%d grid=%d\n", S.name, M, N, K, G, grid);
    fflush(stdout);

    /* ── deterministic inputs: fixed per-shape seed so the weight and the
     *    activation (hence the reference) are identical across all grid runs
     *    of the same shape. ─────────────────────────────────────────────── */
    std::mt19937 rng(1000u + (unsigned)shape_idx);
    std::normal_distribution<float> wdist(0.0f, 0.40f), adist(0.0f, 1.0f);

    /* weight: PyTorch (out=N, in=K) row-major — the engine transposes it. */
    std::vector<__half> h_W((size_t)N * K), h_A((size_t)M * K);
    for (auto &x : h_W) x = __float2half(wdist(rng));
    for (auto &x : h_A) x = __float2half(adist(rng));

    void *d_W = nullptr, *d_A = nullptr, *d_Cm = nullptr;
    CK(cudaMalloc(&d_W,  (size_t)N * K * sizeof(__half)));
    CK(cudaMalloc(&d_A,  (size_t)M * K * sizeof(__half)));
    CK(cudaMalloc(&d_Cm, (size_t)M * N * sizeof(__half)));
    CK(cudaMemcpy(d_W, h_W.data(), (size_t)N*K*sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_A, h_A.data(), (size_t)M*K*sizeof(__half), cudaMemcpyHostToDevice));

    if (cipher_rt_marlin_engine_init() != 0) {
        printf("RESULT status=ENGINE_INIT_FAIL\n"); return 3;
    }

    /* ── quantize + repack the weight (engine path) ────────────────────── */
    if (cipher_rt_marlin_engine_quantize_repack(d_W, K, N) != 0) {
        printf("RESULT status=QUANT_REPACK_FAIL\n"); return 3;
    }
    void *mB = nullptr, *mS = nullptr; int oK=0,oN=0,oG=0;
    if (!cipher_rt_marlin_engine_lookup(d_W, &mB, &mS, &oK, &oN, &oG)) {
        printf("RESULT status=LOOKUP_FAIL\n"); return 3;
    }

    /* ── reference: dequant the SAME int4+scales, cuBLAS FP16 GEMM ──────── */
    std::vector<uint8_t>  h_int4((size_t)K * (N/2));
    std::vector<uint16_t> h_scl ((size_t)(K/G) * N);
    if (cipher_diag_quantize_export(d_W, K, N, h_int4.data(), h_scl.data()) != 0) {
        printf("RESULT status=QUANT_EXPORT_FAIL\n"); return 3;
    }
    std::vector<__half> h_Wdeq((size_t)K * N);
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
    void *d_Wdeq = nullptr, *d_Cref = nullptr;
    CK(cudaMalloc(&d_Wdeq, (size_t)K*N*sizeof(__half)));
    CK(cudaMalloc(&d_Cref, (size_t)M*N*sizeof(__half)));
    CK(cudaMemcpy(d_Wdeq, h_Wdeq.data(), (size_t)K*N*sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t cb;
    if (cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS) { printf("RESULT status=CUBLAS_INIT_FAIL\n"); return 3; }
    float alpha = 1.0f, beta = 0.0f;
    /* row-major C[M,N] = A[M,K] @ Wdeq[K,N]  ≡  col-major (N,M)=(N,K)*(K,M) */
    cublasStatus_t cs = cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha,
                                     d_Wdeq, CUDA_R_16F, N,
                                     d_A,    CUDA_R_16F, K,
                                     &beta,
                                     d_Cref, CUDA_R_16F, N,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (cs != CUBLAS_STATUS_SUCCESS) { printf("RESULT status=CUBLAS_GEMM_FAIL cs=%d\n", cs); return 3; }
    CK(cudaDeviceSynchronize());

    /* ── Marlin GEMM at the experimental grid ──────────────────────────── */
    cipher_diag_grid_override = grid;
    CK(cudaMemset(d_Cm, 0, (size_t)M*N*sizeof(__half)));
    printf("DISPATCH grid=%d ...\n", grid); fflush(stdout);
    int rc = cipher_rt_marlin_engine_dispatch(d_A, mB, mS, d_Cm, M, N, K, G, nullptr);
    if (rc != 0) { printf("RESULT status=DISPATCH_RC_FAIL rc=%d grid=%d\n", rc, grid); return 3; }

    /* If the split-K protocol deadlocks at this grid, the next line hangs
     * forever and `timeout` kills us — no COMPLETED line is printed. */
    cudaError_t syncerr = cudaDeviceSynchronize();
    printf("COMPLETED grid=%d sync=%s\n", grid, cudaGetErrorString(syncerr));
    fflush(stdout);
    if (syncerr != cudaSuccess) { printf("RESULT status=SYNC_ERROR grid=%d err=%s\n",
                                          grid, cudaGetErrorString(syncerr)); return 3; }

    /* ── compare Marlin output vs reference ────────────────────────────── */
    std::vector<__half> h_Cm((size_t)M*N), h_Cref((size_t)M*N);
    CK(cudaMemcpy(h_Cm.data(),   d_Cm,   (size_t)M*N*sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_Cref.data(), d_Cref, (size_t)M*N*sizeof(__half), cudaMemcpyDeviceToHost));
    double max_abs = 0.0, sum_abs = 0.0, max_ref = 0.0;
    int nan_count = 0;
    for (size_t i = 0; i < (size_t)M*N; ++i) {
        float cm = __half2float(h_Cm[i]), cr = __half2float(h_Cref[i]);
        if (std::isnan(cm) || std::isnan(cr)) { nan_count++; continue; }
        double d = std::fabs((double)cm - (double)cr);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        if (std::fabs((double)cr) > max_ref) max_ref = std::fabs((double)cr);
    }
    double mean_abs = sum_abs / ((double)M*N);
    printf("RESULT status=COMPLETED grid=%d max_abs_err=%.6f mean_abs_err=%.6f "
           "max_ref=%.4f rel_max=%.6e nan=%d\n",
           grid, max_abs, mean_abs, max_ref,
           max_ref > 0 ? max_abs/max_ref : 0.0, nan_count);
    fflush(stdout);
    cublasDestroy(cb);
    return 0;
}
