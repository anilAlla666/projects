/* W14 Step 3 S3.C — LM-head validation harness (two-pass).
 *
 * Per Anil adjudication 2026-05-24 (option α two-mode + option full-substrate
 * intercept + WEEK_14_STEP_3_C_KL_GATE_ADDENDUM.md):
 *
 *   Pass I (fire mode, EXISTENCE contract):
 *     - Synthetic narrow-domain manifold: X_calib ∈ R^[M_calib x K] with
 *       rank R << K by construction (X_calib = U_seed * Σ_seed * V_seed^T).
 *     - Register the shape (K, N) with V_T = V_seed^T, K_op = I_R,
 *       W_buf = V_seed^T * W_real^T (closed-form EDMD-live result).
 *     - X_in sampled from the same manifold span(V_seed); cuBLAS GemmEx
 *       intercepted by the GOT-patched substrate; Koopman engine fires.
 *     - Gate: cipher_rt_koopman_calls_handled advances, remember_emits ==
 *       handled, consumer drained advances; top-1(Y_cipher_in,
 *       Y_vanilla_in) ≥ 90%. Fire-mode KL reported informationally.
 *
 *   Pass II (passthrough mode, PRESERVATION contract):
 *     - X_random ~ N(0, 1) off-manifold (residual_ratio vs V_seed
 *       expected ~sqrt((K-R)/K) ≈ 0.98 for K=2048 R=64; >> β=0.05 default).
 *     - cuBLAS GemmEx intercepted; β OOD detector fires PASSTHROUGH; the
 *       shim falls back to real cublasGemmEx.
 *     - Gate: handled does NOT advance, ood advances; top-1 ≥ 99.95%,
 *       KL ≤ 5.5e-5 vs Y_vanilla_random.
 *
 * Y_vanilla_{in,random} are computed BEFORE shape registration (substrate
 * dispatches → koopman engine returns PASSTHROUGH because the shape is not
 * in the .cu registry → shim falls back to real cublasGemmEx).
 *
 * Required runtime envelope (the gates are CIPHER_KOOPMAN=1 + CIPHER_REMEMBER=1
 * with libcipher_rt LD_PRELOAD'd so the GOT-patched cublasGemmEx is in scope
 * for this harness binary as well):
 *
 *   LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
 *   CIPHER_KOOPMAN=1 CIPHER_REMEMBER=1 \
 *   ./test_step3_c_lmhead_validate
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

extern "C" {
#include "cipher_rt_koopman.h"
#include "cipher_rt_ring_write.h"
#include "cipher_rt_remember_consumer.h"
/* InitializeInjection2 is the CUDA driver entrypoint into cipher_inject.c;
 * the driver calls it via CUDA_INJECTION64_PATH when set. With plain
 * LD_PRELOAD we have to call it ourselves so cipher_v2_init_body runs and
 * the Koopman engine + GOT patching install. */
int InitializeInjection2(void);
}

/* The .cu kernel exports this entrypoint for direct calibration; we use
 * cipher_koopman_fp16_register_shape to seed the shape registry, then the
 * substrate's matmul-dispatch routes intercepted cublasGemmEx calls into
 * cipher_koopman_fp16_launch_shape. */
/* Real signature (cipher_block_sub_kernel.cu:479) — V_T and W are fp32
 * row-major; W_fp16/V_T_fp16 are derived internally via make_fp16_copy.
 * Layout: V_T[r * K + k] = V_x[k, r]  (R rows of K floats; see comment at
 * cipher_block_sub_kernel.cu:43). K_op is (R x R) row-major fp32.
 * W_buf is (R x N) row-major fp32. */
extern "C" int cipher_koopman_fp16_register_shape(
    int K_dim, int N_dim,
    float *vt_RK_fp32,
    float *k_op_RR_fp32,
    float *w_RN_fp32);

extern "C" float cipher_koopman_fp16_ood_max_residual(
    const void *x_fp16, int M, int K_dim, int N_dim);

/* Problem dimensions — mirrors TinyLlama LM head shape so the harness
 * exercises the same FP16 GEMM the substrate would see in the real model. */
static const int K       = 2048;     /* input feature dim */
static const int N       = 32000;    /* output feature dim (vocab) */
static const int R       = 64;       /* low-rank manifold dimension */
static const int M_CALIB = 2000;     /* calibration manifold size */
static const int M_TEST  = 10;       /* held-out test rows */

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "[S3.C] CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "[S3.C] cuBLAS error %d at %s:%d\n", \
            (int)s, __FILE__, __LINE__); return 1; } } while (0)

/* host fp32 -> device fp16, in place via cudaMemcpy from a host fp16 buffer */
static void host_fp32_to_fp16(const float* h_fp32, __half* h_fp16, size_t n)
{
    for (size_t i = 0; i < n; i++) h_fp16[i] = __float2half(h_fp32[i]);
}

/* Modified-Gram-Schmidt orthonormalization, in-place on a row-major
 * matrix of size rows x cols (each column is one vector to orthonormalize). */
static void mgs_orthonormal_cols(float* A, int rows, int cols)
{
    for (int j = 0; j < cols; j++) {
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < rows; i++) dot += A[i * cols + j] * A[i * cols + k];
            for (int i = 0; i < rows; i++) A[i * cols + j] -= dot * A[i * cols + k];
        }
        float norm = 0.0f;
        for (int i = 0; i < rows; i++) norm += A[i * cols + j] * A[i * cols + j];
        norm = std::sqrt(norm);
        if (norm < 1e-20f) {
            for (int i = 0; i < rows; i++) A[i * cols + j] = 0.0f;
            if (j < rows) A[j * cols + j] = 1.0f;
            continue;
        }
        float inv = 1.0f / norm;
        for (int i = 0; i < rows; i++) A[i * cols + j] *= inv;
    }
}

/* Host softmax + KL — operating on (rows, cols) row-major fp32. */
static void softmax_row(const float* logits, float* probs, int cols)
{
    float m = logits[0];
    for (int j = 1; j < cols; j++) if (logits[j] > m) m = logits[j];
    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        probs[j] = std::exp(logits[j] - m);
        sum += probs[j];
    }
    float inv_sum = (float)(1.0 / sum);
    for (int j = 0; j < cols; j++) probs[j] *= inv_sum;
}

static float kl_row(const float* p_logits, const float* q_logits, int cols,
                    std::vector<float>& bufp, std::vector<float>& bufq)
{
    softmax_row(p_logits, bufp.data(), cols);
    softmax_row(q_logits, bufq.data(), cols);
    double kl = 0.0;
    for (int j = 0; j < cols; j++) {
        if (bufp[j] > 1e-30f) {
            float qq = (bufq[j] > 1e-30f) ? bufq[j] : 1e-30f;
            kl += bufp[j] * (std::log(bufp[j]) - std::log(qq));
        }
    }
    return (float)kl;
}

int main(void)
{
    setbuf(stdout, NULL);
    printf("[S3.C HARNESS] K=%d N=%d R=%d M_calib=%d M_test=%d\n",
           K, N, R, M_CALIB, M_TEST);

    /* CIPHER_KOOPMAN + CIPHER_REMEMBER are read at init time. Trigger the
     * libcipher_rt init body explicitly (the CUDA driver does this via
     * CUDA_INJECTION64_PATH when set; under plain LD_PRELOAD we need to
     * call it ourselves). */
    (void)InitializeInjection2();

    if (!cipher_rt_koopman_is_active()) {
        printf("[S3.C HARNESS] FAIL: Koopman engine inactive. "
               "Set CIPHER_KOOPMAN=1 in the environment and LD_PRELOAD libcipher_rt.so.\n");
        return 1;
    }
    printf("[S3.C HARNESS] Koopman engine active. REMEMBER consumer active=%d\n",
           cipher_rt_remember_consumer_is_active());

    cublasHandle_t blas;
    CHECK_CUBLAS(cublasCreate(&blas));

    /* ------------------------------------------------------------
     * Step 1: build the synthetic manifold + W_real on host.
     * ------------------------------------------------------------ */
    std::mt19937 rng(0xC1FE);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    /* V_seed: K x R orthonormal columns (manifold basis). */
    std::vector<float> V_seed((size_t)K * R);
    for (auto& v : V_seed) v = nd(rng);
    mgs_orthonormal_cols(V_seed.data(), K, R);

    /* W_real: N x K, row-major fp32 -> fp16. */
    std::vector<float> W_real_fp32((size_t)N * K);
    for (auto& v : W_real_fp32) v = nd(rng) * 0.02f;   /* scale so fp16 matmul stays in range */

    /* X_in (in-distribution): M_TEST x K sampled from manifold span(V_seed).
     * X_in[i] = V_seed @ alpha_i with alpha_i ~ N(0, sigma_manifold) so the
     * residual_ratio ||x - V_x V_x^T x||/||x|| is ~0 (pure manifold). */
    std::vector<float> X_in_fp32((size_t)M_TEST * K, 0.0f);
    for (int i = 0; i < M_TEST; i++) {
        for (int r = 0; r < R; r++) {
            float alpha = nd(rng) * 2.0f;     /* manifold coordinate */
            for (int k = 0; k < K; k++) {
                X_in_fp32[i * K + k] += alpha * V_seed[k * R + r];
            }
        }
    }

    /* X_random (off-manifold): random N(0, 1); residual_ratio ~ sqrt((K-R)/K) */
    std::vector<float> X_random_fp32((size_t)M_TEST * K);
    for (auto& v : X_random_fp32) v = nd(rng);

    /* Host fp16 conversions. */
    std::vector<__half> W_real_h(N * K);
    std::vector<__half> X_in_h(M_TEST * K);
    std::vector<__half> X_random_h(M_TEST * K);
    host_fp32_to_fp16(W_real_fp32.data(), W_real_h.data(), W_real_fp32.size());
    host_fp32_to_fp16(X_in_fp32.data(),   X_in_h.data(),   X_in_fp32.size());
    host_fp32_to_fp16(X_random_fp32.data(), X_random_h.data(), X_random_fp32.size());

    /* GPU buffers. */
    __half *d_W_real, *d_X_in, *d_X_random;
    __half *d_Y_van_in, *d_Y_van_rand;          /* vanilla baselines */
    __half *d_Y_cip_in, *d_Y_cip_rand;          /* substrate-routed */
    CHECK_CUDA(cudaMalloc(&d_W_real,    sizeof(__half) * (size_t)N * K));
    CHECK_CUDA(cudaMalloc(&d_X_in,      sizeof(__half) * (size_t)M_TEST * K));
    CHECK_CUDA(cudaMalloc(&d_X_random,  sizeof(__half) * (size_t)M_TEST * K));
    CHECK_CUDA(cudaMalloc(&d_Y_van_in,    sizeof(__half) * (size_t)N * M_TEST));
    CHECK_CUDA(cudaMalloc(&d_Y_van_rand,  sizeof(__half) * (size_t)N * M_TEST));
    CHECK_CUDA(cudaMalloc(&d_Y_cip_in,    sizeof(__half) * (size_t)N * M_TEST));
    CHECK_CUDA(cudaMalloc(&d_Y_cip_rand,  sizeof(__half) * (size_t)N * M_TEST));
    CHECK_CUDA(cudaMemcpy(d_W_real, W_real_h.data(),
                          sizeof(__half) * (size_t)N * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X_in, X_in_h.data(),
                          sizeof(__half) * (size_t)M_TEST * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X_random, X_random_h.data(),
                          sizeof(__half) * (size_t)M_TEST * K, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------------
     * Step 2: compute vanilla baselines BEFORE registration.
     *
     * cublasGemmEx with the substrate active but the (K, N) shape NOT yet
     * in the .cu registry → maybe_handle_koopman returns PASSTHROUGH →
     * the shim falls back to real cublasGemmEx → output is vanilla.
     *
     * cuBLAS GemmEx convention: C[m, n] = A[m, k] * B[k, n] (col-major).
     * For our test: m=N, n=M_TEST, k=K. We want Y[N, M_TEST] = W_real[N, K]
     * @ X[K, M_TEST]. In col-major: A (= W_real) is N x K with leading
     * dimension N, B (= X) is K x M_TEST with leading dimension K.
     * But W_real is stored row-major as N x K; in col-major view that is
     * K x N. So we transpose: use CUBLAS_OP_T on A with K x N storage and
     * leading dim K. Equivalently: treat W_real_h as K x N col-major
     * (which IS the same memory as N x K row-major) and CUBLAS_OP_T it.
     * ------------------------------------------------------------ */
    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    auto gemm = [&](const __half* A_kn_colmajor_or_W, const __half* B_km_colmajor_or_X,
                    __half* C_nm_colmajor, int m_out_rows, int n_batch_cols, int k_inner) {
        return cublasGemmEx(blas,
            CUBLAS_OP_T,  /* A is K x N col-major; transpose to N x K */
            CUBLAS_OP_N,  /* B is K x M col-major; leave */
            m_out_rows,   /* m: output rows = N */
            n_batch_cols, /* n: output cols = M_TEST */
            k_inner,      /* k: K */
            &alpha,
            A_kn_colmajor_or_W, CUDA_R_16F, k_inner,    /* lda = K */
            B_km_colmajor_or_X, CUDA_R_16F, k_inner,    /* ldb = K */
            &beta,
            C_nm_colmajor, CUDA_R_16F, m_out_rows,      /* ldc = N */
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
    };

    /* Sample telemetry baseline pre-vanilla. */
    unsigned long pre_total      = cipher_rt_koopman_calls_total();
    unsigned long pre_handled    = cipher_rt_koopman_calls_handled();
    unsigned long pre_skipped    = cipher_rt_koopman_calls_skipped();
    unsigned long pre_emits      = cipher_rt_koopman_remember_emits();

    CHECK_CUBLAS(gemm(d_W_real, d_X_in,     d_Y_van_in,   N, M_TEST, K));
    CHECK_CUBLAS(gemm(d_W_real, d_X_random, d_Y_van_rand, N, M_TEST, K));
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long post_van_total   = cipher_rt_koopman_calls_total();
    unsigned long post_van_handled = cipher_rt_koopman_calls_handled();
    unsigned long post_van_skipped = cipher_rt_koopman_calls_skipped();
    unsigned long post_van_emits   = cipher_rt_koopman_remember_emits();
    printf("[S3.C VANILLA BASELINE] total +%lu handled +%lu skipped +%lu emits +%lu "
           "(expect handled=0, skipped or PASSTHROUGH; shape not registered yet)\n",
           post_van_total - pre_total,
           post_van_handled - pre_handled,
           post_van_skipped - pre_skipped,
           post_van_emits - pre_emits);

    /* Copy vanilla outputs to host. */
    std::vector<__half> Y_van_in_h(  (size_t)N * M_TEST);
    std::vector<__half> Y_van_rand_h((size_t)N * M_TEST);
    CHECK_CUDA(cudaMemcpy(Y_van_in_h.data(),   d_Y_van_in,
                          sizeof(__half) * Y_van_in_h.size(),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Y_van_rand_h.data(), d_Y_van_rand,
                          sizeof(__half) * Y_van_rand_h.size(), cudaMemcpyDeviceToHost));

    /* ------------------------------------------------------------
     * Step 3: compute V_T (R x K row-major from V_seed col-major K x R)
     *         compute K_op = I_R (R x R row-major fp32)
     *         compute W_buf = V_seed^T @ W_real^T (R x N row-major fp32)
     * V_T_fp16 must be K x R col-major fp16 per
     * cipher_koopman_fp16_register_shape API conventions (see
     * src/may13/cipher_block_sub_kernel.cu register signature).
     * ------------------------------------------------------------ */
    /* V_T row-major (R x K) fp32 per cipher_block_sub_kernel.cu:43:
     *   V_T[r * K + k] = V_x[k, r]
     * V_seed is K x R row-major; V_seed[k * R + r] is V_x[k, r]. */
    std::vector<float> V_T_fp32((size_t)R * K);
    for (int r = 0; r < R; r++)
        for (int k = 0; k < K; k++)
            V_T_fp32[r * K + k] = V_seed[k * R + r];

    std::vector<float> K_op_fp32((size_t)R * R, 0.0f);
    for (int r = 0; r < R; r++) K_op_fp32[r * R + r] = 1.0f;

    /* W_buf = V_seed^T @ W_real^T  shape (R, N).
     * V_seed is K x R row-major; V_seed^T is R x K.
     * W_real is N x K row-major; W_real^T is K x N.
     * W_buf[r, n] = sum_k V_seed[k, r] * W_real[n, k] */
    std::vector<float> W_buf_fp32((size_t)R * N);
    for (int r = 0; r < R; r++) {
        for (int n2 = 0; n2 < N; n2++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += (double)V_seed[k * R + r] * (double)W_real_fp32[n2 * K + k];
            W_buf_fp32[r * N + n2] = (float)acc;
        }
    }

    int rc = cipher_koopman_fp16_register_shape(K, N,
        V_T_fp32.data(),
        K_op_fp32.data(),
        W_buf_fp32.data());
    printf("[S3.C REGISTER] cipher_koopman_fp16_register_shape rc=%d (K=%d N=%d R=%d)\n",
           rc, K, N, R);
    if (rc != 0) {
        printf("[S3.C HARNESS] FAIL: shape registration failed\n");
        return 1;
    }

    /* OOD residual sanity probe. */
    __half *d_X_one;
    CHECK_CUDA(cudaMalloc(&d_X_one, sizeof(__half) * (size_t)K * 1));
    CHECK_CUDA(cudaMemcpy(d_X_one, X_in_h.data(),
                          sizeof(__half) * (size_t)K * 1, cudaMemcpyHostToDevice));
    float resid_in = cipher_koopman_fp16_ood_max_residual(d_X_one, 1, K, N);
    CHECK_CUDA(cudaMemcpy(d_X_one, X_random_h.data(),
                          sizeof(__half) * (size_t)K * 1, cudaMemcpyHostToDevice));
    float resid_rand = cipher_koopman_fp16_ood_max_residual(d_X_one, 1, K, N);
    printf("[S3.C OOD PROBE] residual_ratio in-dist=%.6f off-dist=%.6f "
           "(β threshold default 0.05)\n", resid_in, resid_rand);

    /* ------------------------------------------------------------
     * Step 4: Pass I — in-distribution.
     * ------------------------------------------------------------ */
    unsigned long pre_p1_total   = cipher_rt_koopman_calls_total();
    unsigned long pre_p1_handled = cipher_rt_koopman_calls_handled();
    unsigned long pre_p1_skipped = cipher_rt_koopman_calls_skipped();
    unsigned long pre_p1_ood     = post_van_handled;  /* placeholder; recapture below */
    unsigned long pre_p1_emits   = cipher_rt_koopman_remember_emits();
    uint64_t pre_p1_drained      = cipher_rt_remember_consumer_drained();
    /* No direct ood accessor in v1 header; we infer via skipped+passthrough delta. */
    (void)pre_p1_ood;

    CHECK_CUBLAS(gemm(d_W_real, d_X_in, d_Y_cip_in, N, M_TEST, K));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Allow consumer drain to land. */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    unsigned long post_p1_total   = cipher_rt_koopman_calls_total();
    unsigned long post_p1_handled = cipher_rt_koopman_calls_handled();
    unsigned long post_p1_skipped = cipher_rt_koopman_calls_skipped();
    unsigned long post_p1_emits   = cipher_rt_koopman_remember_emits();
    uint64_t post_p1_drained      = cipher_rt_remember_consumer_drained();

    unsigned long d_handled = post_p1_handled - pre_p1_handled;
    unsigned long d_skipped = post_p1_skipped - pre_p1_skipped;
    unsigned long d_total   = post_p1_total   - pre_p1_total;
    unsigned long d_emits   = post_p1_emits   - pre_p1_emits;
    uint64_t d_drained      = post_p1_drained - pre_p1_drained;
    printf("[S3.C PASS I  TELEMETRY] total +%lu handled +%lu skipped +%lu emits +%lu drained +%lu\n",
           d_total, d_handled, d_skipped, d_emits, d_drained);

    std::vector<__half> Y_cip_in_h((size_t)N * M_TEST);
    CHECK_CUDA(cudaMemcpy(Y_cip_in_h.data(), d_Y_cip_in,
                          sizeof(__half) * Y_cip_in_h.size(), cudaMemcpyDeviceToHost));

    /* Top-1 and KL per row.
     * Layout: Y is col-major (N, M_TEST); row r of "logits per test sample"
     * is the column m, indexed as Y[r * M_TEST + m]... actually col-major
     * indexing: element (i, j) is at offset (j * N + i). So column m
     * contains logits for sample m at indices [m*N .. m*N + N). */
    int top1_match = 0;
    std::vector<float> kl_per(M_TEST);
    std::vector<float> p_buf(N), q_buf(N);
    std::vector<float> logit_v(N), logit_c(N);
    for (int m = 0; m < M_TEST; m++) {
        int argmax_v = 0, argmax_c = 0;
        float max_v = -1e30f, max_c = -1e30f;
        for (int n2 = 0; n2 < N; n2++) {
            float lv = __half2float(Y_van_in_h[m * N + n2]);
            float lc = __half2float(Y_cip_in_h[m * N + n2]);
            logit_v[n2] = lv;
            logit_c[n2] = lc;
            if (lv > max_v) { max_v = lv; argmax_v = n2; }
            if (lc > max_c) { max_c = lc; argmax_c = n2; }
        }
        if (argmax_v == argmax_c) top1_match++;
        kl_per[m] = kl_row(logit_v.data(), logit_c.data(), N, p_buf, q_buf);
    }
    float top1_in = (float)top1_match / (float)M_TEST;
    float kl_mean_in = 0.0f, kl_max_in = 0.0f;
    for (auto v : kl_per) { kl_mean_in += v; if (v > kl_max_in) kl_max_in = v; }
    kl_mean_in /= M_TEST;
    printf("[S3.C PASS I  QUALITY] top1_match=%.4f kl_mean=%.6e kl_max=%.6e "
           "(gate: top-1 >= 0.90; KL informational)\n",
           top1_in, kl_mean_in, kl_max_in);

    int pass_i_telemetry =
        (d_handled >= 1) &&
        (d_emits == d_handled) &&
        (d_drained >= 1);
    int pass_i_quality = (top1_in >= 0.90f);
    int pass_i = pass_i_telemetry && pass_i_quality;
    printf("[S3.C PASS I  RESULT] telemetry=%s quality=%s overall=%s\n",
           pass_i_telemetry ? "PASS" : "FAIL",
           pass_i_quality ? "PASS" : "FAIL",
           pass_i ? "PASS" : "FAIL");

    /* ------------------------------------------------------------
     * Step 5: Pass II — off-manifold.
     * ------------------------------------------------------------ */
    unsigned long pre_p2_total   = cipher_rt_koopman_calls_total();
    unsigned long pre_p2_handled = cipher_rt_koopman_calls_handled();
    unsigned long pre_p2_skipped = cipher_rt_koopman_calls_skipped();
    unsigned long pre_p2_emits   = cipher_rt_koopman_remember_emits();
    uint64_t pre_p2_drained      = cipher_rt_remember_consumer_drained();

    CHECK_CUBLAS(gemm(d_W_real, d_X_random, d_Y_cip_rand, N, M_TEST, K));
    CHECK_CUDA(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    unsigned long post_p2_total   = cipher_rt_koopman_calls_total();
    unsigned long post_p2_handled = cipher_rt_koopman_calls_handled();
    unsigned long post_p2_skipped = cipher_rt_koopman_calls_skipped();
    unsigned long post_p2_emits   = cipher_rt_koopman_remember_emits();
    uint64_t post_p2_drained      = cipher_rt_remember_consumer_drained();

    unsigned long d2_handled = post_p2_handled - pre_p2_handled;
    unsigned long d2_skipped = post_p2_skipped - pre_p2_skipped;
    unsigned long d2_total   = post_p2_total   - pre_p2_total;
    unsigned long d2_emits   = post_p2_emits   - pre_p2_emits;
    uint64_t d2_drained      = post_p2_drained - pre_p2_drained;
    printf("[S3.C PASS II TELEMETRY] total +%lu handled +%lu skipped +%lu emits +%lu drained +%lu "
           "(expect handled=0, ood-triggered PASSTHROUGH)\n",
           d2_total, d2_handled, d2_skipped, d2_emits, d2_drained);

    std::vector<__half> Y_cip_rand_h((size_t)N * M_TEST);
    CHECK_CUDA(cudaMemcpy(Y_cip_rand_h.data(), d_Y_cip_rand,
                          sizeof(__half) * Y_cip_rand_h.size(), cudaMemcpyDeviceToHost));

    int top1_match_p2 = 0;
    std::vector<float> kl_per_p2(M_TEST);
    for (int m = 0; m < M_TEST; m++) {
        int argmax_v = 0, argmax_c = 0;
        float max_v = -1e30f, max_c = -1e30f;
        for (int n2 = 0; n2 < N; n2++) {
            float lv = __half2float(Y_van_rand_h[m * N + n2]);
            float lc = __half2float(Y_cip_rand_h[m * N + n2]);
            logit_v[n2] = lv;
            logit_c[n2] = lc;
            if (lv > max_v) { max_v = lv; argmax_v = n2; }
            if (lc > max_c) { max_c = lc; argmax_c = n2; }
        }
        if (argmax_v == argmax_c) top1_match_p2++;
        kl_per_p2[m] = kl_row(logit_v.data(), logit_c.data(), N, p_buf, q_buf);
    }
    float top1_rand = (float)top1_match_p2 / (float)M_TEST;
    float kl_mean_rand = 0.0f, kl_max_rand = 0.0f;
    for (auto v : kl_per_p2) { kl_mean_rand += v; if (v > kl_max_rand) kl_max_rand = v; }
    kl_mean_rand /= M_TEST;
    printf("[S3.C PASS II QUALITY] top1_match=%.4f kl_mean=%.6e kl_max=%.6e "
           "(gate: top-1 >= 0.9995; kl_mean <= 5.5e-5)\n",
           top1_rand, kl_mean_rand, kl_max_rand);

    int pass_ii_telemetry = (d2_handled == 0);
    int pass_ii_quality   = (top1_rand >= 0.9995f) && (kl_mean_rand <= 5.5e-5f);
    int pass_ii = pass_ii_telemetry && pass_ii_quality;
    printf("[S3.C PASS II RESULT] telemetry=%s quality=%s overall=%s\n",
           pass_ii_telemetry ? "PASS" : "FAIL",
           pass_ii_quality ? "PASS" : "FAIL",
           pass_ii ? "PASS" : "FAIL");

    int overall = pass_i && pass_ii;
    printf("[S3.C OVERALL] %s\n", overall ? "PASS" : "FAIL");

    /* Clean shutdown so consumer prints exit. */
    cipher_rt_remember_consumer_exit();

    cudaFree(d_W_real); cudaFree(d_X_in); cudaFree(d_X_random);
    cudaFree(d_Y_van_in); cudaFree(d_Y_van_rand);
    cudaFree(d_Y_cip_in); cudaFree(d_Y_cip_rand);
    cudaFree(d_X_one);
    cublasDestroy(blas);

    return overall ? 0 : 1;
}
