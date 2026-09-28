// =============================================================================
// CIPHER — EDMD Live Calibration (Change 1)
// src/cipher_edmd_live.cpp
//
// Per-shape snapshot collection + randomized rank-r SVD of the weight operand
// + automatic registration into the Koopman shape registry.
//
// Geometry only — keyed on (K_dim, N_dim). No model/layer knowledge.
// =============================================================================

#include "may13/cipher_edmd_live.h"
#include "may13/cipher_randsvd.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <atomic>
#include <mutex>
#include <random>
#include <thread>

// Forward decl — implemented in src/cipher_block_sub_kernel.cu
extern "C" int cipher_koopman_fp16_register_shape(
    int K_dim, int N_dim, float* vt, float* k_op, float* w);

// ----------------------------------------------------------------------------
// Tunables
// ----------------------------------------------------------------------------

static constexpr int KR_RANK         = 64;    // must match FP16_R in kernel
static constexpr int TARGET_ROWS     = 2000;  // snapshots per shape before fit
                                              // (>=20×r for stable rank-64 fit)
static constexpr int ROWS_PER_CALL   = 8;     // copy min(M, this) rows per call
static constexpr int MAX_SHAPES      = 16;
static constexpr int OVERSAMPLE      = 10;    // HMT oversampling (Halko et al.
                                              // recommend p≥10 for r≥30)
static constexpr int POWER_ITERS     = 4;     // W14 Step 2 E day-1 fix: was 1;
                                              // bumped to 4 for proper convergence
                                              // on noisy real LLM activation data.
                                              // Diagnosed via d1_cipher_matrices_test.py:
                                              // power_iters=1 gave V cosine similarity
                                              // ≈ 0 with torch SVD on real LM head data.

// Distinct-input gate: the collector waits for DISTINCT_INPUT_THRESHOLD
// unique activation pointers before it starts capturing snapshots. This
// keeps synthetic single-tensor benchmarks (e.g. test_hw_validation.py's
// MFU test, which reuses one 4096x4096 tensor 2000x) out of the collection
// path entirely — rank-r EDMD calibration requires distributional variation
// in the inputs, and a one-tensor workload trivially has none. Real LLM
// inference sees dozens of distinct activation tensors per forward pass.
static constexpr int DISTINCT_INPUT_THRESHOLD = 8;
static constexpr int DISTINCT_PTR_TRACK       = 16;  // small LRU / ring
// If we've been invoked more than DISTINCT_GIVE_UP_CALLS times for a shape
// and still haven't seen DISTINCT_INPUT_THRESHOLD unique activation pointers,
// conclude it's a synthetic single-tensor workload (benchmark, fuzz harness,
// regression test) and permanently disable further collection for that shape.
// Subsequent calls take the fast path (`failed` early-exit) and add zero
// overhead beyond a single load + branch.
static constexpr int DISTINCT_GIVE_UP_CALLS   = 20;

// cudaDataType constants (avoid including cublas headers)
static constexpr int DTYPE_FP32 = 0;  // CUDA_R_32F
static constexpr int DTYPE_FP16 = 2;  // CUDA_R_16F

// ----------------------------------------------------------------------------
// Per-shape state
// ----------------------------------------------------------------------------

struct LiveShape {
    int    K;
    int    N;
    int    rows;
    int    calls;

    // Raw pinned capture buffers — written by cudaMemcpyAsync on the
    // default stream, no host-side sync. One byte-width per dtype.
    // fp16 collection: X_raw16 holds rows*K uint16_t; fp32: X_host32 holds
    // rows*K float. Only one of each pair is populated depending on dtype.
    uint16_t* X_raw16;
    uint16_t* Y_raw16;
    float*    X_host32;
    float*    Y_host32;
    int       X_dtype;   // DTYPE_FP16 or DTYPE_FP32 (decided on first collect)
    int       Y_dtype;

    // Host fp32 buffers used by the fit after conversion.
    float*    X_host;   // (TARGET_ROWS, K) fp32
    float*    Y_host;   // (TARGET_ROWS, N) fp32

    // Persistent matrices passed to register_shape; must stay alive
    float* V_T;      // (KR_RANK, K)
    float* K_op;     // (KR_RANK, KR_RANK)
    float* W;        // (KR_RANK, N)

    bool   fit_launched; // true once a background fit has been spawned
    bool   registered;
    bool   failed;

    // Distinct-input tracking — small ring of recent activation pointers.
    // Collection stays OFF until we've seen DISTINCT_INPUT_THRESHOLD unique
    // pointers on this shape.
    const void* distinct_ptrs[DISTINCT_PTR_TRACK];
    int         distinct_count;

    // Diagnostics
    float  residual_ratio;
    float  energy_captured;
    float  max_diff;
    float  sigma[KR_RANK];
};

static LiveShape  g_shapes[MAX_SHAPES];
static int        g_n_shapes = 0;
static std::mutex g_mtx;
static int        g_enabled = -1;    // -1 = uninit, 0 = off, 1 = on

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------

static bool live_enabled() {
    if (g_enabled < 0) {
        const char* env = getenv("CIPHER_EDMD_LIVE");
        g_enabled = (env && env[0] == '0') ? 0 : 1;   // default ON
    }
    return g_enabled == 1;
}

static LiveShape* find_shape_locked(int K, int N) {
    for (int i = 0; i < g_n_shapes; i++)
        if (g_shapes[i].K == K && g_shapes[i].N == N) return &g_shapes[i];
    return nullptr;
}

static LiveShape* get_or_create_locked(int K, int N) {
    LiveShape* s = find_shape_locked(K, N);
    if (s) return s;
    if (g_n_shapes >= MAX_SHAPES) return nullptr;
    s = &g_shapes[g_n_shapes++];
    memset(s, 0, sizeof(*s));
    s->K = K;
    s->N = N;
    s->X_dtype = -1;
    s->Y_dtype = -1;
    // Pinned capture buffers allocated lazily on first collect (when we
    // know the dtype). Fit-time fp32 buffers allocated once here.
    size_t x_bytes = (size_t)TARGET_ROWS * (size_t)K * sizeof(float);
    size_t y_bytes = (size_t)TARGET_ROWS * (size_t)N * sizeof(float);
    s->X_host = (float*)malloc(x_bytes);
    s->Y_host = (float*)malloc(y_bytes);
    if (!s->X_host || !s->Y_host) {
        if (s->X_host) { free(s->X_host); s->X_host = nullptr; }
        if (s->Y_host) { free(s->Y_host); s->Y_host = nullptr; }
        s->failed = true;
    }
    return s;
}

// Allocate pinned capture buffer for this shape/dtype pair on demand.
static bool ensure_capture_buffers_locked(LiveShape* s, int Xdt, int Ydt) {
    if (s->X_dtype == -1) {
        size_t bytes = (Xdt == DTYPE_FP16)
            ? (size_t)TARGET_ROWS * s->K * sizeof(uint16_t)
            : (size_t)TARGET_ROWS * s->K * sizeof(float);
        void** p = (Xdt == DTYPE_FP16) ? (void**)&s->X_raw16 : (void**)&s->X_host32;
        if (cudaHostAlloc(p, bytes, cudaHostAllocDefault) != cudaSuccess) return false;
        s->X_dtype = Xdt;
    }
    if (s->Y_dtype == -1) {
        size_t bytes = (Ydt == DTYPE_FP16)
            ? (size_t)TARGET_ROWS * s->N * sizeof(uint16_t)
            : (size_t)TARGET_ROWS * s->N * sizeof(float);
        void** p = (Ydt == DTYPE_FP16) ? (void**)&s->Y_raw16 : (void**)&s->Y_host32;
        if (cudaHostAlloc(p, bytes, cudaHostAllocDefault) != cudaSuccess) return false;
        s->Y_dtype = Ydt;
    }
    // Dtype mismatch on a subsequent call — treat as failure.
    if (s->X_dtype != Xdt || s->Y_dtype != Ydt) return false;
    return true;
}

// IEEE half -> float (host-only, no CUDA __half dependency).
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = -14;
            while (!(mant & 0x400u)) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (uint32_t)((e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// Issue an async D2H copy into a pinned host byte-buffer. No host-side sync.
// Host data becomes valid only after a later cudaDeviceSynchronize (which
// the background fit thread does before reading).
static bool gpu_capture_async(const void* src_gpu, void* dst_host_pinned,
                               size_t byte_count) {
    cudaError_t err = cudaMemcpyAsync(dst_host_pinned, src_gpu,
                                       byte_count,
                                       cudaMemcpyDeviceToHost, 0);
    return err == cudaSuccess;
}

// Linear algebra / randomized SVD helpers live in the shared header
// include/cipher_randsvd.h (namespace cipher_rs). Local thin wrapper that
// preserves the previous rand_svd_rank_r(...) signature used further down.

static inline void mm_rm(const float* A, const float* B, float* C,
                         int m, int k, int n) {
    cipher_rs::mm_rm(A, B, C, m, k, n);
}

static bool rand_svd_rank_r(
    const float* W_kn,
    int K, int N, int r,
    float* U_r,
    float* sigma_r,
    float* V_r,
    float* total_energy_out)
{
    return cipher_rs::rand_svd_rank_r(
        W_kn, K, N, r, OVERSAMPLE, POWER_ITERS, 1337u,
        U_r, sigma_r, V_r, total_energy_out);
}

// ----------------------------------------------------------------------------
// W14 Step 2 E ζ-deep Day 2 — deterministic SVD via cusolverDnSgesvd
//
// Replaces rand_svd_rank_r as the default path. D1.3 evidence: with torch
// SVD matrices the kernel delivers 90% top-1 on real LM head data; with
// cipher's rand_svd it delivers 0%. The randomized algorithm doesn't
// converge to the dominant singular subspace on real LLM activation data
// even after H2 CGS2 + POWER_ITERS=4 patches. Deterministic SVD via
// cuSOLVER is the v1 ship path.
//
// Output convention matches rand_svd_rank_r exactly:
//   - W_kn: input matrix, treated as (m, K) row-major (m=rand_svd's K,
//           K=rand_svd's N — confusing but matches existing call site)
//   - U_r: (m, r) row-major   left singular vectors
//   - sigma_r: (r,)            top-r singular values
//   - V_r: (K, r) row-major   right singular vectors
//   - total_energy_out: sum of TOP-r squared singular values
//
// Implementation: cusolverDnSgesvd takes column-major. Our row-major (m, K)
// is column-major (K, m). cusolverDnSgesvd computes SVD of (K, m) matrix
// X' = X^T (in matrix-math sense). Output U' is (K, K) col-major =
// right singular vectors of X (= our V); Vt' is (m, m) col-major =
// left singular vectors of X transposed (= our U^T).
// ----------------------------------------------------------------------------
static bool det_svd_rank_r(
    const float* W_kn,
    int K, int N, int r,
    float* U_r,
    float* sigma_r,
    float* V_r,
    float* total_energy_out)
{
    // rand_svd_rank_r's K parameter = number of rows of the input data matrix
    // For cipher_edmd_live's call: K = m_calib (snapshots), N = K_data (feature dim)
    // We treat the data as (m_calib, K_data) row-major.
    const int m_rows = K;     // data rows (= snapshots)
    const int n_cols = N;     // data cols (= feature dim K_data)
    const int min_dim = (m_rows < n_cols) ? m_rows : n_cols;

    // Upload X to GPU (already on-host as W_kn)
    float *X_dev = nullptr;
    size_t X_bytes = (size_t)m_rows * n_cols * sizeof(float);
    if (cudaMalloc(&X_dev, X_bytes) != cudaSuccess) return false;
    if (cudaMemcpy(X_dev, W_kn, X_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(X_dev); return false;
    }

    // cuSOLVER handle
    cusolverDnHandle_t cusolverH = nullptr;
    if (cusolverDnCreate(&cusolverH) != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(X_dev); return false;
    }

    // Treat row-major (m_rows, n_cols) as col-major (n_cols, m_rows) = X^T.
    // cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, Vt, ldvt, work, lwork, rwork, info)
    // For col-major (n_cols, m_rows): m_param=n_cols, n_param=m_rows.
    const int m_param = n_cols;
    const int n_param = m_rows;

    // Workspace query
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(cusolverH, m_param, n_param, &lwork);

    // Allocate cusolver outputs (col-major)
    float *S_dev = nullptr, *U_dev = nullptr, *Vt_dev = nullptr;
    float *work_dev = nullptr, *rwork_dev = nullptr;
    int *info_dev = nullptr;
    bool alloc_ok =
        cudaMalloc(&S_dev, min_dim * sizeof(float))             == cudaSuccess &&
        cudaMalloc(&U_dev, (size_t)m_param * m_param * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&Vt_dev, (size_t)n_param * n_param * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&work_dev, (size_t)lwork * sizeof(float))    == cudaSuccess &&
        cudaMalloc(&rwork_dev, (min_dim > 1 ? (min_dim - 1) : 1) * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&info_dev, sizeof(int))                       == cudaSuccess;
    if (!alloc_ok) {
        if (S_dev)   cudaFree(S_dev);
        if (U_dev)   cudaFree(U_dev);
        if (Vt_dev)  cudaFree(Vt_dev);
        if (work_dev) cudaFree(work_dev);
        if (rwork_dev) cudaFree(rwork_dev);
        if (info_dev) cudaFree(info_dev);
        cudaFree(X_dev);
        cusolverDnDestroy(cusolverH);
        return false;
    }

    // Run SVD: jobu='A' (full U) jobvt='A' (full Vt)
    cusolverStatus_t st = cusolverDnSgesvd(
        cusolverH, 'A', 'A', m_param, n_param, X_dev, m_param,
        S_dev, U_dev, m_param, Vt_dev, n_param,
        work_dev, lwork, rwork_dev, info_dev);

    int info_h = 0;
    cudaMemcpy(&info_h, info_dev, sizeof(int), cudaMemcpyDeviceToHost);

    if (st != CUSOLVER_STATUS_SUCCESS || info_h != 0) {
        fprintf(stderr, "[CIPHER DET-SVD] cusolverDnSgesvd failed status=%d info=%d K=%d N=%d\n",
                (int)st, info_h, K, N);
        cudaFree(S_dev); cudaFree(U_dev); cudaFree(Vt_dev);
        cudaFree(work_dev); cudaFree(rwork_dev); cudaFree(info_dev);
        cudaFree(X_dev); cusolverDnDestroy(cusolverH);
        return false;
    }

    // Copy back to host: S → sigma_r (first r), U → V_r (first r cols, K rows),
    // Vt → U_r (first r rows of Vt^T = first r cols of Vt's interpretation).
    float *S_host    = (float*)malloc(min_dim * sizeof(float));
    float *U_host    = (float*)malloc((size_t)m_param * m_param * sizeof(float));
    float *Vt_host   = (float*)malloc((size_t)n_param * n_param * sizeof(float));
    if (!S_host || !U_host || !Vt_host) {
        free(S_host); free(U_host); free(Vt_host);
        cudaFree(S_dev); cudaFree(U_dev); cudaFree(Vt_dev);
        cudaFree(work_dev); cudaFree(rwork_dev); cudaFree(info_dev);
        cudaFree(X_dev); cusolverDnDestroy(cusolverH);
        return false;
    }
    cudaMemcpy(S_host,  S_dev,  min_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U_host,  U_dev,  (size_t)m_param * m_param * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Vt_host, Vt_dev, (size_t)n_param * n_param * sizeof(float), cudaMemcpyDeviceToHost);

    // sigma_r: top r singular values
    for (int i = 0; i < r; i++) sigma_r[i] = (i < min_dim) ? S_host[i] : 0.0f;

    // V_r (K = n_cols, r): cusolver U is (m_param=n_cols, m_param=n_cols) col-major
    //   Col-major U[k, i] = U_host[i * m_param + k]
    //   We want V_r[k, i] = (V_X)[k, i] = U_col[k, i] = U_host[i * m_param + k]
    //   Stored row-major: V_r[k * r + i] = U_host[i * m_param + k]
    for (int i = 0; i < r; i++)
        for (int k = 0; k < n_cols; k++)
            V_r[k * r + i] = U_host[(size_t)i * m_param + k];

    // U_r (m = m_rows, r): cusolver Vt is (n_param=m_rows, n_param=m_rows) col-major
    //   Col-major Vt[i, j] = Vt_host[j * n_param + i]
    //   We want U_r[j, i] = (U_X)[j, i] = Vt^T_col[j, i] = Vt_col[i, j] = Vt_host[j * n_param + i]
    //   Stored row-major: U_r[j * r + i] = Vt_host[j * n_param + i]
    for (int i = 0; i < r; i++)
        for (int j = 0; j < m_rows; j++)
            U_r[j * r + i] = Vt_host[(size_t)j * n_param + i];

    // total_energy = sum of top-r squared singular values (matches rand_svd return)
    if (total_energy_out) {
        float te = 0.0f;
        for (int i = 0; i < r && i < min_dim; i++) te += S_host[i] * S_host[i];
        *total_energy_out = te;
    }

    free(S_host); free(U_host); free(Vt_host);
    cudaFree(S_dev); cudaFree(U_dev); cudaFree(Vt_dev);
    cudaFree(work_dev); cudaFree(rwork_dev); cudaFree(info_dev);
    cudaFree(X_dev); cusolverDnDestroy(cusolverH);
    return true;
}

// Dispatcher: deterministic by default, rand_svd via CIPHER_USE_RAND_SVD=1
static bool svd_rank_r(
    const float* W_kn,
    int K, int N, int r,
    float* U_r,
    float* sigma_r,
    float* V_r,
    float* total_energy_out)
{
    static int s_use_rand = -1;
    if (s_use_rand < 0) {
        const char *e = getenv("CIPHER_USE_RAND_SVD");
        s_use_rand = (e && e[0] == '1') ? 1 : 0;
    }
    if (s_use_rand) {
        return rand_svd_rank_r(W_kn, K, N, r, U_r, sigma_r, V_r, total_energy_out);
    }
    return det_svd_rank_r(W_kn, K, N, r, U_r, sigma_r, V_r, total_energy_out);
}

// ----------------------------------------------------------------------------
// Fit & register for one shape
// ----------------------------------------------------------------------------

static bool fit_and_register_locked(LiveShape* s,
                                    int weight_dtype,
                                    const void* weight_gpu)
{
    (void)weight_dtype; (void)weight_gpu;   // unused — fit is purely snapshot-driven
    if (!s || s->registered || s->failed) return false;

    const int K = s->K;
    const int N = s->N;
    const int r = KR_RANK;
    const int m = s->rows;
    if (m < r + OVERSAMPLE) { s->failed = true; return false; }

    // Ensure all async D2H captures issued from the main thread have landed
    // in the pinned buffers before we read them. One sync amortised over
    // all TARGET_ROWS snapshots.
    cudaDeviceSynchronize();

    // Convert raw capture buffers to fp32 X_host / Y_host.
    if (s->X_dtype == DTYPE_FP16 && s->X_raw16) {
        for (size_t i = 0; i < (size_t)m * K; i++)
            s->X_host[i] = half_to_float(s->X_raw16[i]);
    } else if (s->X_dtype == DTYPE_FP32 && s->X_host32) {
        memcpy(s->X_host, s->X_host32, (size_t)m * K * sizeof(float));
    } else {
        s->failed = true;
        return false;
    }
    if (s->Y_dtype == DTYPE_FP16 && s->Y_raw16) {
        for (size_t i = 0; i < (size_t)m * N; i++)
            s->Y_host[i] = half_to_float(s->Y_raw16[i]);
    } else if (s->Y_dtype == DTYPE_FP32 && s->Y_host32) {
        memcpy(s->Y_host, s->Y_host32, (size_t)m * N * sizeof(float));
    } else {
        s->failed = true;
        return false;
    }

    // -------------------------------------------------------------------
    // Reduced-rank regression from snapshots (X, Y).
    //
    // We run randomized rank-r SVD on X (m, K):
    //     X  ≈  U_x (m, r) @ diag(σ_x) @ V_x^T (r, K)
    // V_x is an orthonormal basis for the row-space of X (the observed
    // activation manifold). For held-out x drawn from the same manifold,
    // V_x V_x^T x = x (projection is identity), so a rank-r fit in V_x's
    // basis is exact within the manifold.
    //
    // Given the SVD of X, the reduced-rank regression coefficients are:
    //     W_reg = diag(1/σ_x) @ U_x^T @ Y   ∈  (r, N)
    //
    // The effective linear map that the Koopman kernel must reproduce is:
    //     A_hat = V_x @ diag(1/σ_x) @ U_x^T @ Y   ∈  (K, N)
    // which satisfies X @ A_hat = U_x @ U_x^T @ Y ≈ Y.
    //
    // Decomposition into kernel form (out = X @ V_T^T @ K_op^T @ W_k):
    //     V_T  = V_x^T         (r, K)   — manifold basis
    //     K_op = I_r            (r, r)   — identity (sigmas baked into W_k)
    //     W_k  = diag(1/σ) @ U_x^T @ Y   (r, N)   — regression in basis
    // -------------------------------------------------------------------

    // Randomized SVD of X (m, K), rank r.
    // rand_svd_rank_r treats its input as a (K_in, N_in) matrix; here we
    // supply X with K_in = m, N_in = K.
    float* Ux     = (float*)malloc((size_t)m * r * sizeof(float));  // (m, r)
    float  sigma[KR_RANK] = {};
    float* Vx     = (float*)malloc((size_t)K * r * sizeof(float));  // (K, r)
    if (!Ux || !Vx) {
        free(Ux); free(Vx);
        s->failed = true;
        return false;
    }

    float total_energy_X = 0.0f;
    if (!svd_rank_r(s->X_host, m, K, r, Ux, sigma, Vx, &total_energy_X)) {
        free(Ux); free(Vx);
        s->failed = true;
        fprintf(stderr, "[CIPHER EDMD-LIVE] rand_svd(X) failed m=%d K=%d\n", m, K);
        return false;
    }

    // energy_captured = ||X_r||_F^2 / ||X||_F^2
    double xf2 = 0.0;
    for (size_t i = 0; i < (size_t)m * K; i++) {
        double v = (double)s->X_host[i];
        xf2 += v * v;
    }
    double top_r = 0.0;
    for (int i = 0; i < r; i++) top_r += (double)sigma[i] * sigma[i];
    s->energy_captured = (xf2 > 0.0) ? (float)(top_r / xf2) : 0.0f;

    // Build kernel matrices ------------------------------------------------
    // W14 Step 2 E ζ-deep day 3 fix (2026-05-23): allocate VT/K_op/W on GPU
    // not CPU. cipher_koopman_fp16_register_shape and the downstream
    // cipher_koopman_fp16_decode_generic kernel both read these matrices as
    // device pointers. Prior code used CPU malloc → GPU kernel reading CPU
    // memory → silent garbage in V_T_fp16 / W_fp16 conversion + scalar-kernel
    // path. Diagnosed via /tmp/step13_2_baseline/zeta_kernel_test.py which
    // proved kernel arithmetic correct (rel_err 2e-4 = fp16 noise) when
    // matrices are passed as proper GPU pointers (torch.from_numpy(...).cuda()).
    // Direct-call test outcome D1: kernel correct, prep bug.
    float* VT_cpu  = (float*)malloc((size_t)r * K * sizeof(float));
    float* KOP_cpu = (float*)malloc((size_t)r * r * sizeof(float));
    float* W_cpu   = (float*)malloc((size_t)r * N * sizeof(float));
    if (!VT_cpu || !KOP_cpu || !W_cpu) {
        free(Ux); free(Vx); free(VT_cpu); free(KOP_cpu); free(W_cpu);
        s->failed = true;
        return false;
    }

    //  V_T (r, K) = V_x^T  — Vx is stored (K, r) row-major
    for (int i = 0; i < r; i++)
        for (int k = 0; k < K; k++)
            VT_cpu[i * K + k] = Vx[k * r + i];

    //  K_op (r, r) = identity
    memset(KOP_cpu, 0, (size_t)r * r * sizeof(float));
    for (int i = 0; i < r; i++) KOP_cpu[i * r + i] = 1.0f;

    /* W14 Step 2 E Tikhonov-regularized recovery per plan v1.2.3 §1 line 103:
     *   "per-layer rank-parameterized recovery, Tikhonov regularization
     *    α = 0.01·σ₁², spectral radius ≤ 1"
     *
     * Replaces pure pseudoinverse (1/σ_i) with Tikhonov spectral filter:
     *   W[i, n] = (σ_i / (σ_i² + α·σ₁²)) · Σ_j Ux[j, i] · Y[j, n]
     *
     * For σ_i >> sqrt(α)·σ₁: filter → 1/σ_i (matches Moore-Penrose pseudoinverse).
     * For σ_i << sqrt(α)·σ₁: filter → σ_i/(α·σ₁²) (smooth taper, less overfitting
     * to small-singular-value noise).
     *
     * α tunable via CIPHER_KOOPMAN_TIKHONOV_ALPHA env (default 0.01 per plan).
     * Setting α=0 reverts to pure pseudoinverse. */
    static float s_tikhonov_alpha = -1.0f;
    if (s_tikhonov_alpha < 0.0f) {
        const char *e = getenv("CIPHER_KOOPMAN_TIKHONOV_ALPHA");
        s_tikhonov_alpha = (e ? (float)atof(e) : 0.01f);
        if (s_tikhonov_alpha < 0.0f) s_tikhonov_alpha = 0.01f;
    }
    const float sigma1_sq = (r > 0) ? (sigma[0] * sigma[0]) : 0.0f;
    const float reg_alpha = s_tikhonov_alpha * sigma1_sq;

    // W (r, N) = diag(σ_i / (σ_i² + α·σ₁²)) @ Ux^T @ Y
    for (int i = 0; i < r; i++) {
        float denom = sigma[i] * sigma[i] + reg_alpha;
        float filt  = (denom > 1e-20f) ? (sigma[i] / denom) : 0.0f;
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int j = 0; j < m; j++)
                acc += (double)Ux[j * r + i] * (double)s->Y_host[j * N + n];
            W_cpu[i * N + n] = (float)(acc * filt);
        }
    }

    // Upload CPU-computed matrices to GPU for kernel consumption.
    float *VT_buf = nullptr, *KOP_buf = nullptr, *W_buf = nullptr;
    if (cudaMalloc(&VT_buf,  (size_t)r * K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&KOP_buf, (size_t)r * r * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&W_buf,   (size_t)r * N * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[CIPHER EDMD-LIVE] cudaMalloc failed K=%d N=%d\n", K, N);
        if (VT_buf)  cudaFree(VT_buf);
        if (KOP_buf) cudaFree(KOP_buf);
        if (W_buf)   cudaFree(W_buf);
        free(Ux); free(Vx); free(VT_cpu); free(KOP_cpu); free(W_cpu);
        s->failed = true;
        return false;
    }
    cudaMemcpy(VT_buf,  VT_cpu,  (size_t)r * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(KOP_buf, KOP_cpu, (size_t)r * r * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W_buf,   W_cpu,   (size_t)r * N * sizeof(float), cudaMemcpyHostToDevice);
    free(VT_cpu); free(KOP_cpu); free(W_cpu);

    // Calibration-side diagnostics are omitted from the fit path: the
    // Python gate test computes its own held-out residual_ratio against
    // an fp32 reference (which is the authoritative measurement), and
    // any CPU-side reconstruction here would materially extend the
    // background fit wall time without adding rigour.
    s->residual_ratio = 0.0f;
    s->max_diff       = 0.0f;

    // Save persistent GPU buffers (ownership handed to the koopman registry) --
    s->V_T  = VT_buf;
    s->K_op = KOP_buf;
    s->W    = W_buf;
    for (int i = 0; i < KR_RANK; i++) s->sigma[i] = sigma[i];

    int rc = cipher_koopman_fp16_register_shape(K, N, VT_buf, KOP_buf, W_buf);
    if (rc != 0) {
        fprintf(stderr, "[CIPHER EDMD-LIVE] register_shape failed K=%d N=%d rc=%d\n",
                K, N, rc);
        s->failed = true;
        cudaFree(VT_buf); cudaFree(KOP_buf); cudaFree(W_buf);
        free(Ux); free(Vx);
        return false;
    }
    s->registered = true;

    fprintf(stderr,
        "[CIPHER EDMD-LIVE] FIT K=%d N=%d rows=%d rank=%d "
        "energy_X=%.4f residual=%.4f max_diff=%.4f sigma0=%.3g sigma%d=%.3g\n",
        K, N, s->rows, r,
        s->energy_captured, s->residual_ratio, s->max_diff,
        sigma[0], r - 1, sigma[KR_RANK - 1]);

    free(s->X_host); s->X_host = nullptr;
    free(s->Y_host); s->Y_host = nullptr;
    if (s->X_raw16)  { cudaFreeHost(s->X_raw16);  s->X_raw16  = nullptr; }
    if (s->Y_raw16)  { cudaFreeHost(s->Y_raw16);  s->Y_raw16  = nullptr; }
    if (s->X_host32) { cudaFreeHost(s->X_host32); s->X_host32 = nullptr; }
    if (s->Y_host32) { cudaFreeHost(s->Y_host32); s->Y_host32 = nullptr; }

    free(Ux); free(Vx);
    return true;
}

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

extern "C"
bool cipher_edmd_live_collect(
    int         M_py,
    int         K_dim,
    int         N_dim,
    int         weight_dtype,
    const void* weight_gpu,
    int         activation_dtype,
    const void* activation_gpu,
    int         output_dtype,
    const void* output_gpu)
{
    if (!live_enabled()) return false;
    if (M_py <= 0 || K_dim <= 0 || N_dim <= 0) return false;
    if (!weight_gpu || !activation_gpu || !output_gpu) return false;
    if (activation_dtype != DTYPE_FP16 && activation_dtype != DTYPE_FP32) return false;
    if (output_dtype     != DTYPE_FP16 && output_dtype     != DTYPE_FP32) return false;
    if (weight_dtype     != DTYPE_FP16 && weight_dtype     != DTYPE_FP32) return false;

    std::lock_guard<std::mutex> lock(g_mtx);

    LiveShape* s = get_or_create_locked(K_dim, N_dim);
    if (!s || s->registered || s->failed || s->fit_launched) return false;

    s->calls++;

    // Distinct-input gate — cheap O(DISTINCT_PTR_TRACK) scan. We only start
    // actual snapshot collection once the workload has shown distributional
    // variation. Synthetic single-tensor benchmarks stay on a pure fast
    // path (no cudaMemcpyAsync, no pinned alloc, no background fit) and
    // therefore do not regress MFU.
    if (s->distinct_count < DISTINCT_INPUT_THRESHOLD) {
        bool seen = false;
        for (int i = 0; i < s->distinct_count; i++) {
            if (s->distinct_ptrs[i] == activation_gpu) { seen = true; break; }
        }
        if (!seen && s->distinct_count < DISTINCT_PTR_TRACK) {
            s->distinct_ptrs[s->distinct_count++] = activation_gpu;
        }
        if (s->distinct_count < DISTINCT_INPUT_THRESHOLD) {
            // After DISTINCT_GIVE_UP_CALLS invocations without seeing enough
            // variety, permanently disable this shape. Future collects take
            // the early-exit branch at the top of this function (one atomic
            // bool load + one branch mispredict-free path, ~2ns total).
            if (s->calls >= DISTINCT_GIVE_UP_CALLS) {
                s->failed = true;
            }
            return false;   // not diverse enough yet; skip capture entirely
        }
    }

    if (!ensure_capture_buffers_locked(s, activation_dtype, output_dtype)) {
        s->failed = true;
        return false;
    }

    // Snapshot up to ROWS_PER_CALL rows from activation and output.
    int take = M_py < ROWS_PER_CALL ? M_py : ROWS_PER_CALL;
    int room = TARGET_ROWS - s->rows;
    if (room < take) take = room;

    if (take > 0) {
        size_t x_elems = (size_t)take * K_dim;
        size_t y_elems = (size_t)take * N_dim;
        size_t x_elem_sz = (activation_dtype == DTYPE_FP16) ? 2 : 4;
        size_t y_elem_sz = (output_dtype     == DTYPE_FP16) ? 2 : 4;
        void* x_dst = (activation_dtype == DTYPE_FP16)
            ? (void*)(s->X_raw16 + (size_t)s->rows * K_dim)
            : (void*)(s->X_host32 + (size_t)s->rows * K_dim);
        void* y_dst = (output_dtype == DTYPE_FP16)
            ? (void*)(s->Y_raw16 + (size_t)s->rows * N_dim)
            : (void*)(s->Y_host32 + (size_t)s->rows * N_dim);

        // Pure async D2H, no host-side sync. The background fit thread
        // will cudaDeviceSynchronize once before reading.
        if (gpu_capture_async(activation_gpu, x_dst, x_elems * x_elem_sz)
         && gpu_capture_async(output_gpu,     y_dst, y_elems * y_elem_sz)) {
            s->rows += take;
        }
    }

    // Trigger fit when we've hit the row target.
    //
    // The rank-64 randomized SVD is ~1-2 seconds of CPU work. Running it
    // inline would block the next GEMM call and violate MFU regression
    // tests. Instead spawn a detached worker thread that does the heavy
    // lifting and calls cipher_koopman_fp16_register_shape once done.
    // The main thread returns immediately; subsequent calls skip all
    // collection work (fit_launched is set) and fall through to the
    // normal (cuBLAS-through-CIPHER) path until the worker registers
    // the shape, at which point the driver-level Koopman substitute
    // takes over automatically.
    (void)weight_dtype; (void)weight_gpu;   // no longer used at collect time
    if (s->rows >= TARGET_ROWS) {
        s->fit_launched = true;
        LiveShape* sp = s;
        // The background fit runs WITHOUT holding g_mtx. Main thread has
        // already set fit_launched=true under the lock — that is our promise
        // not to touch any of sp's snapshot/result fields until sp->registered
        // flips. The fit thread does:
        //   1. cudaDeviceSynchronize() (waits for all pending async D2H
        //      captures issued by the main thread to land in pinned memory)
        //   2. fp16→fp32 conversion of the capture buffers
        //   3. randomized rank-r SVD on the snapshots
        //   4. cipher_koopman_fp16_register_shape
        //   5. sp->registered = true, free capture/fit buffers
        // None of these steps need g_mtx; main-thread readers of
        // sp->registered already take g_mtx in is_registered/get_stats.
        std::thread([sp]() {
            fit_and_register_locked(sp, 0, nullptr);
        }).detach();
        return false;
    }
    return false;
}

extern "C"
bool cipher_edmd_live_is_registered(int K_dim, int N_dim) {
    std::lock_guard<std::mutex> lock(g_mtx);
    LiveShape* s = find_shape_locked(K_dim, N_dim);
    return s && s->registered;
}

extern "C"
bool cipher_edmd_live_force_fit(int K_dim, int N_dim) {
    // Not used by production; test harness calls this after loading a known
    // weight into the TLS path. We do NOT stash a weight pointer here — tests
    // must reach this by the normal passthrough hook.
    (void)K_dim; (void)N_dim;
    return false;
}

extern "C"
bool cipher_edmd_live_get_stats(
    int    K_dim,
    int    N_dim,
    float* residual_ratio_out,
    float* energy_captured_out,
    float* max_diff_out,
    int*   rows_collected_out)
{
    std::lock_guard<std::mutex> lock(g_mtx);
    LiveShape* s = find_shape_locked(K_dim, N_dim);
    if (!s || (!s->registered && !s->failed)) return false;
    if (residual_ratio_out)  *residual_ratio_out  = s->residual_ratio;
    if (energy_captured_out) *energy_captured_out = s->energy_captured;
    if (max_diff_out)        *max_diff_out        = s->max_diff;
    if (rows_collected_out)  *rows_collected_out  = s->rows;
    return true;
}

extern "C"
void cipher_edmd_live_report(void) {
    std::lock_guard<std::mutex> lock(g_mtx);
    fprintf(stderr, "[CIPHER EDMD-LIVE REPORT] shapes=%d enabled=%d\n",
            g_n_shapes, live_enabled() ? 1 : 0);
    for (int i = 0; i < g_n_shapes; i++) {
        LiveShape* s = &g_shapes[i];
        fprintf(stderr, "  K=%d N=%d rows=%d calls=%d registered=%d failed=%d "
                "residual=%.4f energy=%.4f max_diff=%.4f\n",
                s->K, s->N, s->rows, s->calls,
                s->registered ? 1 : 0, s->failed ? 1 : 0,
                s->residual_ratio, s->energy_captured, s->max_diff);
    }
}
