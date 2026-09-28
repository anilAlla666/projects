// =============================================================================
// CIPHER — Shared randomized SVD helpers (Halko-Martinsson-Tropp)
// include/cipher_randsvd.h
//
// Header-only library of small linear-algebra primitives used by both the
// EDMD-live shape calibrator (src/cipher_edmd_live.cpp) and the attention
// Koopman fit path (src/cipher_attn_koopman.cpp). All routines operate on
// row-major fp32 on the CPU. Helpers are `static inline` so each translation
// unit that includes this header gets its own copy — no ODR violations,
// no link-time gymnastics.
//
// Functions:
//   cipher_rs::mm_rm          C = A · B  row-major
//   cipher_rs::mm_tn          C = A^T · B  (A stored m×k row-major)
//   cipher_rs::mgs_qr_thin    thin QR via modified Gram-Schmidt (in place)
//   cipher_rs::jacobi_symm    cyclic Jacobi eigendecomposition (symmetric)
//   cipher_rs::rand_svd_rank_r  rank-r randomized SVD of W (K×N) → U_r, σ, V_r
// =============================================================================

#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

namespace cipher_rs {

// ---------- matmul: C (m×n) = A (m×k) · B (k×n) ----------
static inline void mm_rm(const float* A, const float* B, float* C,
                         int m, int k, int n) {
    for (int i = 0; i < m * n; i++) C[i] = 0.0f;
    for (int i = 0; i < m; i++) {
        const float* arow = &A[i * k];
        float* crow = &C[i * n];
        for (int p = 0; p < k; p++) {
            float a = arow[p];
            const float* brow = &B[p * n];
            for (int j = 0; j < n; j++) crow[j] += a * brow[j];
        }
    }
}

// ---------- matmul-transpose: C (k×n) = A^T · B, A is (m,k) row-major ----------
static inline void mm_tn(const float* A, const float* B, float* C,
                         int m, int k, int n) {
    for (int i = 0; i < k * n; i++) C[i] = 0.0f;
    for (int p = 0; p < m; p++) {
        const float* arow = &A[p * k];
        const float* brow = &B[p * n];
        for (int i = 0; i < k; i++) {
            float a = arow[i];
            float* crow = &C[i * n];
            for (int j = 0; j < n; j++) crow[j] += a * brow[j];
        }
    }
}

// ---------- thin QR via modified Gram-Schmidt (in place, orthonormal columns) ----------
static inline void mgs_qr_thin(float* A, int m, int n) {
    for (int k = 0; k < n; k++) {
        float norm = 0.0f;
        for (int i = 0; i < m; i++) {
            float v = A[i * n + k];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm < 1e-20f) {
            for (int i = 0; i < m; i++) A[i * n + k] = 0.0f;
            if (k < m) A[k * n + k] = 1.0f;
            continue;
        }
        float inv = 1.0f / norm;
        for (int i = 0; i < m; i++) A[i * n + k] *= inv;
        for (int j = k + 1; j < n; j++) {
            float dot = 0.0f;
            for (int i = 0; i < m; i++) dot += A[i * n + k] * A[i * n + j];
            for (int i = 0; i < m; i++) A[i * n + j] -= dot * A[i * n + k];
        }
    }
}

// ---------- cyclic Jacobi eigendecomposition of symmetric n×n ----------
// On exit: A diagonal holds eigenvalues, V holds eigenvectors in columns.
static inline void jacobi_symm(float* A, float* V, int n) {
    for (int i = 0; i < n * n; i++) V[i] = 0.0f;
    for (int i = 0; i < n; i++) V[i * n + i] = 1.0f;

    for (int sweep = 0; sweep < 100; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += A[i * n + j] * A[i * n + j];
        if (off < 1e-20f) return;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float apq = A[p * n + q];
                if (std::fabs(apq) < 1e-14f) continue;
                float app = A[p * n + p];
                float aqq = A[q * n + q];
                float theta = (aqq - app) / (2.0f * apq);
                float t;
                if (std::fabs(theta) > 1e6f) t = 0.5f / theta;
                else if (theta >= 0.0f)
                    t =  1.0f / ( theta + std::sqrt(theta * theta + 1.0f));
                else
                    t = -1.0f / (-theta + std::sqrt(theta * theta + 1.0f));
                float c = 1.0f / std::sqrt(1.0f + t * t);
                float s = t * c;

                A[p * n + p] = app - t * apq;
                A[q * n + q] = aqq + t * apq;
                A[p * n + q] = 0.0f;
                A[q * n + p] = 0.0f;
                for (int i = 0; i < n; i++) {
                    if (i == p || i == q) continue;
                    float aip = A[i * n + p];
                    float aiq = A[i * n + q];
                    float new_ip = c * aip - s * aiq;
                    float new_iq = s * aip + c * aiq;
                    A[i * n + p] = new_ip;
                    A[p * n + i] = new_ip;
                    A[i * n + q] = new_iq;
                    A[q * n + i] = new_iq;
                }
                for (int i = 0; i < n; i++) {
                    float vip = V[i * n + p];
                    float viq = V[i * n + q];
                    V[i * n + p] = c * vip - s * viq;
                    V[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
}

// ---------- rank-r randomized SVD of W (K × N) row-major ----------
// Produces: U_r (K, r), σ_r (r,), V_r (N, r) s.t.  W ≈ U_r · diag(σ) · V_r^T
// Returns false on allocation failure or degenerate sizes.
static inline bool rand_svd_rank_r(
    const float* W_kn,
    int K, int N, int r,
    int oversample, int power_iters,
    unsigned rng_seed,
    float* U_r,
    float* sigma_r,
    float* V_r,
    float* total_energy_out)
{
    const int l = r + oversample;
    if (l > K || l > N || l <= 0) return false;

    float* Omega = (float*)std::malloc((size_t)N * l * sizeof(float));
    float* Y     = (float*)std::malloc((size_t)K * l * sizeof(float));
    float* Z     = (float*)std::malloc((size_t)N * l * sizeof(float));
    float* Btil  = (float*)std::malloc((size_t)l * N * sizeof(float));
    float* BBt   = (float*)std::malloc((size_t)l * l * sizeof(float));
    float* Ub    = (float*)std::malloc((size_t)l * l * sizeof(float));

    if (!Omega || !Y || !Z || !Btil || !BBt || !Ub) {
        std::free(Omega); std::free(Y); std::free(Z);
        std::free(Btil); std::free(BBt); std::free(Ub);
        return false;
    }

    std::mt19937 rng(rng_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < N * l; i++) Omega[i] = nd(rng);

    mm_rm(W_kn, Omega, Y, K, N, l);

    /* W14 Step 2 E ζ-deep H2 fix (2026-05-23): apply MGS TWICE per call ("CGS2"
     * / "twice is enough" — Giraud, Langou, Rozloznik 2005). Single-pass MGS
     * loses orthogonality when input columns are near-linearly-dependent (e.g.
     * when oversample l = r + 10 exceeds true rank(X)). Loss-of-orthogonality
     * causes ||Y^T · X||_F^2 > ||X||_F^2 and inflates the rand_svd sigma
     * spectrum, producing energy_X > 1.0 and σ values up to √2 too large.
     * Diagnosed at LM head shape (K=2048, N=32000) where true rank ≤ 64 and
     * oversample-induced near-linear-dependency caused σ_0 = 4435 vs true 3137.
     * Verified fix in /tmp/step13_2_baseline/h2_randsvd_clone.py. */
    for (int it = 0; it < power_iters; it++) {
        mgs_qr_thin(Y, K, l);
        mgs_qr_thin(Y, K, l);
        mm_tn(W_kn, Y, Z, K, N, l);
        mgs_qr_thin(Z, N, l);
        mgs_qr_thin(Z, N, l);
        mm_rm(W_kn, Z, Y, K, N, l);
    }
    mgs_qr_thin(Y, K, l);   // final orthonormal Q (K, l), pass 1
    mgs_qr_thin(Y, K, l);   // final orthonormal Q (K, l), pass 2 (CGS2)

    mm_tn(Y, W_kn, Btil, K, l, N);

    for (int i = 0; i < l; i++) {
        for (int j = 0; j < l; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++)
                acc += (double)Btil[i * N + k] * (double)Btil[j * N + k];
            BBt[i * l + j] = (float)acc;
        }
    }
    jacobi_symm(BBt, Ub, l);

    struct Pair { float lam; int idx; };
    Pair* pairs = (Pair*)std::malloc(sizeof(Pair) * l);
    for (int i = 0; i < l; i++) {
        pairs[i].lam = BBt[i * l + i];
        pairs[i].idx = i;
    }
    for (int i = 1; i < l; i++) {
        Pair x = pairs[i];
        int j = i - 1;
        while (j >= 0 && pairs[j].lam < x.lam) {
            pairs[j + 1] = pairs[j];
            j--;
        }
        pairs[j + 1] = x;
    }

    float total_energy = 0.0f;
    for (int i = 0; i < l; i++) {
        float lam = pairs[i].lam;
        if (lam < 0.0f) lam = 0.0f;
        total_energy += lam;
    }
    if (total_energy_out) *total_energy_out = total_energy;

    for (int rank = 0; rank < r; rank++) {
        int ci = pairs[rank].idx;
        float lam = pairs[rank].lam;
        if (lam < 0.0f) lam = 0.0f;
        float sig = std::sqrt(lam);
        sigma_r[rank] = sig;

        for (int row = 0; row < K; row++) {
            double acc = 0.0;
            for (int k = 0; k < l; k++)
                acc += (double)Y[row * l + k] * (double)Ub[k * l + ci];
            U_r[row * r + rank] = (float)acc;
        }

        if (sig > 1e-20f) {
            float inv = 1.0f / sig;
            for (int n = 0; n < N; n++) {
                double acc = 0.0;
                for (int k = 0; k < l; k++)
                    acc += (double)Btil[k * N + n] * (double)Ub[k * l + ci];
                V_r[n * r + rank] = (float)(acc * inv);
            }
        } else {
            for (int n = 0; n < N; n++) V_r[n * r + rank] = 0.0f;
        }
    }

    std::free(pairs);
    std::free(Omega); std::free(Y); std::free(Z);
    std::free(Btil); std::free(BBt); std::free(Ub);
    return true;
}

} // namespace cipher_rs
