// =============================================================================
// CIPHER — Change 3: Fused Attention Koopman — CUDA Kernel
// src/cipher_attn_koopman_kernel.cu
//
// One thread block per query row (M rows total). The fused pipeline runs
// entirely in shared memory with no global intermediates:
//
//   Phase 1:  alpha[j]   = sum_k V_T[j,k] · Q[m,k]     (k ∈ [0, d))
//   Phase 2:  alpha2[j]  = sum_i K_op[j,i] · alpha[i]  (i ∈ [0, r))
//   Phase 3:  weights[j] = softmax_j(alpha2)           (over r dims)
//   Phase 4:  out[m,k]   = sum_j weights[j] · V_comp[j,k]   (j ∈ [0, r))
//
// Cost: O(r·d) per row, independent of seq_len N. For typical decode with
// M=1, r=64, d=128 this is ~16K multiply-adds — two orders of magnitude
// below the O(N·d) cost of naïve attention at long context.
// =============================================================================

#include "cipher_attn_koopman.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

// Tunable: per-block shared memory budget for alpha / alpha2 / weights
#define CIPHER_ATTN_MAX_R  128   // fits r=64 or r=128, 512 bytes each buffer

__global__ void cipher_attn_fused_kernel(
    const float* __restrict__ Q,            // (M, d)
    const float* __restrict__ V_T,          // (r, d)   row-major
    const float* __restrict__ K_op,         // (r, r)   row-major
    const float* __restrict__ V_compressed, // (r, d)   row-major
    float*       __restrict__ out,          // (M, d)
    int M, int r, int d)
{
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;  // block size should be >= max(r, d)

    extern __shared__ float smem[];
    float* alpha   = smem;                 // [r]
    float* alpha2  = alpha   + CIPHER_ATTN_MAX_R;  // [r]
    float* weights = alpha2  + CIPHER_ATTN_MAX_R;  // [r]

    // ── Phase 1: alpha[j] = V_T[j, :] · Q[m, :] ───────────────────────────
    // One thread per j ∈ [0, r); each loops over d.
    if (tid < r) {
        const float* vt_row = V_T + tid * d;
        const float* q_row  = Q  + m * d;
        float acc = 0.0f;
        for (int k = 0; k < d; k++) acc += vt_row[k] * q_row[k];
        alpha[tid] = acc;
    }
    __syncthreads();

    // ── Phase 2: alpha2[j] = sum_i K_op[j, i] · alpha[i] ──────────────────
    if (tid < r) {
        const float* k_row = K_op + tid * r;
        float acc = 0.0f;
        for (int i = 0; i < r; i++) acc += k_row[i] * alpha[i];
        alpha2[tid] = acc;
    }
    __syncthreads();

    // ── Phase 3: row-wise softmax over r ──────────────────────────────────
    // Reduce max(alpha2), then sum(exp(alpha2 - max)), then normalize.
    // r is small (≤128) so one thread can do this serially in tid==0.
    if (tid == 0) {
        float amax = alpha2[0];
        for (int i = 1; i < r; i++) if (alpha2[i] > amax) amax = alpha2[i];
        float sum = 0.0f;
        for (int i = 0; i < r; i++) {
            float e = expf(alpha2[i] - amax);
            weights[i] = e;
            sum += e;
        }
        float inv = (sum > 1e-30f) ? (1.0f / sum) : 0.0f;
        for (int i = 0; i < r; i++) weights[i] *= inv;
    }
    __syncthreads();

    // ── Phase 4: out[m, k] = sum_j weights[j] · V_compressed[j, k] ────────
    // One thread per k ∈ [0, d); each loops over r.
    if (tid < d) {
        float acc = 0.0f;
        for (int j = 0; j < r; j++)
            acc += weights[j] * V_compressed[j * d + tid];
        out[m * d + tid] = acc;
    }
}

// -----------------------------------------------------------------------------
// V_compressed = U_x^T · V_cache   where U_x is (N, r), V_cache is (N, d).
// Result is (r, d). Implemented as a straightforward kernel rather than
// cuBLAS to keep Change 3 self-contained. For N=16K, r=64, d=128 this is
// one ~130M-FLOP launch, one-time per prefill.
// -----------------------------------------------------------------------------

__global__ void cipher_attn_precompute_vcompressed_kernel(
    const float* __restrict__ U_x,          // (N, r) row-major
    const float* __restrict__ V_cache,      // (N, d) row-major
    float*       __restrict__ V_compressed, // (r, d) row-major
    int N, int r, int d)
{
    // grid: (r × d), one thread per output element
    const int j = blockIdx.x;  // ∈ [0, r)
    const int k = threadIdx.x; // ∈ [0, d)
    if (j >= r || k >= d) return;

    double acc = 0.0;
    for (int n = 0; n < N; n++) {
        acc += (double)U_x[n * r + j] * (double)V_cache[n * d + k];
    }
    V_compressed[j * d + k] = (float)acc;
}

// =============================================================================
// Public C API (called from src/cipher_attn_koopman.cpp and the gate test)
// =============================================================================

extern "C"
int cipher_attn_fused_launch_fp32(
    const float* Q,
    const float* V_T,
    const float* K_op,
    const float* V_compressed,
    float*       out,
    int          M,
    int          r,
    int          d,
    void*        stream)
{
    if (!Q || !V_T || !K_op || !V_compressed || !out) return -1;
    if (M <= 0 || r <= 0 || d <= 0) return -2;
    if (r > CIPHER_ATTN_MAX_R) return -3;

    const int block = 256;
    // Block must cover max(r, d) threads (phases 1/2 use tid<r, phase 4 uses tid<d).
    // 256 comfortably covers r ≤ 128 and d ∈ {64, 96, 128}.
    (void)block;

    // Shared memory: 3 × r floats
    size_t shmem_bytes = (size_t)(3 * CIPHER_ATTN_MAX_R) * sizeof(float);

    dim3 grid((unsigned)M);
    dim3 blk(256);
    cipher_attn_fused_kernel<<<grid, blk, shmem_bytes, (cudaStream_t)stream>>>(
        Q, V_T, K_op, V_compressed, out, M, r, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CIPHER-ATTN] fused launch error: %s\n", cudaGetErrorString(err));
        return -4;
    }
    return 0;
}

extern "C"
int cipher_attn_precompute_vcompressed(
    const float* U_x,
    const float* V_cache,
    float*       V_compressed,
    int          N, int r, int d,
    void*        stream)
{
    if (!U_x || !V_cache || !V_compressed) return -1;
    if (N <= 0 || r <= 0 || d <= 0) return -2;
    if (d > 512) return -3;   // block size ceiling

    dim3 grid((unsigned)r);
    dim3 blk((unsigned)d);
    cipher_attn_precompute_vcompressed_kernel<<<grid, blk, 0, (cudaStream_t)stream>>>(
        U_x, V_cache, V_compressed, N, r, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CIPHER-ATTN] precompute launch error: %s\n", cudaGetErrorString(err));
        return -4;
    }
    return 0;
}
