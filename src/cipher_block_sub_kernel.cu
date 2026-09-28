// =============================================================================
// CIPHER — Block-level O(Kr) substitute kernel (Session 5, optimized)
// cipher_block_sub_kernel.cu
//
// V stored transposed as V_T (r×K, row-major) for coalesced reads.
// Float4 vectorized loads. Warp shuffle reduction for dot products.
//
// Grid: 1 block, 256 threads.
// Shared memory: alpha[r] + alpha_hat[r] = 64 bytes.
// =============================================================================

#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#else
#  include <cuda_runtime.h>
#endif

#include <stdio.h>

#define BLOCK_SUB_K  4096
#define BLOCK_SUB_R  8

// K / 4 (float4 elements per row)
#define BLOCK_SUB_K4 (BLOCK_SUB_K / 4)
// float4s per thread: 1024 / 256 = 4
#define F4_PER_THREAD (BLOCK_SUB_K4 / 256)

// ---------------------------------------------------------------------------
// Warp shuffle reduction — sum across 32 threads in a warp
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Optimized kernel: transposed V + float4 vectorized loads + warp shuffle
//
// V_T layout: V_T[j * K + i] = V_original[i][j]  (r rows of K floats each)
// x_in:  K floats, contiguous
// x_out: K floats, contiguous
//
// Phase 1: alpha[j] = dot(V_T[j,:], x_in)  for j=0..r-1
//   256 threads, each loads 4 float4s (16 floats) per dot product
//   Warp shuffle to reduce within warp, then atomic to shared
//
// Phase 2: alpha_hat = K_block @ alpha  (r=8, trivial, single thread)
//
// Phase 3: x_out[i] = dot(W_block[i,:], alpha_hat)
//   256 threads, each handles 16 output elements
// ---------------------------------------------------------------------------

__global__ void cipher_block_sub_kernel(
    const float* __restrict__ x_in,
    float*       __restrict__ x_out,
    const float* __restrict__ V_T,      // r x K, row-major (transposed)
    const float* __restrict__ K_block,  // r x r, row-major
    const float* __restrict__ W_block,  // K x r, row-major
    int K, int r)
{
    __shared__ float alpha[BLOCK_SUB_R];
    __shared__ float alpha_hat[BLOCK_SUB_R];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;

    // ── Phase 1: alpha[j] = V_T[j,:] . x_in  (vectorized + shuffle) ─────
    // 256 threads = 8 warps. Each warp handles one alpha[j] (r=8).
    // Each thread in a warp loads 4 float4s = 16 floats, computes partial dot.
    // 32 threads * 16 floats = 512 floats per warp — need 4096/512 = 8 passes?
    // No: 32 threads * 4 float4s * 4 = 512, but K=4096 needs 1024 float4s.
    // 1024 / 32 = 32 float4s per thread.

    if (warp_id < r) {
        const float4* vt_row = (const float4*)(V_T + warp_id * K);
        const float4* x4 = (const float4*)x_in;
        float partial = 0.0f;

        // Each of 32 threads handles 1024/32 = 32 float4s
        #pragma unroll 8
        for (int k = lane_id; k < BLOCK_SUB_K4; k += 32) {
            float4 v4 = vt_row[k];
            float4 xi = x4[k];
            partial += v4.x * xi.x + v4.y * xi.y + v4.z * xi.z + v4.w * xi.w;
        }

        // Warp shuffle reduction
        partial = warp_reduce_sum(partial);

        if (lane_id == 0)
            alpha[warp_id] = partial;
    }
    __syncthreads();

    // ── Phase 2: alpha_hat = K_block @ alpha  (r=8, trivial) ─────────────
    if (tid < r) {
        float acc = 0.0f;
        for (int j = 0; j < r; j++)
            acc += K_block[tid * r + j] * alpha[j];
        alpha_hat[tid] = acc;
    }
    __syncthreads();

    // ── Phase 3: x_out = W_block @ alpha_hat ─────────────────────────────
    // 256 threads, each handles K/256 = 16 output elements
    {
        const int CHUNK = BLOCK_SUB_K / 256;  // 16
        int base = tid * CHUNK;
        for (int i = base; i < base + CHUNK; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < BLOCK_SUB_R; j++)
                acc += W_block[i * r + j] * alpha_hat[j];
            x_out[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Device pointers — set by cipher_block_sub_gpu_init() in cipher_edmd.cpp
// ---------------------------------------------------------------------------

static float* d_V_T     = nullptr;  // transposed V (r×K)
static float* d_K_block = nullptr;
static float* d_W_block = nullptr;
static bool   s_gpu_ready = false;

extern "C"
void cipher_block_sub_gpu_set_ptrs(float* v_t, float* k, float* w) {
    d_V_T = v_t;
    d_K_block = k;
    d_W_block = w;
    s_gpu_ready = true;
    fprintf(stderr, "[CIPHER BLOCK-GPU] Device ptrs set: V_T=%p K=%p W=%p\n",
            (void*)v_t, (void*)k, (void*)w);
}

extern "C"
void cipher_block_sub_gpu_launch(const float* x_in_dev, float* x_out_dev) {
    if (!s_gpu_ready) return;
    cipher_block_sub_kernel<<<1, 256>>>(
        x_in_dev, x_out_dev,
        d_V_T, d_K_block, d_W_block,
        BLOCK_SUB_K, BLOCK_SUB_R);
}

// =============================================================================
// Session 7: fp16 fused Koopman decode kernel
//
// Single kernel: fp16 input → fp32 project → evolve → readout → fp16 output
// One block per row m. 256 threads cooperatively compute alpha[16] via
// warp shuffle, then each thread writes 16 output elements.
//
// r=16, K=4096, threads=256
// Phase 1: alpha[j] = sum_k x[m,k] * V[k,j]  — 16 warp reductions
// Phase 2: alpha2[j] = sum_i alpha[i] * K[i,j] — 16 threads
// Phase 3: out[m,n] = sum_j alpha2[j] * W[j,n] — 256 threads × 16 outputs
// =============================================================================

#include <cuda_fp16.h>

#define FP16_R  16
#define FP16_K  4096

__device__ __forceinline__ float warp_reduce_sum_16(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void cipher_koopman_fp16_decode(
    const __half* __restrict__ x,       // (M, 4096)
    __half*       __restrict__ out,     // (M, 4096)
    const float*  __restrict__ V_T,     // (16, 4096) row-major (transposed V)
    const float*  __restrict__ K_op,    // (16, 16)
    const float*  __restrict__ W,       // (16, 4096) row-major
    int M)
{
    int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;        // 0..255
    const int warp_id = tid / 32;       // 0..7
    const int lane_id = tid & 31;

    __shared__ float alpha[FP16_R];
    __shared__ float alpha2[FP16_R];

    // ── Phase 1: alpha[j] = V_T[j,:] . x[m,:]  ─────────────────────────
    // 256 threads = 8 warps. r=16 needs 16 reductions.
    // Each warp handles 2 alpha values (warp_id*2 and warp_id*2+1).
    // 32 threads per warp, each accumulates 4096/32 = 128 elements.

    const __half* x_row = x + m * FP16_K;

    for (int pass = 0; pass < 2; pass++) {
        int j = warp_id * 2 + pass;
        if (j < FP16_R) {
            const float* vt_row = V_T + j * FP16_K;
            float partial = 0.0f;
            // Each lane handles elements lane_id, lane_id+32, lane_id+64, ...
            for (int k = lane_id; k < FP16_K; k += 32) {
                partial += __half2float(x_row[k]) * vt_row[k];
            }
            partial = warp_reduce_sum_16(partial);
            if (lane_id == 0)
                alpha[j] = partial;
        }
    }
    __syncthreads();

    // ── Phase 2: alpha2 = K_op @ alpha  (16×16, trivial) ────────────────
    if (tid < FP16_R) {
        float acc = 0.0f;
        for (int j = 0; j < FP16_R; j++)
            acc += K_op[tid * FP16_R + j] * alpha[j];
        alpha2[tid] = acc;
    }
    __syncthreads();

    // ── Phase 3: out[m,n] = W[:,n] . alpha2  ────────────────────────────
    // 256 threads, each writes 4096/256 = 16 output elements
    {
        const int CHUNK = FP16_K / 256;  // 16
        int base = tid * CHUNK;
        for (int n = base; n < base + CHUNK; n++) {
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP16_R; j++)
                acc += alpha2[j] * W[j * FP16_K + n];
            out[m * FP16_K + n] = __float2half(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer device pointers (32 layers × 3 matrices each)
// ---------------------------------------------------------------------------

#define MAX_LAYERS 32

struct LayerPtrs {
    float* V_T;   // (16, 4096)
    float* K_op;  // (16, 16)
    float* W;     // (16, 4096)
};

static LayerPtrs s_layers[MAX_LAYERS] = {};
static int       s_n_layers = 0;
static bool      s_fp16_ready = false;

// Legacy single-layer API (backward compat)
extern "C"
void cipher_koopman_fp16_set_ptrs(float* vt, float* k, float* w) {
    s_layers[0].V_T  = vt;
    s_layers[0].K_op = k;
    s_layers[0].W    = w;
    if (s_n_layers < 1) s_n_layers = 1;
    s_fp16_ready = true;
}

// Per-layer API
extern "C"
void cipher_koopman_fp16_set_layer(int layer_idx, float* vt, float* k, float* w) {
    if (layer_idx < 0 || layer_idx >= MAX_LAYERS) return;
    s_layers[layer_idx].V_T  = vt;
    s_layers[layer_idx].K_op = k;
    s_layers[layer_idx].W    = w;
    if (layer_idx >= s_n_layers) s_n_layers = layer_idx + 1;
    s_fp16_ready = true;
}

// Launch with explicit layer selection
extern "C"
int cipher_koopman_fp16_launch_layer(const void* x_fp16, void* out_fp16, int M, int layer_idx) {
    if (!s_fp16_ready || layer_idx < 0 || layer_idx >= s_n_layers) return -1;
    LayerPtrs* lp = &s_layers[layer_idx];
    if (!lp->V_T || !lp->K_op || !lp->W) return -1;
    cipher_koopman_fp16_decode<<<M, 256>>>(
        (const __half*)x_fp16, (__half*)out_fp16,
        lp->V_T, lp->K_op, lp->W, M);
    return 0;
}

// Legacy launch (uses layer 0)
extern "C"
int cipher_koopman_fp16_launch(const void* x_fp16, void* out_fp16, int M) {
    return cipher_koopman_fp16_launch_layer(x_fp16, out_fp16, M, 0);
}
