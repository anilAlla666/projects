// cipher_fp8_fused_quant.cu — single-launch fp16 → fp8 quantize via grid-sync.
//
// Replaces the prior 3-kernel pipeline (cipher_fp8_absmax + finalize + quant).
// Uses cooperative_groups grid_group::sync() to coordinate the absmax → scale
// → quantize phases inside ONE kernel launch on H100 (sm_90).
//
// Scale convention: writes scale_out[0] = absmax / 448.0 (matches the existing
// CIPHER FP8 path and cuBLAS Lt's BScalePtr expectation: fp8 × scale → fp16).
//
// Block size 256, capped at MAX_BLOCKS to fit the H100 cooperative-launch
// resident-block budget. Each thread strides over the input twice — once for
// absmax, once for quant — both passes are HBM-bandwidth bound.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
// 132 SMs × 4 resident blocks/SM. Conservative; lets cooperative-launch fit
// on H100 without occupancy worries (BLOCK_SIZE×blocks_per_sm well under
// MaxThreadsPerSM = 2048).
#define MAX_BLOCKS 528

extern "C" __global__ void cipher_fused_fp16_to_fp8_kernel(
    const __half*    __restrict__ input,
    __nv_fp8_e4m3*   __restrict__ output,
    float*           __restrict__ scale_out,
    int*             __restrict__ absmax_bits,   // 4 B device scratch
    int                            numel)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ float sdata[BLOCK_SIZE];

    int tid    = (int)threadIdx.x;
    int gtid   = (int)blockIdx.x * BLOCK_SIZE + tid;
    int gstride = BLOCK_SIZE * (int)gridDim.x;

    // ── Phase 0: zero global absmax (block 0, thread 0) ─────────────────
    if (blockIdx.x == 0 && tid == 0) {
        *absmax_bits = 0;
    }
    grid.sync();

    // ── Phase 1: per-thread → per-block absmax → atomicMax to global ───
    float local = 0.0f;
    for (int i = gtid; i < numel; i += gstride) {
        float v  = __half2float(input[i]);
        float av = v < 0.0f ? -v : v;
        if (av > local) local = av;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            if (b > a) sdata[tid] = b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Bitwise atomicMax on the IEEE-754 representation works because
        // absmax values are non-negative and IEEE float ordering matches
        // unsigned-integer ordering for non-negatives.
        atomicMax(absmax_bits, __float_as_int(sdata[0]));
    }
    grid.sync();

    // ── Phase 2: compute scale = absmax / 448.0 (block 0, thread 0) ────
    if (blockIdx.x == 0 && tid == 0) {
        float am   = __int_as_float(*absmax_bits);
        scale_out[0] = (am > 0.0f) ? (am / 448.0f) : 1.0f;
    }
    grid.sync();

    // ── Phase 3: quantize using the resolved scale ─────────────────────
    float scale = scale_out[0];
    float inv   = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
    for (int i = gtid; i < numel; i += gstride) {
        float v = __half2float(input[i]) * inv;
        if (v >  448.0f) v =  448.0f;
        if (v < -448.0f) v = -448.0f;
        output[i] = __nv_fp8_e4m3(v);
    }
}

// Host-side launcher. Sets the cooperative-launch func attr (one-shot)
// and launches via cudaLaunchCooperativeKernel.
//
// Returns 1 on success, 0 on failure (caller falls back to passthrough).
extern "C" int cipher_fused_fp16_to_fp8(
    const void*  input_,
    void*        output_,
    float*       scale_out,
    int          numel,
    int*         absmax_workspace,    // 4 B device scratch (caller-owned)
    cudaStream_t stream)
{
    if (!input_ || !output_ || !scale_out || !absmax_workspace || numel <= 0)
        return 0;
    const __half*  input  = (const __half*)input_;
    __nv_fp8_e4m3* output = (__nv_fp8_e4m3*)output_;

    int needed_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks = needed_blocks < MAX_BLOCKS ? needed_blocks : MAX_BLOCKS;
    if (blocks < 1) blocks = 1;

    void* args[] = {
        (void*)&input,
        (void*)&output,
        (void*)&scale_out,
        (void*)&absmax_workspace,
        (void*)&numel,
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (const void*)cipher_fused_fp16_to_fp8_kernel,
        dim3(blocks),
        dim3(BLOCK_SIZE),
        args,
        /*sharedMem*/ 0,
        stream);
    return err == cudaSuccess ? 1 : 0;
}
