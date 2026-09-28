// Fused decode GEMV + ABFT output checksum, M=1.  (v2: per-block partials, no global atomics)
//   y[n]    = sum_k A[k] * W[n,k]          (W row-major [N,K], the decode linear)
//   rowsum  = sum_n (fp32 acc[n])          (fp32 output checksum; detects GEMM COMPUTE SDC.
//                                           HBM storage SDC is ECC-covered on H100.)
//   ref     = sum_k A[k] * w_ref[k]        (w_ref[k]=sum_n W[n,k], precomputed once, fp32)
//   residual= |sum(partial_rowsum) - ref|  (final 128-elem reduce read OUT-OF-BAND)
// Per-block partial sums written to partial_rs[blockIdx] (OVERWRITE -> no memset, no atomics).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define WARP 32

// MODE: 0 = no checksum; 1 = rowsum partials only; 2 = rowsum + reference
template<int MODE>
__global__ void gemv_kernel(const __half* __restrict__ A,
                            const __half* __restrict__ W,
                            const float*  __restrict__ w_ref,
                            __half* __restrict__ y,
                            float* __restrict__ partial_rs,  // [gridDim.x]
                            float* __restrict__ partial_ref, // [gridDim.x] (distributed ref partials)
                            int K, int N)
{
    int wpb = blockDim.x / WARP;
    int warp_in_blk = threadIdx.x / WARP;
    int warp_id = blockIdx.x * wpb + warp_in_blk;
    int lane = threadIdx.x & (WARP-1);
    float acc = 0.f;
    if (warp_id < N) {
        const __half* wrow = W + (size_t)warp_id * K;
        for (int k = lane; k < K; k += WARP)
            acc += __half2float(A[k]) * __half2float(wrow[k]);
        #pragma unroll
        for (int o = WARP/2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) y[warp_id] = __float2half(acc);   // acc now = y[warp_id] (lane0 only)
    }
    if (MODE >= 1) {
        // block-reduce the per-warp acc (lane0 holds it) into one partial per block
        __shared__ float sh[32];
        if (lane == 0) sh[warp_in_blk] = (warp_id < N) ? acc : 0.f;
        __syncthreads();
        if (warp_in_blk == 0) {
            float v = (lane < wpb) ? sh[lane] : 0.f;
            #pragma unroll
            for (int o = WARP/2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
            if (lane == 0) partial_rs[blockIdx.x] = v;     // OVERWRITE: no memset needed
        }
        // reference A.w_ref DISTRIBUTED across the whole grid (same launch, no serial tail):
        // each thread owns a grid-strided K-slice; block-reduce -> partial_ref[blockIdx] (gridDim.x).
        // A[] is already L2-resident from the GEMV; w_ref[] (fp32, K) is read once across the grid.
        // Overlaps the GEMV's W-bandwidth wall -> the compute piggybacks for ~free.
        if (MODE >= 2) {
            float r = 0.f;
            for (int k = blockIdx.x*blockDim.x + threadIdx.x; k < K; k += gridDim.x*blockDim.x)
                r += __half2float(A[k]) * w_ref[k];
            #pragma unroll
            for (int o = WARP/2; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffff, r, o);
            __shared__ float shr[32];
            if (lane == 0) shr[warp_in_blk] = r;
            __syncthreads();
            if (warp_in_blk == 0) {
                float v = (lane < wpb) ? shr[lane] : 0.f;
                #pragma unroll
                for (int o = WARP/2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
                if (lane == 0) partial_ref[blockIdx.x] = v;   // OVERWRITE: no memset needed
            }
        }
    }
}

// Standalone parallel reference GEMV: ref = sum_k A[k]*w_ref[k], split-K across blocks.
// Each block reduces its K-slice -> partial_ref[blockIdx] (overwrite). Final reduce out-of-band.
__global__ void ref_kernel(const __half* __restrict__ A, const float* __restrict__ w_ref,
                           float* __restrict__ partial_ref, int K)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    float r = 0.f;
    for (int k = idx; k < K; k += stride) r += __half2float(A[k]) * w_ref[k];
    int lane = threadIdx.x & (WARP-1);
    #pragma unroll
    for (int o = WARP/2; o>0; o>>=1) r += __shfl_down_sync(0xffffffff, r, o);
    __shared__ float sh[32];
    if (lane==0) sh[threadIdx.x/WARP]=r;
    __syncthreads();
    if (threadIdx.x < WARP){
        float v = (threadIdx.x < blockDim.x/WARP)? sh[threadIdx.x]:0.f;
        #pragma unroll
        for (int o=WARP/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
        if (threadIdx.x==0) partial_ref[blockIdx.x]=v;
    }
}

// Prefill reference: ref[m] = sum_k A[m,k]*w_ref[k], A row-major [M,K]. One warp per row.
// fp16 A read (real traffic), fp32 w_ref, fp32 accumulate.
__global__ void ref_m_kernel(const __half* __restrict__ A, const float* __restrict__ w_ref,
                             float* __restrict__ ref_out, int M, int K)
{
    int wpb = blockDim.x/WARP;
    int m = blockIdx.x*wpb + threadIdx.x/WARP;
    int lane = threadIdx.x & (WARP-1);
    if (m >= M) return;
    const __half* arow = A + (size_t)m*K;
    float acc=0.f;
    for (int k=lane;k<K;k+=WARP) acc += __half2float(arow[k])*w_ref[k];
    #pragma unroll
    for (int o=WARP/2;o>0;o>>=1) acc += __shfl_down_sync(0xffffffff,acc,o);
    if (lane==0) ref_out[m]=acc;
}

extern "C" {
int gemv_nblocks(int N){ int threads=256; int wpb=threads/WARP; return (N+wpb-1)/wpb; }
int ref_nblocks(){ return 64; }
void launch_ref(const __half* A, const float* w_ref, float* partial_ref, int K, cudaStream_t s){
    ref_kernel<<<64,256,0,s>>>(A,w_ref,partial_ref,K);
}
void launch_ref_m(const __half* A, const float* w_ref, float* ref_out, int M, int K, cudaStream_t s){
    int threads=256, wpb=threads/WARP, blocks=(M+wpb-1)/wpb;
    ref_m_kernel<<<blocks,threads,0,s>>>(A,w_ref,ref_out,M,K);
}
void launch_gemv(const __half* A, const __half* W, const float* w_ref,
                 __half* y, float* partial_rs, float* partial_ref,
                 int K, int N, int do_checksum, cudaStream_t s)
{
    int threads = 256;
    int blocks = (N + threads/WARP - 1) / (threads/WARP);
    if      (do_checksum==2) gemv_kernel<2><<<blocks,threads,0,s>>>(A,W,w_ref,y,partial_rs,partial_ref,K,N);
    else if (do_checksum==1) gemv_kernel<1><<<blocks,threads,0,s>>>(A,W,w_ref,y,partial_rs,partial_ref,K,N);
    else                     gemv_kernel<0><<<blocks,threads,0,s>>>(A,W,w_ref,y,partial_rs,partial_ref,K,N);
}
}
