// Tuned fp16-read / fp32-accum coalesced row-reduction gemv, + a peak-read-bandwidth probe.
// rowgemv: y[m] = sum_k X[m,k]*w[k]  for X row-major [M,K] (row contiguous => fully coalesced).
//   This is the FAVORABLE reduction direction and serves BOTH Freivalds terms that gate the
//   best case:  v=A@u  (X=A,w=u, reduce over K)  and  Cg=C@g  (X=C,w=g, reduce over N).
//   One warp per output row; each lane strides the row with float4 (8 fp16) vectorized loads;
//   warp-shuffle reduction; fp32 accumulate.  This is the bandwidth lower bound for the A-read.
// read_bw: pure streaming fp16 read (grid-stride float4 sum) -> achievable HBM read BW at this clock.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define WARP 32

__global__ void rowgemv_kernel(const __half* __restrict__ X, const __half* __restrict__ w,
                               float* __restrict__ y, int M, int K) {
  int warp_id = (blockIdx.x * (blockDim.x / WARP)) + (threadIdx.x / WARP);
  int lane    = threadIdx.x % WARP;
  if (warp_id >= M) return;
  const __half* row = X + (size_t)warp_id * K;
  float acc = 0.f;
  // float4 = 8 fp16 per load; lane l handles elements [l*8 .. l*8+7], stride WARP*8
  int k4 = K / 8;                  // number of 8-wide chunks
  const float4* row4 = reinterpret_cast<const float4*>(row);
  const float4* w4   = reinterpret_cast<const float4*>(w);
  for (int c = lane; c < k4; c += WARP) {
    float4 xv = row4[c];
    float4 wv = w4[c];
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    const __half* wh = reinterpret_cast<const __half*>(&wv);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc += __half2float(xh[i]) * __half2float(wh[i]);
  }
  // tail (K not multiple of 8)
  for (int k = k4 * 8 + lane; k < K; k += WARP) acc += __half2float(row[k]) * __half2float(w[k]);
  // warp reduce
  #pragma unroll
  for (int o = WARP/2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[warp_id] = acc;
}

extern "C" int rowgemv(const void* X, const void* w, void* y, int M, int K, void* stream_v) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_v);
  int warps_per_block = 8;               // 256 threads
  int tpb = warps_per_block * WARP;
  int blocks = (M + warps_per_block - 1) / warps_per_block;
  rowgemv_kernel<<<blocks, tpb, 0, st>>>(reinterpret_cast<const __half*>(X),
      reinterpret_cast<const __half*>(w), reinterpret_cast<float*>(y), M, K);
  return 0;
}

// ---- peak read bandwidth probe ----
__global__ void read_kernel(const __half* __restrict__ X, size_t n8, float* __restrict__ out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  const float4* X4 = reinterpret_cast<const float4*>(X);
  for (size_t c = i; c < n8; c += stride) {
    float4 v = X4[c];
    const __half* h = reinterpret_cast<const __half*>(&v);
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc += __half2float(h[j]);
  }
  if (acc == -1.0f) out[i % 1024] = acc;   // prevent DCE
}
extern "C" int read_bw(const void* X, long n_elems, void* out, void* stream_v) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_v);
  size_t n8 = (size_t)n_elems / 8;
  int tpb = 256; int blocks = 132 * 16;    // saturate H100 (132 SMs)
  read_kernel<<<blocks, tpb, 0, st>>>(reinterpret_cast<const __half*>(X), n8,
                                      reinterpret_cast<float*>(out));
  return 0;
}
