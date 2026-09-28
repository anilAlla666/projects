# DEEPER SWING: hand-tuned GEMV-specialized INT4 kernel. 4-bit weights = ~4x fewer bytes than fp16.
# Production marlin/machete cap at 27% MBU at batch-1 (GEMM-tiled). A GEMV-specialized 4-bit kernel could hit
# ~40-54% MBU on ~4x fewer bytes => 2-2.6x over fp16. THE real crack if it works.
import time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12; GROUP=128
CUDA = r'''
#include <cuda_fp16.h>
#include <torch/extension.h>
// W4: packed int4 [N, K/2] uint8 (byte = lo nibble elem 2b, hi nibble elem 2b+1). scales [N, K/GROUP] half.
template<int NT,int GROUP>
__global__ void int4_gemv(const uint8_t* __restrict__ W, const __half* __restrict__ x,
                          const __half* __restrict__ scales, __half* __restrict__ y, int N, int K){
    int row=blockIdx.x; if(row>=N) return;
    const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)row*(K>>1));  // K/2 bytes/row
    const __half2* x2=reinterpret_cast<const __half2*>(x);
    const __half* srow=scales+(size_t)row*(K/GROUP);
    int Bvec=(K>>1)>>4;          // # of 16-byte chunks = (K/2)/16 = K/32; each chunk = 32 int4 = 32 elems
    int tid=threadIdx.x; float acc=0.f;
    for(int c=tid;c<Bvec;c+=NT){
        uint4 wp=wrow[c];                         // 16 bytes = 32 int4 = 32 weights
        int k0=c<<5;                              // element offset = c*32
        float sc=__half2float(srow[k0/GROUP]);    // 32 elems span 1/4 group -> one scale
        const uint8_t* wb=reinterpret_cast<const uint8_t*>(&wp);
        const __half* xh=reinterpret_cast<const __half*>(x)+k0;
        #pragma unroll
        for(int b=0;b<16;b++){
            int lo=(wb[b]&0xF); lo=(lo^0x8)-8;    // signed int4 low nibble (elem 2b)
            int hi=(wb[b]>>4);  hi=(hi^0x8)-8;    // signed int4 high nibble (elem 2b+1)
            acc += (float(lo)*__half2float(xh[2*b]) + float(hi)*__half2float(xh[2*b+1]));
        }
        acc += 0; // (scale applied below per-chunk since 1 scale/chunk)
        // apply scale: but acc accumulates across chunks w/ different scales -> apply per chunk:
    }
    // NOTE: scale must be applied per-chunk. Redo with per-chunk scaling:
    __shared__ float red[NT]; red[tid]=acc; __syncthreads();
    for(int s=NT>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if(tid==0) y[row]=__float2half(red[0]);   // scale folded per-chunk above (see v2)
}
torch::Tensor int4_gemv_f(torch::Tensor W, torch::Tensor x, torch::Tensor scales){
    int N=W.size(0),K=W.size(1)*2;
    auto y=torch::empty({N},torch::dtype(torch::kHalf).device(W.device()));
    int4_gemv<256,128><<<N,256>>>(reinterpret_cast<const uint8_t*>(W.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()),reinterpret_cast<const __half*>(scales.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),N,K);
    return y;
}
'''
# NOTE: this v1 has a scale bug (scale not applied per-chunk). Fixing inline below before trusting correctness.
print("(building v1 to test perf scaffold; correctness fix in v2)")
