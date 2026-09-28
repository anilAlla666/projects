#!/usr/bin/env python3
# Hand-tuned CUDA FP8 GEMV (batch-1 decode) — the genuine swing at beating cutlass's 54% MBU.
# Vectorized 16-byte fp8 loads (uint4) + hardware fp8->half conversion + block-per-row coalesced reduction.
# Benchmark vs cuBLAS fp16 (74% MBU bar) and torch._scaled_mm (cutlass FP8, 54%) on real Mistral shapes.
import os, time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12

CUDA = r'''
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

template<int NT>
__global__ void fp8_gemv_kernel(const __nv_fp8_e4m3* __restrict__ W, const __half* __restrict__ x,
                                const __half* __restrict__ scale, __half* __restrict__ y, int N, int K){
    int row = blockIdx.x; if(row>=N) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(W + (size_t)row*K);
    const uint4* xv4  = reinterpret_cast<const uint4*>(x);    // 8 halves per uint4
    int tid = threadIdx.x;
    int Kvec = K >> 4;                                        // # of 16-fp8 chunks
    float acc = 0.f;
    for(int c = tid; c < Kvec; c += NT){
        uint4 wp = wrow[c];                                   // 16 fp8 bytes, coalesced
        const __nv_fp8_e4m3* wb = reinterpret_cast<const __nv_fp8_e4m3*>(&wp);
        int k0 = c << 4;
        uint4 xa = xv4[(k0)>>3];                              // x[k0 .. k0+7]
        uint4 xb = xv4[(k0+8)>>3];                            // x[k0+8 .. k0+15]
        const __half* xa_h = reinterpret_cast<const __half*>(&xa);
        const __half* xb_h = reinterpret_cast<const __half*>(&xb);
        #pragma unroll
        for(int i=0;i<8;i++)  acc += float(wb[i])   * __half2float(xa_h[i]);
        #pragma unroll
        for(int i=0;i<8;i++)  acc += float(wb[8+i]) * __half2float(xb_h[i]);
    }
    __shared__ float red[NT];
    red[tid]=acc; __syncthreads();
    for(int s=NT>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if(tid==0) y[row] = __float2half(red[0] * __half2float(scale[row]));
}

torch::Tensor fp8_gemv(torch::Tensor W, torch::Tensor x, torch::Tensor scale){
    int N = W.size(0), K = W.size(1);
    auto y = torch::empty({N}, torch::dtype(torch::kHalf).device(W.device()));
    const int NT=256;
    fp8_gemv_kernel<NT><<<N, NT>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(W.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()), N, K);
    return y;
}
'''
mod = load_inline(name="fp8gemv", cpp_sources="torch::Tensor fp8_gemv(torch::Tensor,torch::Tensor,torch::Tensor);",
                  cuda_sources=CUDA, functions=["fp8_gemv"], extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_90a"], verbose=False)

SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float(); s=(w.abs().amax(1)/448.0).clamp(min=1e-6)
    W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous(); SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
# correctness vs fp16 ref
k='down';o,i=SH[k]; ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
got=mod.fp8_gemv(W8[k],x[k],SC[k])
print("correctness rel_err:", ((got.float()-ref.float()).norm()/ref.float().norm()).item())
def bench(fn,it=60,wm=20):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it
b16=sum(o*i*2 for o,i in SH.values())*L; b8=sum(o*i+o*4 for o,i in SH.values())*L
def run_cublas16():
    for _ in range(L):
        for k,(o,i) in SH.items(): torch.nn.functional.linear(x[k].unsqueeze(0),W16[k])
def run_ours():
    for _ in range(L):
        for k,(o,i) in SH.items(): mod.fp8_gemv(W8[k],x[k],SC[k])
tc=bench(run_cublas16); to=bench(run_ours)
print(f"cuBLAS fp16   : {tc*1000:.3f}ms  MBU={b16/tc/PEAK:.3f}  tok/s={1/tc:.0f}  (the 74% bar)")
print(f"OURS fp8 GEMV : {to*1000:.3f}ms  MBU={b8/to/PEAK:.3f}  tok/s={1/to:.0f}  SPEEDUP={tc/to:.2f}x  (cutlass=1.53x, target=2x)")
