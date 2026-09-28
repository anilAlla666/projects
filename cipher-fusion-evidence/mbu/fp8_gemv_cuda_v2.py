import os, time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12
CUDA = r'''
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
template<int NT>
__global__ void fp8_gemv_v2(const __nv_fp8_e4m3* __restrict__ W, const __half* __restrict__ x,
                            const __half* __restrict__ scale, __half* __restrict__ y, int N, int K){
    extern __shared__ __half xs[];                          // x cached in shared (loaded once/block)
    int tid=threadIdx.x;
    for(int j=tid;j<K;j+=NT) xs[j]=x[j];
    __syncthreads();
    const __half2* xs2=reinterpret_cast<const __half2*>(xs);
    int row=blockIdx.x; if(row>=N) return;
    const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)row*K);
    int Kvec=K>>4;                                          // 16 fp8 / chunk
    float2 acc=make_float2(0.f,0.f);
    for(int c=tid;c<Kvec;c+=NT){
        uint4 wp=wrow[c];
        const __nv_fp8x2_e4m3* w2=reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);  // 8 fp8x2
        int h2=(c<<3);                                      // half2 index = c*8
        #pragma unroll
        for(int j=0;j<8;j++){
            __half2 wv=(__half2)w2[j];                      // vectorized hw convert fp8x2->half2
            __half2 p=__hmul2(wv, xs2[h2+j]);
            acc.x+=__low2float(p); acc.y+=__high2float(p);
        }
    }
    float a=acc.x+acc.y;
    __shared__ float red[NT];
    red[tid]=a; __syncthreads();
    for(int s=NT>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if(tid==0) y[row]=__float2half(red[0]*__half2float(scale[row]));
}
torch::Tensor fp8_gemv2(torch::Tensor W, torch::Tensor x, torch::Tensor scale){
    int N=W.size(0),K=W.size(1);
    auto y=torch::empty({N},torch::dtype(torch::kHalf).device(W.device()));
    const int NT=256; size_t shmem=K*sizeof(__half);
    auto k=fp8_gemv_v2<NT>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    k<<<N,NT,shmem>>>(reinterpret_cast<const __nv_fp8_e4m3*>(W.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()),reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),N,K);
    return y;
}
'''
mod=load_inline(name="fp8gemv2",cpp_sources="torch::Tensor fp8_gemv2(torch::Tensor,torch::Tensor,torch::Tensor);",
    cuda_sources=CUDA,functions=["fp8_gemv2"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_90a"],verbose=False)
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}; W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float();s=(w.abs().amax(1)/448.0).clamp(min=1e-6);W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous();SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
k='down';o,i=SH[k];ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
print("rel_err:",((mod.fp8_gemv2(W8[k],x[k],SC[k]).float()-ref.float()).norm()/ref.float().norm()).item())
def bench(fn,it=60,wm=20):
    for _ in range(wm):fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it):fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it
b16=sum(o*i*2 for o,i in SH.values())*L;b8=sum(o*i+o*4 for o,i in SH.values())*L
def rc():
    for _ in range(L):
        for k,(o,i) in SH.items():torch.nn.functional.linear(x[k].unsqueeze(0),W16[k])
def ro():
    for _ in range(L):
        for k,(o,i) in SH.items():mod.fp8_gemv2(W8[k],x[k],SC[k])
tc=bench(rc);to=bench(ro)
print(f"cuBLAS fp16   : MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
print(f"OURS v2 fp8   : MBU={b8/to/PEAK:.3f} tok/s={1/to:.0f}  SPEEDUP={tc/to:.2f}x  (v1=1.47x, cutlass=1.53x, target 2x)")
