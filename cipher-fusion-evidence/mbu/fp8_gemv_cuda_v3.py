import time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12
CUDA = r'''
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
// one WARP per output row: max occupancy, warp-shuffle reduce (no shared, no syncthreads)
__global__ void fp8_gemv_v3(const __nv_fp8_e4m3* __restrict__ W, const __half* __restrict__ x,
                            const __half* __restrict__ scale, __half* __restrict__ y, int N, int K){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int lane=threadIdx.x&31;
    if(warp>=N) return;
    const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)warp*K);
    const __half2* x2=reinterpret_cast<const __half2*>(x);
    int Kvec=K>>4;                                   // 16 fp8 / uint4
    float acc=0.f;
    for(int c=lane;c<Kvec;c+=32){                    // 32 lanes stride -> coalesced 512B/iter
        uint4 wp=wrow[c];
        const __nv_fp8x2_e4m3* w2=reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);
        int h2=c<<3;
        #pragma unroll
        for(int j=0;j<8;j++){
            __half2 wv=(__half2)w2[j];
            __half2 p=__hmul2(wv, x2[h2+j]);
            acc+=__low2float(p)+__high2float(p);
        }
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[warp]=__float2half(acc*__half2float(scale[warp]));
}
torch::Tensor fp8_gemv3(torch::Tensor W, torch::Tensor x, torch::Tensor scale){
    int N=W.size(0),K=W.size(1);
    auto y=torch::empty({N},torch::dtype(torch::kHalf).device(W.device()));
    const int NT=256, warps=NT>>5;                   // 8 warps/block
    int blocks=(N+warps-1)/warps;
    fp8_gemv_v3<<<blocks,NT>>>(reinterpret_cast<const __nv_fp8_e4m3*>(W.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()),reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),N,K);
    return y;
}
'''
mod=load_inline(name="fp8gemv3",cpp_sources="torch::Tensor fp8_gemv3(torch::Tensor,torch::Tensor,torch::Tensor);",
    cuda_sources=CUDA,functions=["fp8_gemv3"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_90a"],verbose=False)
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}; W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float();s=(w.abs().amax(1)/448.0).clamp(min=1e-6);W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous();SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
k='down';o,i=SH[k];ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
print("rel_err:",((mod.fp8_gemv3(W8[k],x[k],SC[k]).float()-ref.float()).norm()/ref.float().norm()).item())
def bench(fn,it=80,wm=25):
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
        for k,(o,i) in SH.items():mod.fp8_gemv3(W8[k],x[k],SC[k])
tc=bench(rc);to=bench(ro)
print(f"cuBLAS fp16 : MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
print(f"OURS v3 fp8 : MBU={b8/to/PEAK:.3f} tok/s={1/to:.0f}  SPEEDUP={tc/to:.2f}x  (v1=1.47,cutlass=1.53,target 2x)")
