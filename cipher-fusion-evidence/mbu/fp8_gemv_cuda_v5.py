import time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12
CUDA = r'''
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
// block-per-row + COALESCED ILP: thread reads chunks tid, tid+NT, tid+2NT, tid+3NT (stays coalesced), 4 loads in flight
template<int NT,int ILP>
__global__ void fp8_gemv_v5(const __nv_fp8_e4m3* __restrict__ W, const __half* __restrict__ x,
                            const __half* __restrict__ scale, __half* __restrict__ y, int N, int K){
    int row=blockIdx.x; if(row>=N) return;
    const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)row*K);
    const __half2* x2=reinterpret_cast<const __half2*>(x);
    int Kvec=K>>4; int tid=threadIdx.x; int stride=NT*ILP;
    float acc=0.f;
    for(int b=tid; b<Kvec; b+=stride){
        uint4 wp[ILP];
        #pragma unroll
        for(int u=0;u<ILP;u++){ int c=b+u*NT; wp[u]= c<Kvec? wrow[c] : make_uint4(0,0,0,0); }  // coalesced, 4 in flight
        #pragma unroll
        for(int u=0;u<ILP;u++){
            int c=b+u*NT; if(c>=Kvec) continue;
            const __nv_fp8x2_e4m3* w2=reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp[u]);
            int h2=c<<3;
            #pragma unroll
            for(int j=0;j<8;j++){ __half2 p=__hmul2((__half2)w2[j], x2[h2+j]); acc+=__low2float(p)+__high2float(p); }
        }
    }
    __shared__ float red[NT]; red[tid]=acc; __syncthreads();
    for(int s=NT>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if(tid==0) y[row]=__float2half(red[0]*__half2float(scale[row]));
}
torch::Tensor fp8_gemv5(torch::Tensor W, torch::Tensor x, torch::Tensor scale){
    int N=W.size(0),K=W.size(1);
    auto y=torch::empty({N},torch::dtype(torch::kHalf).device(W.device()));
    fp8_gemv_v5<256,4><<<N,256>>>(reinterpret_cast<const __nv_fp8_e4m3*>(W.data_ptr()),
        reinterpret_cast<const __half*>(x.data_ptr()),reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<__half*>(y.data_ptr()),N,K);
    return y;
}
'''
mod=load_inline(name="fp8gemv5",cpp_sources="torch::Tensor fp8_gemv5(torch::Tensor,torch::Tensor,torch::Tensor);",
    cuda_sources=CUDA,functions=["fp8_gemv5"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_90a"],verbose=False)
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}; W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float();s=(w.abs().amax(1)/448.0).clamp(min=1e-6);W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous();SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
k='down';o,i=SH[k];ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
print("rel_err:",((mod.fp8_gemv5(W8[k],x[k],SC[k]).float()-ref.float()).norm()/ref.float().norm()).item())
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
        for k,(o,i) in SH.items():mod.fp8_gemv5(W8[k],x[k],SC[k])
tc=bench(rc);to=bench(ro)
print(f"cuBLAS fp16 : MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
print(f"OURS v5 fp8 : MBU={b8/to/PEAK:.3f} tok/s={1/to:.0f}  SPEEDUP={tc/to:.2f}x  (best v1=1.47, cutlass=1.53, 2x target)")
