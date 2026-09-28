import time, torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0); DEV="cuda"; PEAK=3.35e12; GROUP=128
CUDA = r'''
#include <cuda_fp16.h>
#include <torch/extension.h>
template<int NT,int GROUP>
__global__ void int4_gemv(const uint8_t* __restrict__ W, const __half* __restrict__ x,
                          const __half* __restrict__ scales, __half* __restrict__ y, int N, int K){
    int row=blockIdx.x; if(row>=N) return;
    const uint4* wrow=reinterpret_cast<const uint4*>(W+(size_t)row*(K>>1));
    const __half* srow=scales+(size_t)row*(K/GROUP);
    const __half2* x2=reinterpret_cast<const __half2*>(x);
    int Bvec=K>>5; int tid=threadIdx.x; float acc=0.f;
    for(int c=tid;c<Bvec;c+=NT){
        uint4 wp=wrow[c]; int k0=c<<5;
        float sc=__half2float(srow[k0/GROUP]);
        const uint32_t* w32=reinterpret_cast<const uint32_t*>(&wp);   // 4 uint32 = 32 nibbles
        const __half2* xh2=x2+(k0>>1);                                // x[k0..k0+31] as 16 half2
        float part=0.f;
        #pragma unroll
        for(int w=0;w<4;w++){
            uint32_t v=w32[w];                                        // 8 nibbles
            #pragma unroll
            for(int n=0;n<8;n++){
                int q=((v>>(4*n))&0xF)-8;                             // signed int4 (offset-binary decode: q = nibble-8)
                __half2 xv=xh2[w*4 + (n>>1)];                         // the half2 holding x[k0+8w+n]
                float xf = (n&1)? __high2float(xv) : __low2float(xv);
                part += float(q)*xf;
            }
        }
        acc += part*sc;
    }
    __shared__ float red[NT]; red[tid]=acc; __syncthreads();
    for(int s=NT>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if(tid==0) y[row]=__float2half(red[0]);
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
mod=load_inline(name="int4gemv3",cpp_sources="torch::Tensor int4_gemv_f(torch::Tensor,torch::Tensor,torch::Tensor);",
    cuda_sources=CUDA,functions=["int4_gemv_f"],extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_90a"],verbose=False)
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
def quant4(w16):
    o,i=w16.shape; w=w16.float().view(o,i//GROUP,GROUP)
    s=(w.abs().amax(-1,keepdim=True)/7.0).clamp(min=1e-6)
    q=(w/s).round().clamp(-8,7).to(torch.int32).view(o,i)
    qn=(q+8).to(torch.uint8)
    packed=(qn[:,0::2] | (qn[:,1::2]<<4)).contiguous()
    return packed, s.squeeze(-1).half().contiguous(), q
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}; W4={};SC={};Q={}
for k,(o,i) in SH.items(): W4[k],SC[k],Q[k]=quant4(W16[k])
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
k='down';o,i=SH[k]
deq=(Q[k].float().view(o,i//GROUP,GROUP)*SC[k].float().unsqueeze(-1)).view(o,i)
ref=(x[k].float()@deq.t())
print("rel_err:", ((mod.int4_gemv_f(W4[k],x[k],SC[k]).float()-ref).norm()/ref.norm()).item())
def bench(fn,it=80,wm=25):
    for _ in range(wm):fn()
    torch.cuda.synchronize();t0=time.perf_counter()
    for _ in range(it):fn()
    torch.cuda.synchronize();return (time.perf_counter()-t0)/it
b16=sum(o*i*2 for o,i in SH.values())*L; b4=sum(o*i//2+(o*i//GROUP)*2 for o,i in SH.values())*L
def rc():
    for _ in range(L):
        for k,(o,i) in SH.items():torch.nn.functional.linear(x[k].unsqueeze(0),W16[k])
def ro():
    for _ in range(L):
        for k,(o,i) in SH.items():mod.int4_gemv_f(W4[k],x[k],SC[k])
tc=bench(rc);to=bench(ro)
print(f"cuBLAS fp16   : MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
print(f"OURS INT4 GEMV: MBU={b4/to/PEAK:.3f} tok/s={1/to:.0f}  SPEEDUP={tc/to:.2f}x  (marlin27%=1.35x, fp8=1.47x, 2x target)")
