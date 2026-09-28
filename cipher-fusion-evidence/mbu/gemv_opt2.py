import torch, time, triton, triton.language as tl
torch.manual_seed(0); DEV='cuda'; PEAK=3.35e12
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),
    'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32
# fp8 GEMV: native fp8->fp16 convert, multiple rows/program, large coalesced K-stream
@triton.autotune(configs=[triton.Config({'BLOCK_N':bn,'BLOCK_K':bk},num_warps=w,num_stages=s)
    for bn in [8,16,32,64] for bk in [1024,2048,4096] for w in [4,8] for s in [3,4]], key=['N','K'])
@triton.jit
def gemv8(x_ptr,w_ptr,s_ptr,y_ptr,N,K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid=tl.program_id(0); rows=pid*BLOCK_N+tl.arange(0,BLOCK_N)
    acc=tl.zeros((BLOCK_N,),dtype=tl.float32)
    for k0 in range(0,K,BLOCK_K):
        ks=k0+tl.arange(0,BLOCK_K)
        xk=tl.load(x_ptr+ks,mask=ks<K,other=0.0)                       # fp16
        w=tl.load(w_ptr+rows[:,None]*K+ks[None,:],mask=(rows[:,None]<N)&(ks[None,:]<K),other=0.0)  # fp8
        acc+=tl.sum(w.to(tl.float16)*xk[None,:],axis=1).to(tl.float32)  # native fp8->fp16 convert
    sc=tl.load(s_ptr+rows,mask=rows<N,other=0.0).to(tl.float32)
    tl.store(y_ptr+rows,(acc*sc).to(tl.float16),mask=rows<N)
def gemv(x,w,s,N,K):
    y=torch.empty(N,device=DEV,dtype=torch.float16)
    gemv8[lambda M:(triton.cdiv(N,M['BLOCK_N']),)](x,w,s,y,N,K); return y
def bench(fn,it=50,wm=15):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it
W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float();s=(w.abs().amax(1)/448.0);W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous();SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
b16=sum(o*i*2 for o,i in SH.values())*L; b8=sum(o*i+o*4 for o,i in SH.values())*L
k='down';o,i=SH[k];ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
print("rel_err:",((gemv(x[k],W8[k],SC[k],o,i).float()-ref.float()).norm()/ref.float().norm()).item())
def r8():
    for _ in range(L):
        for k,(o,i) in SH.items(): gemv(x[k],W8[k],SC[k],o,i)
def rc():
    for _ in range(L):
        for k,(o,i) in SH.items(): torch.nn.functional.linear(x[k].unsqueeze(0),W16[k])
tc=bench(rc); t8=bench(r8)
print(f"cuBLAS fp16: MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
print(f"Triton fp8 (native conv): MBU={b8/t8/PEAK:.3f} tok/s={1/t8:.0f}  speedup_vs_fp16={tc/t8:.2f}x  (cutlass fp8 ref=1.53x)")
