import torch, time, triton, triton.language as tl
torch.manual_seed(0); DEV='cuda'; PEAK=3.35e12
SH={'q':(4096,4096),'k':(1024,4096),'v':(1024,4096),'o':(4096,4096),
    'gate':(14336,4096),'up':(14336,4096),'down':(4096,14336)}; L=32

# autotuned GEMV: program owns BLOCK_N rows, streams K in BLOCK_K contiguous chunks
@triton.autotune(
  configs=[triton.Config({'BLOCK_N':bn,'BLOCK_K':bk}, num_warps=w, num_stages=s)
           for bn in [16,32,64] for bk in [512,1024,2048] for w in [4,8] for s in [2,3]],
  key=['N','K'])
@triton.jit
def gemv_k(x_ptr, w_ptr, s_ptr, y_ptr, N, K, fp8: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid=tl.program_id(0); rows=pid*BLOCK_N+tl.arange(0,BLOCK_N)
    acc=tl.zeros((BLOCK_N,),dtype=tl.float32)
    for k0 in range(0,K,BLOCK_K):
        ks=k0+tl.arange(0,BLOCK_K)
        xk=tl.load(x_ptr+ks, mask=ks<K, other=0.0).to(tl.float32)
        w=tl.load(w_ptr+rows[:,None]*K+ks[None,:], mask=(rows[:,None]<N)&(ks[None,:]<K), other=0.0)
        acc+=tl.sum(w.to(tl.float32)*xk[None,:],axis=1)
    if fp8:
        sc=tl.load(s_ptr+rows, mask=rows<N, other=0.0).to(tl.float32); acc=acc*sc
    tl.store(y_ptr+rows, acc.to(tl.float16), mask=rows<N)

def gemv(x,w,s,N,K,fp8):
    y=torch.empty(N,device=DEV,dtype=torch.float16)
    gemv_k[(triton.cdiv(N,1),)](x,w,s,y,N,K,fp8, BLOCK_N=32,BLOCK_K=1024) if False else \
    gemv_k[lambda M:(triton.cdiv(N,M['BLOCK_N']),)](x,w,s,y,N,K,fp8)
    return y

def bench(fn,it=50,wm=15):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it

W16={k:torch.randn(o,i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}
W8={};SC={}
for k,(o,i) in SH.items():
    w=W16[k].float(); s=(w.abs().amax(1)/448.0); W8[k]=(w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous(); SC[k]=s.half().contiguous()
x={k:torch.randn(i,device=DEV,dtype=torch.float16) for k,(o,i) in SH.items()}

b16=sum(o*i*2 for o,i in SH.values())*L; b8=sum(o*i+o*4 for o,i in SH.values())*L
# triton fp16 GEMV (diagnostic: can my structure match cuBLAS?)
def run_tri16():
    for _ in range(L):
        for k,(o,i) in SH.items(): gemv(x[k],W16[k],SC[k],o,i,False)
def run_tri8():
    for _ in range(L):
        for k,(o,i) in SH.items(): gemv(x[k],W8[k],SC[k],o,i,True)
def run_cublas16():
    for _ in range(L):
        for k,(o,i) in SH.items(): torch.nn.functional.linear(x[k].unsqueeze(0),W16[k])

# correctness
k='down';o,i=SH[k]; ref=torch.nn.functional.linear(x[k].unsqueeze(0),W16[k]).squeeze(0)
print("fp8 rel_err:",((gemv(x[k],W8[k],SC[k],o,i,True).float()-ref.float()).norm()/ref.float().norm()).item())
tc=bench(run_cublas16); print(f"cuBLAS fp16: {tc*1000:.3f}ms MBU={b16/tc/PEAK:.3f} tok/s={1/tc:.0f}")
tt16=bench(run_tri16);  print(f"Triton fp16: {tt16*1000:.3f}ms MBU={b16/tt16/PEAK:.3f} tok/s={1/tt16:.0f}  (vs cuBLAS {tc/tt16:.2f}x)")
tt8=bench(run_tri8);    print(f"Triton fp8 : {tt8*1000:.3f}ms MBU={b8/tt8/PEAK:.3f} tok/s={1/tt8:.0f}  SPEEDUP_vs_cuBLASfp16={tc/tt8:.2f}x")
