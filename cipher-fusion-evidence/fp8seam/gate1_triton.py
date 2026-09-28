# GATE 1: can a fusable Triton FP8 GEMM (the ownable kernel) match cutlass_scaled_mm? (decode M=1 is
# the make-or-break.) Standard tiled FP8 matmul, autotuned.
import torch, vllm._custom_ops, triton, triton.language as tl
dev="cuda"; torch.manual_seed(0)
_cut=torch.ops._C.cutlass_scaled_mm

@triton.autotune(
  configs=[triton.Config({'BM':bm,'BN':bn,'BK':bk},num_stages=s,num_warps=w)
    for bm in [16,32] for bn in [64,128,256] for bk in [64,128,256] for s in [3,4,5] for w in [4,8]],
  key=['M','N','K'])
@triton.jit
def _fp8gemm(A,B,C,sa,sb,M,N,K,sam,sak,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    offm=(pid_m*BM+tl.arange(0,BM))%M; offn=(pid_n*BN+tl.arange(0,BN))%N; offk=tl.arange(0,BK)
    ap=A+offm[:,None]*sam+offk[None,:]*sak; bp=B+offk[:,None]*sbk+offn[None,:]*sbn
    acc=tl.zeros((BM,BN),tl.float32)
    for k in range(0,tl.cdiv(K,BK)):
        a=tl.load(ap,mask=offk[None,:]<K-k*BK,other=0.0); b=tl.load(bp,mask=offk[:,None]<K-k*BK,other=0.0)
        acc+=tl.dot(a,b)
        ap+=BK*sak; bp+=BK*sbk
    acc=acc*sa*sb
    cm=pid_m*BM+tl.arange(0,BM); cn=pid_n*BN+tl.arange(0,BN)
    cp=C+cm[:,None]*scm+cn[None,:]*scn
    tl.store(cp,acc.to(tl.bfloat16),mask=(cm[:,None]<M)&(cn[None,:]<N))

def trigemm(a,b,sa,sb):
    M,K=a.shape; N=b.shape[1]; c=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=lambda meta:(triton.cdiv(M,meta['BM']),triton.cdiv(N,meta['BN']))
    _fp8gemm[grid](a,b,c,sa.item(),sb.item(),M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1))
    return c
def mk(M,K,N):
    a=(torch.randn(M,K,device=dev)*0.1).to(torch.float8_e4m3fn)
    b=(torch.randn(N,K,device=dev)*0.1).to(torch.float8_e4m3fn).t()   # [K,N] col-major (cutlass-accepted)
    return a,b,torch.tensor(0.7,device=dev),torch.tensor(1.3,device=dev)
def tcut(a,b,sa,sb):
    o=torch.empty((a.shape[0],b.shape[1]),dtype=torch.bfloat16,device=dev); _cut(o,a,b,sa,sb,None); return o
def ev(fn,a,b,sa,sb,it=1000):
    for _ in range(30): fn(a,b,sa,sb)
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn(a,b,sa,sb)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it

if __name__=="__main__":
    import json
    for nm,M,K,N in [("decode_bigN",1,4096,14336),("decode_down",1,14336,4096),("decode_qkv",1,4096,6144),("prefill_gateup",512,4096,28672)]:
        a,b,sa,sb=mk(M,K,N); oc=tcut(a,b,sa,sb)
        try:
            ot=trigemm(a,b,sa,sb); rel=((oc.float()-ot.float()).abs().max()/(oc.abs().max()+1e-6)).item()
            tc=ev(tcut,a,b,sa,sb); tt=ev(trigemm,a,b,sa,sb)
            print("TRITON",json.dumps({nm:f"M{M}K{K}N{N}","cutlass_ms":round(tc,5),"triton_ms":round(tt,5),"gap%":round((tt/tc-1)*100,1),"rel":round(rel,4)}))
        except Exception as ex:
            import traceback; print("TRITON",json.dumps({nm:"FAIL "+repr(ex)[:80]}))
