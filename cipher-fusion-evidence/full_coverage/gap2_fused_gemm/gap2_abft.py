#!/usr/bin/env python3
"""
GAP 2 — per-step GEMM ABFT row-sum checksum to cuBLAS parity.
Y = F.linear(x,W) = x @ W.T,  W:[out,in], Y:[M,out].  Row-sum identity:
  sum_o Y[m,o] = sum_i x[m,i] * (sum_o W[o,i])  =>  chk = Y.sum(1)  ==  ref = x @ s,  s = W.sum(0) (precomp ONCE).
Question: can the per-GEMM output checksum be added at <3%/step over PLAIN cuBLAS via a SEPARATE reduction
(no custom kernel), or does it need a fused epilogue cuBLASLt CANNOT host (Finding 1: BGRAD reduces over K
not out) / a custom CUTLASS GEMM slower than cuBLAS (the substitution gap)?
Rule-4 floors on paper (FLOP 1/N+1/2K ~ 0.01%; bandwidth re-read X+Y). Measure separate-reduction @real shapes.
READ-ONLY, not injected. fp32 reductions. CUDA-graph (deployed-in-graph regime: kernel DURATION+BW, no launch gaps).
"""
import json, statistics, torch, torch.nn.functional as F
torch.manual_seed(0); dev='cuda'
H,I,KV,L = 4096,14336,1024,32
HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/gap2_fused_gemm'
PEAK_BW=2.97e12

def Wt(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[Wt(H,H) for _ in range(L)]; Wk=[Wt(KV,H) for _ in range(L)]; Wv=[Wt(KV,H) for _ in range(L)]
Wo=[Wt(H,H) for _ in range(L)]; Wg=[Wt(I,H) for _ in range(L)]; Wu=[Wt(I,H) for _ in range(L)]; Wd=[Wt(H,I) for _ in range(L)]
# precompute checksum vectors s = W.sum(0) [in], ONCE (amortized over all steps)
def sv(Ws): return [w.float().sum(0) for w in Ws]
Sq,Sk,Sv,So,Sg,Su,Sd = sv(Wq),sv(Wk),sv(Wv),sv(Wo),sv(Wg),sv(Wu),sv(Wd)
def rms(t): return t*torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+1e-5).half()

def lin(x, W, s, check, mbox):
    Y=F.linear(x,W)
    if check:
        ref = x.float() @ s            # matvec [M,K]@[K] -> [M]
        chk = Y.float().sum(1)         # row-sum [M,N] -> [M]
        mbox[0]=torch.maximum(mbox[0],(ref-chk).abs().amax())
    return Y

def block(x0, check_layers=0, mbox=None):
    x=x0
    for i in range(L):
        chk = i < check_layers
        h=rms(x)
        q=lin(h,Wq[i],Sq[i],chk,mbox); k=lin(h,Wk[i],Sk[i],chk,mbox); v=lin(h,Wv[i],Sv[i],chk,mbox)
        o=lin(q,Wo[i],So[i],chk,mbox); x=(x+o).clamp(-30,30)
        h2=rms(x); gt=lin(h2,Wg[i],Sg[i],chk,mbox); up=lin(h2,Wu[i],Su[i],chk,mbox)
        act=F.silu(gt)*up; dn=lin(act,Wd[i],Sd[i],chk,mbox); x=(x+dn).clamp(-30,30)
    return x

def capture(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g
def med(g,it=50,wm=15):
    for _ in range(wm): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();g.replay();e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)
def count_kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof: fn(); torch.cuda.synchronize()
    return sum(int(getattr(e,'count',1) or 1) for e in prof.events() if e.device_type.name=='CUDA' and e.cuda_time_total>0)

out={}
for tag,M in [('decode',128),('prefill',512)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half(); mbox=[torch.zeros((),device=dev,dtype=torch.float32)]
    gb=capture(lambda:block(x0,0)); tb=med(gb)
    sweep={}
    for cl in [0,8,16,24,32]:
        g=capture(lambda cl=cl:block(x0,cl,mbox)); t=med(g); sweep[cl]=dict(ms=t,overhead_pct=100*(t-tb)/tb)
        print(f"[{tag}] check_layers={cl:2d}  {t:.3f}ms  (+{100*(t-tb)/tb:.2f}%)",flush=True)
    kb=count_kernels(lambda:block(x0,0)); kc=count_kernels(lambda:block(x0,L,mbox)); extra_k=kc-kb
    full_extra=sweep[32]['ms']-tb; per_kernel_us=1e3*full_extra/max(1,extra_k)
    # FLOP floor: per GEMM checksum = matvec 2MK + rowsum MN vs GEMM 2MNK -> 1/N + 1/2K
    gemms=[('q',H,H),('k',KV,H),('v',KV,H),('o',H,H),('gate',I,H),('up',I,H),('down',H,I)]  # (name,N=out,K=in)
    flop_gemm=sum(2*M*N*K for _,N,K in gemms)*L
    flop_chk=sum(2*M*K + M*N for _,N,K in gemms)*L
    flop_floor_pct=100*flop_chk/flop_gemm
    # bandwidth floor: checksum re-reads X[M,K] + Y[M,N] per GEMM (fp16 2B) vs GEMM weight read N*K*2
    bytes_chk=sum((M*K + M*N) for _,N,K in gemms)*2*L
    bytes_gemm_w=sum(N*K for _,N,K in gemms)*2*L
    bw_floor_ms=1e3*bytes_chk/PEAK_BW; bw_floor_pct=100*bw_floor_ms/tb
    out[tag]=dict(M=M,base_ms=tb,full_check_ms=sweep[32]['ms'],full_overhead_pct=sweep[32]['overhead_pct'],
                  sweep=sweep,kernels_base=kb,kernels_fullcheck=kc,extra_kernels=extra_k,per_kernel_us=per_kernel_us,
                  flop_floor_pct=flop_floor_pct,bw_floor_ms=bw_floor_ms,bw_floor_pct=bw_floor_pct,
                  bytes_chk=bytes_chk,bytes_gemm_weights=bytes_gemm_w)
    print(f"[{tag}] base {tb:.3f}ms | +ABFT(sep-reduction over cuBLAS) +{sweep[32]['overhead_pct']:.1f}% | "
          f"kernels {kb}->{kc}(+{extra_k}) per-kernel {per_kernel_us:.2f}us | FLOP-floor {flop_floor_pct:.3f}% | "
          f"BW-floor {bw_floor_ms:.3f}ms={bw_floor_pct:.2f}%",flush=True)

# ---- coverage confirmation: inject a flip in a GEMM output, confirm caught (Step-A class) ----
M=128; x0=(torch.randn(M,H,device=dev)*0.1).half()
mbox=[torch.zeros((),device=dev,dtype=torch.float32)]
# clean T over output checksum
def one_block_check(xin, fault=None):
    mb=[torch.zeros((),device=dev,dtype=torch.float32)]; x=xin
    for i in range(L):
        h=rms(x)
        q=lin(h,Wq[i],Sq[i],True,mb); k=lin(h,Wk[i],Sk[i],True,mb); v=lin(h,Wv[i],Sv[i],True,mb)
        o=lin(q,Wo[i],So[i],True,mb)
        if fault==(i,'o'):
            fl=o.view(-1); u=fl.view(torch.int16); u[fl.shape[0]//2]^=(1<<14)
            # separate reduction over the (now corrupted) cuBLAS output Y detects the flip
            ref=q.float()@So[i]; chk=o.float().sum(1); mb[0]=torch.maximum(mb[0],(ref-chk).abs().amax())
        x=(x+o).clamp(-30,30)
        h2=rms(x); gt=lin(h2,Wg[i],Sg[i],True,mb); up=lin(h2,Wu[i],Su[i],True,mb)
        act=F.silu(gt)*up; dn=lin(act,Wd[i],Sd[i],True,mb); x=(x+dn).clamp(-30,30)
    return float(mb[0])
Tclean=max(one_block_check((torch.randn(M,H,device=dev)*0.1).half()) for _ in range(8))
caught=0; ntr=30
import random as _r; rng=_r.Random(0)
for _ in range(ntr):
    sig=one_block_check((torch.randn(M,H,device=dev)*0.1).half(), fault=(rng.randrange(L),'o'))
    if sig>Tclean*8: caught+=1
out['coverage']=dict(T_clean=Tclean,caught=caught,n=ntr)
print(f"[gap2 coverage] T_clean={Tclean:.3e} | {caught}/{ntr} GEMM-output flips caught (T=8x)",flush=True)
json.dump(out,open(f"{HERE}/gap2_abft_result.json",'w'),indent=1,default=str)
print("[gap2] wrote gap2_abft_result.json",flush=True)
