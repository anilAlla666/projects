#!/usr/bin/env python3
"""
GAP 4 DIAGNOSTIC — is the +71%/+54% the bandwidth FLOOR or the kernel-COUNT wall?
Rule-4 evidence (advisor): PROFILE, don't assert.

(1) sweep #checked-layers 0..32 -> overhead linear in layer count = kernel-count signature.
(2) torch.profiler kernel COUNT for base vs full-check (graph removes launch gaps, not count/duration).
(3) analytical activation-traffic floor per phase (separate-pass re-read of cached non-GEMM I/O).
(4) realizable FUSED floor: a single-pass reduction streaming the dominant cached tensors once
    (gate,up,act [M,14336]) = the irreducible separate-pass cost a fused recompute+compare can hit.
READ-ONLY, not injected. fp32 reductions. Weights random (cost is value-independent).
"""
import sys, json, statistics
import torch, torch.nn.functional as F
torch.manual_seed(0); dev='cuda'
H,I,KV,V,L = 4096,14336,1024,32000,32
HEADS, HD = 32, 128
HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/gap4_nongemm'
PEAK_BW = 2.97e12  # measured H100 read BW @1200 (prior memory); used for analytical floor

def Wt(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[Wt(H,H) for _ in range(L)]; Wk=[Wt(KV,H) for _ in range(L)]; Wv=[Wt(KV,H) for _ in range(L)]
Wo=[Wt(H,H) for _ in range(L)]; Wg=[Wt(I,H) for _ in range(L)]; Wu=[Wt(I,H) for _ in range(L)]; Wd=[Wt(H,I) for _ in range(L)]

def rms(t): return t*torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+1e-5).half()
def make_rope(M):
    pos=torch.arange(M,device=dev).float(); inv=1.0/(10000**(torch.arange(0,HD,2,device=dev).float()/HD))
    ang=torch.outer(pos,inv); emb=torch.cat((ang,ang),dim=-1); return torch.cos(emb).half(), torch.sin(emb).half()
def rope(x, cs, sn, nh):
    M=x.shape[0]; x=x.view(M,nh,HD); c=cs.view(M,1,HD); s=sn.view(M,1,HD)
    x1=x[...,:HD//2]; x2=x[...,HD//2:]; rot=torch.cat((-x2,x1),dim=-1); return (x*c+rot*s).reshape(M,nh*HD)

def step(x0, cs, sn, check_layers=0, mbuf=None):
    """Forward; for the first `check_layers` layers ALSO recompute every non-GEMM op + fp32 max|diff|."""
    x=x0; m=torch.zeros((),device=dev,dtype=torch.float32)
    def cmp(a,b):
        nonlocal m; m=torch.maximum(m,(a-b).abs().amax().float())
    for i in range(L):
        chk = i < check_layers
        h=rms(x);                         _=(cmp(h,rms(x)) if chk else None)
        q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        qr=rope(q,cs,sn,HEADS);           _=(cmp(qr,rope(q,cs,sn,HEADS)) if chk else None)
        kr=rope(k,cs,sn,KV//HD);          _=(cmp(kr,rope(k,cs,sn,KV//HD)) if chk else None)
        o=F.linear(qr,Wo[i]); xr=(x+o).clamp(-30,30); _=(cmp(xr,(x+o).clamp(-30,30)) if chk else None); x=xr
        h2=rms(x);                        _=(cmp(h2,rms(x)) if chk else None)
        gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=F.silu(gt)*up;                _=(cmp(act,F.silu(gt)*up) if chk else None)
        dn=F.linear(act,Wd[i]); xr2=(x+dn).clamp(-30,30); _=(cmp(xr2,(x+dn).clamp(-30,30)) if chk else None); x=xr2
    if mbuf is not None: mbuf.copy_(m)
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
    """profile eager fn, count CUDA kernel launches (graph keeps the same count, just no launch gaps)."""
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    n=0
    for e in prof.events():
        if e.device_type.name=='CUDA' and e.cuda_time_total>0:
            n+=int(getattr(e,'count',1) or 1)
    return n

out={}
for tag,M in [('decode',128),('prefill',512)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half(); cs,sn=make_rope(M)
    mbuf=torch.zeros((),device=dev,dtype=torch.float32)
    # (1) sweep checked-layers
    sweep={}
    gb=capture(lambda:step(x0,cs,sn,0)); tb=med(gb)
    for cl in [0,8,16,24,32]:
        g=capture(lambda cl=cl:step(x0,cs,sn,cl,mbuf)); t=med(g)
        sweep[cl]=dict(ms=t, overhead_pct=100*(t-tb)/tb)
        print(f"[{tag}] check_layers={cl:2d}  {t:.3f}ms  (+{100*(t-tb)/tb:.2f}%)",flush=True)
    # slope per layer (linear fit kernel-count signature)
    xs=[0,8,16,24,32]; ys=[sweep[c]['ms'] for c in xs]
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(a*a for a in xs); sxy=sum(a*b for a,b in zip(xs,ys))
    slope_ms_per_layer=(n*sxy-sx*sy)/(n*sxx-sx*sx)
    # (2) kernel counts (eager)
    kb=count_kernels(lambda:step(x0,cs,sn,0)); kc=count_kernels(lambda:step(x0,cs,sn,L,mbuf))
    extra_k=kc-kb
    # per-extra-kernel realized duration implied by full-check overhead
    full_extra_ms=sweep[32]['ms']-tb
    per_kernel_us=1e3*full_extra_ms/max(1,extra_k)
    # (3) analytical activation-traffic floor (separate-pass re-read of cached non-GEMM I/O, fp16=2B)
    #   per layer reads: rms1(x+h)=2*H ; rms2(x+h2)=2*H ; rope_q(q+qr)=2*H ; rope_k(k+kr)=2*KV ;
    #   silu(gate+up+act)=3*I ; res1,res2 ~ 4*H.  elements*2B*M*L.
    elem_per_layer = (2*H)+(2*H)+(2*H)+(2*KV)+(3*I)+(4*H)
    bytes_total = elem_per_layer*2*M*L
    floor_ms = 1e3*bytes_total/PEAK_BW
    floor_pct = 100*floor_ms/tb
    # (4) realizable FUSED floor: one streaming reduction over the dominant cached tensors (gate,up,act)
    G=[torch.randn(M,I,device=dev).half() for _ in range(L)]
    Uu=[torch.randn(M,I,device=dev).half() for _ in range(L)]
    Ac=[torch.randn(M,I,device=dev).half() for _ in range(L)]
    def fused_pass():
        m=torch.zeros((),device=dev,dtype=torch.float32)
        for i in range(L):
            m=torch.maximum(m,((F.silu(G[i])*Uu[i])-Ac[i]).abs().amax().float())  # 1 fused-ish reduction/layer
        return m
    gf=capture(fused_pass); tf=med(gf)
    fused_pct=100*tf/tb
    out[tag]=dict(M=M, base_ms=tb, full_check_ms=sweep[32]['ms'], full_overhead_pct=sweep[32]['overhead_pct'],
                  sweep=sweep, slope_ms_per_checked_layer=slope_ms_per_layer,
                  kernels_base=kb, kernels_fullcheck=kc, extra_kernels=extra_k, per_kernel_us=per_kernel_us,
                  bw_floor_ms=floor_ms, bw_floor_pct=floor_pct, bytes_total=bytes_total,
                  fused_dominant_ms=tf, fused_dominant_pct=fused_pct)
    print(f"[{tag}] base {tb:.3f}ms | full-check +{sweep[32]['overhead_pct']:.1f}% | "
          f"slope {slope_ms_per_layer:.4f} ms/layer | kernels {kb}->{kc} (+{extra_k}) "
          f"per-kernel {per_kernel_us:.2f}us | BW-floor {floor_ms:.3f}ms={floor_pct:.2f}% | "
          f"fused-dominant {tf:.3f}ms={fused_pct:.2f}%",flush=True)
    del G,Uu,Ac; torch.cuda.empty_cache()

json.dump(out,open(f"{HERE}/gap4_diag_result.json",'w'),indent=1,default=str)
print("[gap4-diag] wrote gap4_diag_result.json",flush=True)
