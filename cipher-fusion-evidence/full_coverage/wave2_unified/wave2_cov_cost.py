#!/usr/bin/env python3
"""
WAVE-2 close-out (advisor): MEASURE the two asserted claims of the unified deployable detector
= "recompute-from-INPUT, compare GEMM outputs only" (no separate non-GEMM compares).
(A) PROPAGATION COVERAGE: inject a non-GEMM-output storage flip (h=RMSNorm, act=SiLU) AND a GEMM-output
    flip (q) into the ORIGINAL step; confirm the recompute-from-xin GEMM-output compare CATCHES it
    (Gap 4 proved the recompute-from-cached-h detector is BLIND to non-GEMM — this tests the from-input
    variant actually catches it). Clean recompute is bit-identical => T=0 => 0 FP by construction.
(B) REAL COST: base vs detect (forward + recompute-from-xin + 7 GEMM-output compares/layer), real r,
    decode+prefill; E2E=(r-1)/N at N=45 and N=64, wall-weighted (39% dec / 61% pre, ra-e2e) -> which N <3%.
READ-ONLY, not injected (production .so untouched). fp32 reductions. CUDA-graph for cost; eager for catch.
"""
import json, statistics, random, torch, torch.nn.functional as F
torch.manual_seed(0); dev='cuda'
H,I,KV,L = 4096,14336,1024,32
HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/wave2_unified'
def Wt(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[Wt(H,H) for _ in range(L)]; Wk=[Wt(KV,H) for _ in range(L)]; Wv=[Wt(KV,H) for _ in range(L)]
Wo=[Wt(H,H) for _ in range(L)]; Wg=[Wt(I,H) for _ in range(L)]; Wu=[Wt(I,H) for _ in range(L)]; Wd=[Wt(H,I) for _ in range(L)]
def rms(t): return t*torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+1e-5).half()
def make_rope(M):
    pos=torch.arange(M,device=dev).float(); inv=1.0/(10000**(torch.arange(0,128,2,device=dev).float()/128))
    ang=torch.outer(pos,inv); emb=torch.cat((ang,ang),-1); return torch.cos(emb).half(),torch.sin(emb).half()
def rope(x,cs,sn,nh):
    M=x.shape[0]; x=x.view(M,nh,128); c=cs.view(M,1,128); s=sn.view(M,1,128)
    x1=x[...,:64]; x2=x[...,64:]; return (x*c+torch.cat((-x2,x1),-1)*s).reshape(M,nh*128)

def step(x0, cs, sn, mode, fault=None):
    """mode: base | detect. fault=(layer,name) flips a STORAGE bit so the downstream consumer reads it."""
    x=x0; m=torch.zeros((),device=dev,dtype=torch.float32)
    def flip(y,key):
        if fault is not None and key==fault:
            fl=y.reshape(-1); u=fl.view(torch.int16); u[fl.shape[0]//2]^=(1<<14)
        return y
    def cmp(a,b):
        nonlocal m; m=torch.maximum(m,(a.float()-b.float()).abs().amax())
    for i in range(L):
        xin=x
        h=flip(rms(xin),(i,'h'))                      # non-GEMM output (consumed by q/k/v GEMMs)
        q=flip(F.linear(h,Wq[i]),(i,'q')); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])  # q GEMM output
        qr=rope(q,cs,sn,32); kr=rope(k,cs,sn,8)
        o=F.linear(qr,Wo[i]); x=(xin+o).clamp(-30,30)
        h2=rms(x); gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=flip(F.silu(gt)*up,(i,'act')); dn=F.linear(act,Wd[i]); x=(x+dn).clamp(-30,30)  # non-GEMM output (consumed by down GEMM)
        if mode=='detect':                            # recompute-from-xin (clean), compare GEMM outputs ONLY
            hr=rms(xin); qf=F.linear(hr,Wq[i]); kf=F.linear(hr,Wk[i]); vf=F.linear(hr,Wv[i])
            of=F.linear(rope(qf,cs,sn,32),Wo[i]); xr=(xin+of).clamp(-30,30)
            h2f=rms(xr); gtf=F.linear(h2f,Wg[i]); upf=F.linear(h2f,Wu[i]); dnf=F.linear(F.silu(gtf)*upf,Wd[i])
            cmp(q,qf); cmp(k,kf); cmp(v,vf); cmp(o,of); cmp(gt,gtf); cmp(up,upf); cmp(dn,dnf)
    return x, m

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

N1, N2 = 45, 64
WALL_DEC, WALL_PRE = 0.39, 0.61
out={'N1':N1,'N2':N2}
# ---- (B) COST: real r for recompute-from-xin (GEMM-output compares only) ----
rs={}
for tag,M in [('decode',128),('prefill',512)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half(); cs,sn=make_rope(M)
    tb=med(capture(lambda:step(x0,cs,sn,'base')))
    td=med(capture(lambda:step(x0,cs,sn,'detect')))
    r=td/tb; rs[tag]=r
    out[tag]=dict(M=M,base_ms=tb,detect_ms=td,r=r,e2e_N45_pct=100*(r-1)/N1,e2e_N64_pct=100*(r-1)/N2)
    print(f"[{tag}] base {tb:.3f} detect {td:.3f} r={r:.3f} | E2E (r-1)/N: N45 {100*(r-1)/N1:.2f}% N64 {100*(r-1)/N2:.2f}%",flush=True)
e2e_N45 = WALL_DEC*out['decode']['e2e_N45_pct']+WALL_PRE*out['prefill']['e2e_N45_pct']
e2e_N64 = WALL_DEC*out['decode']['e2e_N64_pct']+WALL_PRE*out['prefill']['e2e_N64_pct']
out['e2e_wallweighted']=dict(N45=e2e_N45, N64=e2e_N64)
print(f"[cost] wall-weighted E2E: N=45 -> {e2e_N45:.2f}%  N=64 -> {e2e_N64:.2f}%",flush=True)

# ---- (A) PROPAGATION COVERAGE (eager, value-exact). Clean recompute bit-identical => T=0. ----
cs,sn=make_rope(128); x0=(torch.randn(128,H,device=dev)*0.1).half()
_,m_clean=step(x0,cs,sn,'detect',fault=None); T=float(m_clean)   # expect exactly 0
rng=random.Random(0); sites=['h','q','act']; cov={s:0 for s in sites}; ntr=20
for s in sites:
    for _ in range(ntr):
        xin=(torch.randn(128,H,device=dev)*0.1).half()
        _,mm=step(xin,cs,sn,'detect',fault=(rng.randrange(L),s))
        if float(mm)>T: cov[s]+=1
out['coverage']=dict(T_clean=T, by_site={s:f"{cov[s]}/{ntr}" for s in sites},
    note='h,act = non-GEMM output storage flips (the class Gap-4 showed the shipped from-cached-h detector is BLIND to); q = GEMM output flip')
print(f"[coverage] T_clean={T} (expect 0) | non-GEMM h {cov['h']}/{ntr}  non-GEMM act {cov['act']}/{ntr}  GEMM q {cov['q']}/{ntr}",flush=True)
json.dump(out,open(f'{HERE}/wave2_cov_cost_result.json','w'),indent=1,default=str)
print("[wave2-close] wrote wave2_cov_cost_result.json",flush=True)
