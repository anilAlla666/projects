#!/usr/bin/env python3
"""
WAVE 2 — UNIFIED periodic-recompute detector cost (full coverage in ONE window).
Per-step coverage walls (Gap 1), so the deployable unified detector is PERIODIC (every N steps):
that checked step does the gap-free recompute over cuBLAS = re-run ALL linear GEMMs (2nd cuBLAS pass,
compare vs cached: residual 0) [ra-e2e] + recompute non-GEMM ops (compare) [Gap 4] + attn ABFT
checksums [Gap 3, added analytically: synthetic fwd here uses identity attention]. Catches PERSISTENT
faults within N across linear-GEMM + non-GEMM + attention.
Measures base / gemm-recompute / +nongemm-recompute (decode M=128, prefill M=512) to get the MARGINAL
cost of folding non-GEMM into the recompute window. Absolute r is an UPPER bound (synthetic fwd is
GEMM-heavy, no real attention); E2E is anchored to ra-e2e's real-trace r (1.49 dec / 1.33 pre, +2.96%).
READ-ONLY, not injected. fp32 reductions. CUDA-graph.
"""
import json, statistics, torch, torch.nn.functional as F
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

def step(x0, cs, sn, mode):
    x=x0; m=torch.zeros((),device=dev,dtype=torch.float32)
    def cmp(a,b):
        nonlocal m; m=torch.maximum(m,(a-b).abs().amax().float())
    for i in range(L):
        h=rms(x)
        q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        qr=rope(q,cs,sn,32); kr=rope(k,cs,sn,8)
        o=F.linear(qr,Wo[i]); x=(x+o).clamp(-30,30)
        h2=rms(x); gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=F.silu(gt)*up; dn=F.linear(act,Wd[i]); x=(x+dn).clamp(-30,30)
        if mode in ('gemm','unified'):                       # re-run all 7 GEMMs (2nd cuBLAS pass), compare (residual 0)
            cmp(q,F.linear(h,Wq[i])); cmp(k,F.linear(h,Wk[i])); cmp(v,F.linear(h,Wv[i]))
            cmp(o,F.linear(qr,Wo[i])); cmp(gt,F.linear(h2,Wg[i])); cmp(up,F.linear(h2,Wu[i])); cmp(dn,F.linear(act,Wd[i]))
        if mode=='unified':                                  # + recompute non-GEMM ops, compare
            cmp(h,rms(x0 if i==0 else x)); cmp(qr,rope(q,cs,sn,32)); cmp(kr,rope(k,cs,sn,8)); cmp(act,F.silu(gt)*up)
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

N=45  # ra-e2e shipped period
RA = dict(r_decode=1.49, r_prefill=1.33, e2e_pct=2.96, wall_decode=0.39, wall_prefill=0.61)  # ra-e2e anchors
out={'N':N, 'ra_e2e_anchor':RA}
for tag,M in [('decode',128),('prefill',512)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half(); cs,sn=make_rope(M)
    tb=med(capture(lambda:step(x0,cs,sn,'base')))
    tg=med(capture(lambda:step(x0,cs,sn,'gemm')))
    tu=med(capture(lambda:step(x0,cs,sn,'unified')))
    out[tag]=dict(M=M, base_ms=tb, gemm_recompute_ms=tg, unified_ms=tu,
                  r_gemm=tg/tb, r_unified=tu/tb,
                  nongemm_marginal_pct_of_base=100*(tu-tg)/tb,
                  e2e_gemm_pct=100*(tg/tb-1)/N, e2e_unified_pct=100*(tu/tb-1)/N)
    print(f"[{tag}] base {tb:.3f} | gemm-recompute r={tg/tb:.3f} | unified r={tu/tb:.3f} | "
          f"nonGEMM marginal +{100*(tu-tg)/tb:.2f}% of base | E2E@N{N}: gemm {100*(tg/tb-1)/N:.3f}% unified {100*(tu/tb-1)/N:.3f}%",flush=True)

# Compose E2E anchored to ra-e2e real-trace r (synthetic r is an upper bound; the MARGINAL non-GEMM/attn
# additions are what THIS run contributes). attn-ABFT (Gap 3) FLOP floor 0.488% of step / N (periodic).
attn_e2e = 0.488/N
# marginal non-GEMM E2E using THIS run's nonGEMM-marginal/N, wall-weighted
nm_dec = out['decode']['nongemm_marginal_pct_of_base']/N
nm_pre = out['prefill']['nongemm_marginal_pct_of_base']/N
nm_e2e = RA['wall_decode']*nm_dec + RA['wall_prefill']*nm_pre
unified_e2e = RA['e2e_pct'] + nm_e2e + attn_e2e
out['composed_e2e'] = dict(ra_e2e_gemm_pct=RA['e2e_pct'], nongemm_add_pct=nm_e2e, attn_abft_add_pct=attn_e2e,
                           unified_e2e_pct=unified_e2e,
                           note='ra-e2e GEMM-recompute (real trace) + this-run non-GEMM marginal/N + Gap3 attn-ABFT FLOP/N')
print(f"[wave2] COMPOSED unified E2E @N={N}: {RA['e2e_pct']:.2f}% (GEMM, ra-e2e) + {nm_e2e:.3f}% (nonGEMM) + {attn_e2e:.3f}% (attn) = {unified_e2e:.3f}%",flush=True)
json.dump(out,open(f'{HERE}/wave2_result.json','w'),indent=1,default=str)
print("[wave2] wrote wave2_result.json",flush=True)
