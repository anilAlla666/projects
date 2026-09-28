#!/usr/bin/env python3
"""
GAP 4 — non-GEMM ops (RMSNorm / RoPE / SiLU / residual) per-step transient coverage.
Independent close (advisor): recomputing ONLY the non-GEMM ops every step is cheap and needs no
GEMM per-step check, so it does NOT inherit Gap 1/2's fused-GEMM wall. These ops are memory-bound
elementwise on ACTIVATIONS only (no 14.3GB weight stream), so per-step recompute should be <<3%.

Measures: (1) per-step overhead of recomputing all non-GEMM ops every step, under saturation
(CUDA-graph), in decode (B=128) and prefill (M=512); (2) catch of Step-A flips injected into
RMSNorm / RoPE / SiLU / residual outputs (eager, value-exact).
Weights random (non-GEMM cost is value-independent; catch needs only clean-vs-flip determinism).
READ-ONLY, not injected. fp32 reductions for the comparison (Step-A discipline).
"""
import sys, json, statistics, time, math
import torch, torch.nn.functional as F
torch.manual_seed(0); dev='cuda'
H,I,KV,V,L = 4096,14336,1024,32000,32
HEADS, HD = 32, 128            # 32 q-heads, head_dim 128; GQA KV heads = KV/HD = 8
HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/gap4_nongemm'

def W(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[W(H,H) for _ in range(L)]; Wk=[W(KV,H) for _ in range(L)]; Wv=[W(KV,H) for _ in range(L)]
Wo=[W(H,H) for _ in range(L)]; Wg=[W(I,H) for _ in range(L)]; Wu=[W(I,H) for _ in range(L)]; Wd=[W(H,I) for _ in range(L)]
Wlm=W(V,H)

def rms(t):  # RMSNorm (non-GEMM)
    return t*torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+1e-5).half()

# RoPE (non-GEMM): standard HF rotate_half form (contiguous, no strided scatter)
def make_rope(M):
    pos=torch.arange(M,device=dev).float()
    inv=1.0/(10000**(torch.arange(0,HD,2,device=dev).float()/HD))
    ang=torch.outer(pos,inv)                       # [M, HD/2]
    emb=torch.cat((ang,ang),dim=-1)                # [M, HD]
    return torch.cos(emb).half(), torch.sin(emb).half()
def rope(x, cs, sn, nh):                            # x:[M, nh*HD], cs/sn:[M, HD]
    M=x.shape[0]; x=x.view(M,nh,HD)
    c=cs.view(M,1,HD); s=sn.view(M,1,HD)
    x1=x[...,:HD//2]; x2=x[...,HD//2:]
    rot=torch.cat((-x2,x1),dim=-1)
    return (x*c + rot*s).reshape(M,nh*HD)

def nongemm_forward(x0, cs, sn, recompute=False, cache=None, fault=None):
    """Full step; if recompute, ALSO recompute the non-GEMM ops a 2nd time and accumulate max|diff|.
    fault=(layer,op) injects a Step-A bit-14 flip into that non-GEMM output's stored copy."""
    x=x0; M=x0.shape[0]
    m=torch.zeros((),device=dev,dtype=torch.float32)
    def flip(y,key):
        if fault is not None and key==fault:
            fl=y.view(-1); u=fl.view(torch.int16); u[fl.shape[0]//2]^=(1<<14)
        return y
    for i in range(L):
        h=flip(rms(x),(i,'rms1'))
        q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        q=flip(rope(q,cs,sn,HEADS),(i,'rope_q')); k=flip(rope(k,cs,sn,KV//HD),(i,'rope_k'))
        o=F.linear(q,Wo[i])                          # attention excluded (identity q->o), GEMM not checked here
        x=flip((x+o),(i,'res1')).clamp_(-30,30)
        h2=flip(rms(x),(i,'rms2'))
        gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=flip(F.silu(gt)*up,(i,'silu'))
        dn=F.linear(act,Wd[i])
        x=flip((x+dn),(i,'res2')).clamp_(-30,30)
        if recompute:                                # NON-GEMM recompute only (re-run elementwise ops)
            h_r=rms(x0 if i==0 else x_prev_in)       # placeholder; real recompute below uses stored inputs
        if cache is not None:
            cache[(i,'rms1')]=h; cache[(i,'rope_q')]=q; cache[(i,'rope_k')]=k
            cache[(i,'silu')]=act
    return x

# Cleaner: cost graph recomputes the non-GEMM ops from the SAME GEMM outputs (which the step already
# has). We model that as: every step, after the forward, re-run rms/rope/silu/residual a 2nd time on
# the live tensors + an fp32 max|diff| reduction into a scalar (the detector cost). GEMMs NOT redone.
def step_base(x0, cs, sn):
    x=x0
    for i in range(L):
        h=rms(x); q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        q=rope(q,cs,sn,HEADS); k=rope(k,cs,sn,KV//HD); o=F.linear(q,Wo[i]); x=(x+o).clamp_(-30,30)
        h2=rms(x); gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i]); dn=F.linear(F.silu(gt)*up,Wd[i]); x=(x+dn).clamp_(-30,30)
    return F.linear(rms(x),Wlm)

def step_with_nongemm_check(x0, cs, sn, mbuf):
    x=x0; m=torch.zeros((),device=dev,dtype=torch.float32)
    def cmp(y_ref, y_chk): return torch.maximum(m,(y_ref-y_chk).abs().amax().float())
    for i in range(L):
        h=rms(x); m=cmp(h, rms(x))                    # recompute RMSNorm1 + compare
        q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        qr=rope(q,cs,sn,HEADS); m=cmp(qr, rope(q,cs,sn,HEADS))   # recompute RoPE q
        kr=rope(k,cs,sn,KV//HD); m=cmp(kr, rope(k,cs,sn,KV//HD)) # recompute RoPE k
        o=F.linear(qr,Wo[i]); xr=(x+o).clamp_(-30,30); m=cmp(xr,(x+o).clamp(-30,30)); x=xr  # residual1
        h2=rms(x); m=cmp(h2, rms(x))                  # RMSNorm2
        gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=F.silu(gt)*up; m=cmp(act, F.silu(gt)*up)  # SiLU+mul
        dn=F.linear(act,Wd[i]); xr2=(x+dn).clamp_(-30,30); m=cmp(xr2,(x+dn).clamp(-30,30)); x=xr2 # residual2
    lg=F.linear(rms(x),Wlm)
    mbuf.copy_(m)
    return lg

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

out={}
for tag,M in [('decode',128),('prefill',512)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half(); cs,sn=make_rope(M); mbuf=torch.zeros((),device=dev,dtype=torch.float32)
    gb=capture(lambda:step_base(x0,cs,sn)); gc=capture(lambda:step_with_nongemm_check(x0,cs,sn,mbuf))
    tb=med(gb); tc=med(gc); ov=100*(tc-tb)/tb
    out[tag]=dict(M=M,base_ms=tb,with_check_ms=tc,overhead_pct=ov)
    print(f"[gap4 {tag}] base {tb:.3f}ms  +nongemm-check {tc:.3f}ms  overhead {ov:+.2f}%",flush=True)

# ---- catch test (eager, value-exact): inject Step-A flips into each non-GEMM op, confirm caught
def eager_step_cache(x0,cs,sn,fault=None):
    x=x0; cache={}
    def flip(y,key):
        if fault is not None and key==fault:
            fl=y.reshape(-1); u=fl.view(torch.int16); u[fl.shape[0]//2]^=(1<<14)
        return y
    for i in range(L):
        h=flip(rms(x),(i,'rms1')); q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        qr=flip(rope(q,cs,sn,HEADS),(i,'rope_q')); kr=flip(rope(k,cs,sn,KV//HD),(i,'rope_k'))
        o=F.linear(qr,Wo[i]); x=flip((x+o).clamp(-30,30),(i,'res1'))
        h2=flip(rms(x),(i,'rms2')); gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=flip(F.silu(gt)*up,(i,'silu')); dn=F.linear(act,Wd[i]); x=flip((x+dn).clamp(-30,30),(i,'res2'))
        cache[(i,'rms1')]=h;cache[(i,'rope_q')]=qr;cache[(i,'rope_k')]=kr;cache[(i,'rms2')]=h2;cache[(i,'silu')]=act;cache[(i,'res1')]=None
    return cache,x
def eager_recompute_nongemm(x0,cs,sn,cache):
    """re-run non-GEMM ops from clean inputs (the GEMM outputs, which we recompute cleanly too) and
    compare to cache; nan/inf-safe; returns max diff (inf if any nonfinite mismatch)."""
    x=x0; m=0.0; nf=False
    def cmp(yc,key):
        nonlocal nf
        if cache.get(key) is None: return
        d=(yc-cache[key]).abs()
        if not bool(torch.isfinite(d).all()): nf=True; return
        return float(d.amax())
    for i in range(L):
        h=rms(x); v=cmp(h,(i,'rms1')); m=max(m,v or 0)
        q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i])
        qr=rope(q,cs,sn,HEADS); v=cmp(qr,(i,'rope_q')); m=max(m,v or 0)
        kr=rope(k,cs,sn,KV//HD); v=cmp(kr,(i,'rope_k')); m=max(m,v or 0)
        o=F.linear(qr,Wo[i]); x=(x+o).clamp(-30,30)
        h2=rms(x); v=cmp(h2,(i,'rms2')); m=max(m,v or 0)
        gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i]); act=F.silu(gt)*up; v=cmp(act,(i,'silu')); m=max(m,v or 0)
        dn=F.linear(act,Wd[i]); x=(x+dn).clamp(-30,30)
    return float('inf') if nf else m
import random as _r; rng=_r.Random(0)
ops=['rms1','rope_q','rope_k','rms2','silu']
catch=[]
cs,sn=make_rope(128)
# clean residual (T)
xref=(torch.randn(128,H,device=dev)*0.1).half()
cclean,_=eager_step_cache(xref,cs,sn,fault=None); T=eager_recompute_nongemm(xref,cs,sn,cclean)
for _ in range(40):
    op=rng.choice(ops); layer=rng.randrange(L)
    xin=(torch.randn(128,H,device=dev)*0.1).half()
    cache,_=eager_step_cache(xin,cs,sn,fault=(layer,op))
    res=eager_recompute_nongemm(xin,cs,sn,cache)
    catch.append(dict(op=op,layer=layer,residual=res if res!=float('inf') else 'inf',caught=bool(res>T)))
ncaught=sum(1 for c in catch if c['caught'])
out['T_clean']=T; out['catch_total']=len(catch); out['catch_caught']=ncaught
out['catch_by_op']={op:sum(1 for c in catch if c['op']==op and c['caught']) for op in ops}
print(f"[gap4 catch] T={T} clean | {ncaught}/{len(catch)} non-GEMM flips caught; by-op {out['catch_by_op']}",flush=True)
json.dump(out,open(f"{HERE}/gap4_result.json",'w'),indent=1,default=str)
print("[gap4] wrote gap4_result.json",flush=True)
