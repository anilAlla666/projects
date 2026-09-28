#!/usr/bin/env python3
"""
GAP 1 — phase-matched per-step transient coverage. THE HEART.
Mandate hypothesis: decode is memory-bound (spare COMPUTE), prefill compute-bound (spare BANDWIDTH);
match the check to the IDLE resource. Gap 2 showed the row-sum check is BANDWIDTH-heavy (re-reads Y),
not compute-heavy. So the real, novel test (advisor): can a bandwidth-heavy check OVERLAP the
compute-bound prefill GEMMs on a CONCURRENT stream, hiding under spare HBM (MEM 48-52%)? And does the
same check COLLIDE in bandwidth-bound decode (no spare BW)?

Measures per phase: base (GEMMs only) | serial-check (check on same stream = Gap-2 ballpark) |
concurrent-overlap (check pipelined on a 2nd stream, layer i check overlaps layer i+1 compute).
Prefill is compute-bound => eager wall-clock is GPU-bound (validated vs graph-base). fp32 reductions.
READ-ONLY, not injected. Rule-4 roofline floor printed.
"""
import json, statistics, torch, torch.nn.functional as F
torch.manual_seed(0); dev='cuda'
H,I,KV,L = 4096,14336,1024,32
HERE='/home/ubuntu/cipher-fusion-evidence/full_coverage/gap1_phasematched'
PEAK_BW=2.97e12

def Wt(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[Wt(H,H) for _ in range(L)]; Wk=[Wt(KV,H) for _ in range(L)]; Wv=[Wt(KV,H) for _ in range(L)]
Wo=[Wt(H,H) for _ in range(L)]; Wg=[Wt(I,H) for _ in range(L)]; Wu=[Wt(I,H) for _ in range(L)]; Wd=[Wt(H,I) for _ in range(L)]
def sv(Ws): return [w.float().sum(0) for w in Ws]
Sq,Sk,Sv,So,Sg,Su,Sd = sv(Wq),sv(Wk),sv(Wv),sv(Wo),sv(Wg),sv(Wu),sv(Wd)
def rms(t): return t*torch.rsqrt(t.float().pow(2).mean(-1,keepdim=True)+1e-5).half()
main = torch.cuda.current_stream(); chk_stream = torch.cuda.Stream()

def step(x0, mode):
    """mode: base | serial | overlap. Returns (x, mbox)."""
    x=x0; mbox=torch.zeros((),device=dev,dtype=torch.float32); keep=[]
    for i in range(L):
        h=rms(x); q=F.linear(h,Wq[i]); k=F.linear(h,Wk[i]); v=F.linear(h,Wv[i])
        o=F.linear(q,Wo[i]); x=(x+o).clamp(-30,30)
        h2=rms(x); gt=F.linear(h2,Wg[i]); up=F.linear(h2,Wu[i])
        act=F.silu(gt)*up; dn=F.linear(act,Wd[i]); x=(x+dn).clamp(-30,30)
        if mode=='base': continue
        checks=[(q,Sq[i],h),(k,Sk[i],h),(v,Sv[i],h),(o,So[i],q),(gt,Sg[i],h2),(up,Su[i],h2),(dn,Sd[i],act)]
        if mode=='overlap':
            ev=torch.cuda.Event(); ev.record(main)
            with torch.cuda.stream(chk_stream):
                chk_stream.wait_event(ev)
                for Y,s,xin in checks:
                    mbox=torch.maximum(mbox,(xin.float()@s - Y.float().sum(1)).abs().amax())
            keep.extend([t for t,_,_ in checks]+[xin for _,_,xin in checks])
        else:
            for Y,s,xin in checks:
                mbox=torch.maximum(mbox,(xin.float()@s - Y.float().sum(1)).abs().amax())
    if mode=='overlap': main.wait_stream(chk_stream)
    return x,mbox

def timed(mode, x0, reps=40, warm=12):
    for _ in range(warm): step(x0,mode)
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True)
        e0.record(); step(x0,mode); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

def capture_base(x0):
    s=torch.cuda.Stream(); s.wait_stream(main)
    with torch.cuda.stream(s):
        for _ in range(3): step(x0,'base')
    main.wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step(x0,'base')
    torch.cuda.synchronize()
    for _ in range(12): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(40):
        e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record();g.replay();e1.record();torch.cuda.synchronize();ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

# Rule-4 roofline floor: check BW = re-read X+Y per GEMM; spare BW per phase
gemms=[('q',H,H),('k',KV,H),('v',KV,H),('o',H,H),('gate',I,H),('up',I,H),('down',H,I)]
out={}
for tag,M,memutil in [('decode',128,0.79),('prefill',512,0.50)]:
    x0=(torch.randn(M,H,device=dev)*0.1).half()
    bytes_chk=sum((M*K+M*N) for _,N,K in gemms)*2*L
    chk_bw_ms=1e3*bytes_chk/PEAK_BW
    spare_bw=PEAK_BW*(1.0-memutil)            # idle HBM in this phase
    chk_bw_ms_spare=1e3*bytes_chk/max(1.0,spare_bw)
    tb_eager=timed('base',x0); tb_graph=capture_base(x0)
    ts=timed('serial',x0); to=timed('overlap',x0)
    out[tag]=dict(M=M, mem_util=memutil, base_eager_ms=tb_eager, base_graph_ms=tb_graph,
                  serial_ms=ts, overlap_ms=to,
                  serial_overhead_pct=100*(ts-tb_eager)/tb_eager,
                  overlap_overhead_pct=100*(to-tb_eager)/tb_eager,
                  chk_bytes=bytes_chk, chk_bw_ms_peak=chk_bw_ms, chk_bw_pct_peak=100*chk_bw_ms/tb_eager,
                  chk_bw_ms_spare=chk_bw_ms_spare, chk_bw_pct_spare=100*chk_bw_ms_spare/tb_eager)
    print(f"[{tag}] base eager {tb_eager:.3f} graph {tb_graph:.3f}ms (eager~graph? {abs(tb_eager-tb_graph)/tb_graph*100:.1f}%) | "
          f"serial +{out[tag]['serial_overhead_pct']:.1f}% | OVERLAP +{out[tag]['overlap_overhead_pct']:.1f}% | "
          f"chk-BW {chk_bw_ms:.3f}ms peak={out[tag]['chk_bw_pct_peak']:.1f}% spare({1-memutil:.0%})={out[tag]['chk_bw_pct_spare']:.1f}%",flush=True)

json.dump(out,open(f"{HERE}/gap1_result.json",'w'),indent=1,default=str)
print("[gap1] wrote gap1_result.json",flush=True)
