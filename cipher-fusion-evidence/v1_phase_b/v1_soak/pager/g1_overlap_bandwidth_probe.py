#!/usr/bin/env python3
# PROBE-FIRST (load-bearing, before the months-engine): is the multi-model latency floor REMOVABLE by cross-engine
# OVERLAP, or HBM-BANDWIDTH-bound (overlap can't help -> lever is graph-decode/batching)? Decode = GEMV reading ALL
# weights/token. If one decode SATURATES HBM bw -> two concurrent share one ~3.35TB/s pipe -> no overlap.
#
# Instrument (advisor-corrected): NOT an eager transformers forward (that is dispatch/GIL-bound, ~200 launches +
# Python attention path -> would measure the GIL, not HBM). Instead a GRAPH-CAPTURED GEMV microbench sized to the
# weight BYTES read per token: x[1,K] @ W[K,K] reads bytes(W) -> memory-bound like decode. Graph replay = one async
# launch -> graphA-on-streamA || graphB-on-streamB is a TRUE GPU overlap test (no per-kernel Python). Measure at the
# ENGINE'S regime: int4 (~4GB/7B, what the pager ships) AND fp16 (~14GB, the router's regime) -- the answer FLIPS
# with precision. Prior arithmetic cross-check: graph-decode ~167 tok/s/agent -> 167*14GB=2.3TB/s=70% (fp16,
# bw-bound) vs 167*4GB=0.67TB/s=20% (int4, launch-bound). Confirm, don't discover. Interpret numbers MANUALLY.
import os, time, sys, torch
H100_PEAK=3.35  # TB/s
torch.cuda.init()
def K_for_bytes(b): import math; return int(math.isqrt(b//2))   # fp16 [K,K] = 2*K^2 bytes
STEPS=64
def run_regime(label, wbytes):
    K=K_for_bytes(wbytes)
    WA=torch.randn(K,K,device="cuda",dtype=torch.float16); WB=torch.randn(K,K,device="cuda",dtype=torch.float16)
    xA=torch.randn(1,K,device="cuda",dtype=torch.float16); xB=torch.randn(1,K,device="cuda",dtype=torch.float16)
    yA=torch.empty(1,K,device="cuda",dtype=torch.float16); yB=torch.empty(1,K,device="cuda",dtype=torch.float16)
    realbytes=WA.numel()*2
    # capture each GEMV as a CUDA graph (true single-launch async replay)
    for _ in range(5): torch.mm(xA,WA,out=yA); torch.mm(xB,WB,out=yB)
    torch.cuda.synchronize()
    gA=torch.cuda.CUDAGraph(); gB=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gA): torch.mm(xA,WA,out=yA)
    with torch.cuda.graph(gB): torch.mm(xB,WB,out=yB)
    def time_single(g):
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(STEPS): g.replay()
        torch.cuda.synchronize(); return time.time()-t0
    dtA=time_single(gA); dtB=time_single(gB)
    bwA=realbytes*STEPS/dtA/1e12; bwB=realbytes*STEPS/dtB/1e12
    # serialized vs concurrent (two streams, async graph replays)
    sA=torch.cuda.Stream(); sB=torch.cuda.Stream()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS): gA.replay()
    for _ in range(STEPS): gB.replay()
    torch.cuda.synchronize(); ser=time.time()-t0
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(STEPS):
        with torch.cuda.stream(sA): gA.replay()
        with torch.cuda.stream(sB): gB.replay()
    torch.cuda.synchronize(); con=time.time()-t0
    aggbw=2*realbytes*STEPS/con/1e12
    print(f"[{label}] per-'model' weight-read={realbytes/(1<<30):.1f}GiB (K={K})",flush=True)
    print(f"   (A) single GEMV decode: A bw={bwA:.2f}TB/s ({bwA/H100_PEAK*100:.0f}% peak)  B bw={bwB:.2f}TB/s ({bwB/H100_PEAK*100:.0f}% peak)",flush=True)
    print(f"   (B) serialized={ser*1000:.0f}ms  concurrent={con*1000:.0f}ms  speedup={ser/con:.2f}x",flush=True)
    print(f"   (C) concurrent aggregate bw={aggbw:.2f}TB/s ({aggbw/H100_PEAK*100:.0f}% peak)  [>~70% = HBM is the wall, overlap dead]",flush=True)
    del WA,WB,xA,xB,yA,yB,gA,gB; torch.cuda.empty_cache()
    return bwA,ser/con,aggbw
print("=== overlap-vs-bandwidth probe (graph-replayed GEMV microbench; manual interpretation) ===",flush=True)
print(f"PRIOR arithmetic cross-check (graph-decode ~167 tok/s/agent): fp16 14GB->{167*14e9/1e12:.1f}TB/s "
      f"({167*14e9/1e12/H100_PEAK*100:.0f}% peak); int4 4GB->{167*4e9/1e12:.2f}TB/s ({167*4e9/1e12/H100_PEAK*100:.0f}% peak)",flush=True)
run_regime("int4-7B (4GB, engine regime)", 4*(1<<30))
run_regime("fp16-7B (14GB, router regime)", 14*(1<<30))
print("INTERPRET MANUALLY: int4 launch-bound + overlap-helps => cross-engine concurrency is a real increment-2 lever;",flush=True)
print("int4 bw-bound OR no-overlap => lever is graph-decode/same-model-batching, NOT overlap. Same-model batching",flush=True)
print("(prior 13.77x) amortizes one weight-read across the batch and stands regardless.",flush=True)
sys.stdout.flush(); os._exit(0)
