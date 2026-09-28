#!/usr/bin/env python3
# PREFETCH PROBE-FIRST (the open contention question from go1-binding-limit): does an async H2D page-in OVERLAP a
# concurrent serve (GPU compute) without serializing, at stable latency + correct data? + the honest correlated-case
# check: K concurrent H2D prefetches share the ONE PCIe pipe -> does aggregate page-in throughput stay ~51GB/s (so
# K prefetches still serialize to K x single, i.e. prefetch reorders but cannot create PCIe bandwidth under bursts)?
import torch, time, sys, os
GB=1<<30; fp=torch.float16
def gbps(nbytes, sec): return nbytes/sec/1e9
# A's serve proxy: HBM-bound matmul loop (~ a decode burst keeping the GPU busy)
W=torch.randn(8192,8192,device="cuda",dtype=fp)
def compute(iters):
    x=W
    for _ in range(iters): x=x@W
    return x
# B's page-in: 5 GiB pinned host -> GPU (one 4-bit 7-8B model's weights)
N=5*GB; pinned=torch.empty(N//2, dtype=fp, pin_memory=True); pinned.uniform_(-1,1)
dst=torch.empty(N//2, dtype=fp, device="cuda")
cs=torch.cuda.Stream()
def sync(): torch.cuda.synchronize()
# calibrate compute iters ~ 250ms (a serve window)
sync(); t=time.time(); compute(1); sync(); per=time.time()-t; iters=max(1,int(0.25/per))
# baselines
sync(); t=time.time(); compute(iters); sync(); comp_alone=time.time()-t
sync(); t=time.time()
with torch.cuda.stream(cs): dst.copy_(pinned, non_blocking=True)
cs.synchronize(); h2d_alone=time.time()-t
print(f"baselines: compute_alone={comp_alone*1000:.0f}ms  H2D(5GiB)_alone={h2d_alone*1000:.0f}ms = {gbps(N,h2d_alone):.0f} GB/s",flush=True)
# OVERLAP: H2D on copy stream + compute on default, concurrently
sync(); t=time.time()
with torch.cuda.stream(cs): dst.copy_(pinned, non_blocking=True)
compute(iters)
sync(); overlap=time.time()-t
serial=comp_alone+h2d_alone; perfect=max(comp_alone,h2d_alone)
print(f"OVERLAP wall={overlap*1000:.0f}ms  (serial={serial*1000:.0f}ms, perfect-overlap={perfect*1000:.0f}ms) -> {'OVERLAPS' if overlap<0.85*serial else 'SERIALIZES'} ({serial/overlap:.2f}x)",flush=True)
ok=torch.allclose(dst.float(), pinned.float().cuda(), atol=0); print(f"  H2D-under-compute data correct (bit-identical) = {ok}",flush=True)
# CORRELATED case: K concurrent H2D prefetches share the PCIe pipe -> aggregate throughput
for K in [1,2,4]:
    dsts=[torch.empty(N//2,dtype=fp,device="cuda") for _ in range(K)]
    streams=[torch.cuda.Stream() for _ in range(K)]
    sync(); t=time.time()
    for k in range(K):
        with torch.cuda.stream(streams[k]): dsts[k].copy_(pinned, non_blocking=True)
    sync(); dt=time.time()-t
    print(f"  K={K} concurrent prefetches: {dt*1000:.0f}ms total, aggregate {gbps(K*N,dt):.0f} GB/s (per-model {dt/K*1000:.0f}ms) -> {'PCIe-SHARED (no speedup)' if gbps(K*N,dt)<1.5*gbps(N,h2d_alone) else 'scales'}",flush=True)
print("  -> prefetch HIDES one page-in behind a serve window (if overlap), but K correlated misses still share ~51GB/s PCIe (reorder != bandwidth)",flush=True)
sys.stdout.flush(); os._exit(0)
