#!/usr/bin/env python3
# Correct per-phase overlap test under saturation: does the detector (v+Cg reads only, NOT a second
# forward) hide under the serving step when run on a concurrent stream? Decode (memory-bound, MEM~87%)
# vs prefill (compute-bound, MEM~48%). Answers whether idle-GPU spare bandwidth survives saturation.
import ctypes, sys, statistics, json, time, torch
torch.manual_seed(0); dev='cuda'
M=int(sys.argv[1]) if len(sys.argv)>1 else 128
tk=ctypes.CDLL('/home/ubuntu/cipher-fusion-evidence/step_b_freivalds/libtuned_gemv.so')
tk.rowgemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; tk.rowgemv.restype=ctypes.c_int
H,I,KV,V,L=4096,14336,1024,32000,32
F=torch.nn.functional
def W(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
Wq=[W(H,H) for _ in range(L)];Wk=[W(KV,H) for _ in range(L)];Wv=[W(KV,H) for _ in range(L)]
Wo=[W(H,H) for _ in range(L)];Wg=[W(I,H) for _ in range(L)];Wu=[W(I,H) for _ in range(L)];Wd=[W(H,I) for _ in range(L)]
Wlm=W(V,H); x0=(torch.randn(M,H,device=dev)*0.1).half()
# dummy activation tensors the detector reads (correct byte traffic), + probe vectors
aH=torch.randn(M,H,device=dev).half(); aI=torch.randn(M,I,device=dev).half()
cH=torch.randn(M,H,device=dev).half(); cKV=torch.randn(M,KV,device=dev).half()
cI=torch.randn(M,I,device=dev).half(); cV=torch.randn(M,V,device=dev).half()
uH=torch.randn(H,device=dev).half(); uI=torch.randn(I,device=dev).half()
gH=torch.randn(H,device=dev).half(); gKV=torch.randn(KV,device=dev).half(); gI=torch.randn(I,device=dev).half(); gV=torch.randn(V,device=dev).half()
yv=torch.zeros(M,device=dev,dtype=torch.float32); yc=torch.zeros(M,device=dev,dtype=torch.float32)
def detector_only(st):  # fire ALL the per-layer checks (v reads A, Cg reads C), no forward
    for _ in range(L):
        for A,Kd in [(aH,H),(aH,H),(aH,H),(aH,H),(aH,H),(aH,H),(aI,I)]:
            tk.rowgemv(A.data_ptr(),(uH if Kd==H else uI).data_ptr(),yv.data_ptr(),M,Kd,st)
        for C,Nd in [(cH,H),(cKV,KV),(cKV,KV),(cH,H),(cI,I),(cI,I),(cH,H)]:
            tk.rowgemv(C.data_ptr(),(gH if Nd==H else gKV if Nd==KV else gI).data_ptr(),yc.data_ptr(),M,Nd,st)
    tk.rowgemv(aH.data_ptr(),uH.data_ptr(),yv.data_ptr(),M,H,st); tk.rowgemv(cV.data_ptr(),gV.data_ptr(),yc.data_ptr(),M,V,st)
def step():
    x=x0
    for i in range(L):
        q=F.linear(x,Wq[i]);F.linear(x,Wk[i]);F.linear(x,Wv[i]);o=F.linear(q,Wo[i]);x=x+o
        gt=F.linear(x,Wg[i]);up=F.linear(x,Wu[i]);inter=F.silu(gt)*up;dn=F.linear(inter,Wd[i]);x=x+dn
    return F.linear(x,Wlm)
def capture(fn):
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
gbase=capture(step)
sA=torch.cuda.Stream();sB=torch.cuda.Stream()
def wall(fn,iters=40,warm=12):
    for _ in range(warm): fn()
    torch.cuda.synchronize();ts=[]
    for _ in range(iters):
        torch.cuda.synchronize();t0=time.perf_counter();fn();torch.cuda.synchronize();ts.append((time.perf_counter()-t0)*1e3)
    return statistics.median(ts)
def base(): gbase.replay()
def serial(): gbase.replay(); detector_only(torch.cuda.current_stream().cuda_stream)
def concurrent():
    e=torch.cuda.Event();torch.cuda.current_stream().record_event(e);sA.wait_event(e);sB.wait_event(e)
    with torch.cuda.stream(sA): gbase.replay()
    with torch.cuda.stream(sB): detector_only(sB.cuda_stream)
    sA.synchronize();sB.synchronize()
wb=wall(base);ws=wall(serial);wc=wall(concurrent)
det=ws-wb; hidden=100*(ws-wc)/det if det>0 else 0
regime="DECODE(mem-bound)" if M<=256 else "PREFILL(compute-bound)"
print(f"M={M} {regime}: base {wb:.2f}ms  serial(+det) {ws:.2f}ms  concurrent {wc:.2f}ms")
print(f"  detector serial cost {det:.2f}ms (+{100*det/wb:.1f}%)  | hidden by overlap: {hidden:.0f}%  "
      f"({'overlap HELPS' if hidden>20 else 'NO overlap'})")
json.dump(dict(M=M,base_ms=wb,serial_ms=ws,concurrent_ms=wc,det_ms=det,
               det_pct=100*det/wb,hidden_pct=hidden),open(f'overlap_M{M}.json','w'),indent=1)
