#!/usr/bin/env python3
# Saturated decode step via CUDA graph (no launch gaps) = the real serving decode regime, which
# eager HF cannot reach (launch-bound). One step streams all Mistral-7B weights (~14.3GB) through
# the 7 linears x 32 layers + lm_head at batch B. Measures: (1) saturated decode step time & tok/s,
# (2) MEM% during decode (memory-bound? -> spare BW for overlap?), (3) in-loop detector cost
# (v=A@u + Cg=C@g per linear) serial and (4) on a concurrent stream (overlap under saturation).
import ctypes, os, sys, statistics, json, subprocess, time, torch
torch.manual_seed(0); dev='cuda'
B=int(sys.argv[1]) if len(sys.argv)>1 else 128
tk=ctypes.CDLL('/home/ubuntu/cipher-fusion-evidence/step_b_freivalds/libtuned_gemv.so')
tk.rowgemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; tk.rowgemv.restype=ctypes.c_int
H,I,KV,V,L=4096,14336,1024,32000,32
def W(n,k): return (torch.randn(n,k,device=dev)*0.02).half()
# per-layer weights
Wq=[W(H,H) for _ in range(L)]; Wk=[W(KV,H) for _ in range(L)]; Wv=[W(KV,H) for _ in range(L)]
Wo=[W(H,H) for _ in range(L)]; Wg=[W(I,H) for _ in range(L)]; Wu=[W(I,H) for _ in range(L)]; Wd=[W(H,I) for _ in range(L)]
Wlm=W(V,H)
print(f"weights alloc: {torch.cuda.memory_allocated()/1e9:.1f} GB  (B={B})")
x0=(torch.randn(B,H,device=dev)*0.1).half()
# detector probe vectors (fixed g => u amortized): per distinct shape
gH=torch.randn(H,device=dev).half(); gKV=torch.randn(KV,device=dev).half()
gI=torch.randn(I,device=dev).half(); gV=torch.randn(V,device=dev).half()
uH=torch.randn(H,device=dev).half(); uI=torch.randn(I,device=dev).half()  # u=W^T g precomputed (values irrelevant for timing)
yv=torch.zeros(B,device=dev,dtype=torch.float32); yc=torch.zeros(B,device=dev,dtype=torch.float32)
F=torch.nn.functional
def chk(A,Kdim,C,Ndim,st):  # fire v=A@u (read A) + Cg=C@g (read C) -- the fixed-g check
    tk.rowgemv(A.data_ptr(), (uH if Kdim==H else uI).data_ptr(), yv.data_ptr(), B, Kdim, st)
    tk.rowgemv(C.data_ptr(), (gH if Ndim==H else gKV if Ndim==KV else gI if Ndim==I else gV).data_ptr(), yc.data_ptr(), B, Ndim, st)

def step(detector=False, st=None):
    if detector and st is None: st=torch.cuda.current_stream().cuda_stream  # launch on capture stream
    x=x0
    for i in range(L):
        q=F.linear(x,Wq[i]); k=F.linear(x,Wk[i]); v=F.linear(x,Wv[i]); o=F.linear(q,Wo[i])
        x=x+o
        gt=F.linear(x,Wg[i]); up=F.linear(x,Wu[i]); inter=F.silu(gt)*up; dn=F.linear(inter,Wd[i]); x=x+dn
        if detector:
            chk(x,H,q,H,st); chk(x,H,k,KV,st); chk(x,H,v,KV,st); chk(q,H,o,H,st)
            chk(x,H,gt,I,st); chk(x,H,up,I,st); chk(inter,I,dn,H,st)
    lg=F.linear(x,Wlm)
    if detector: chk(x,H,lg,V,st)
    return lg

def capture(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
def med_replay(g, iters=50, warm=15):
    for _ in range(warm): g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

g_base=capture(lambda: step(False))
g_det =capture(lambda: step(True, None))   # detector kernels in default stream (serial, in-graph)
t_base=med_replay(g_base); t_det=med_replay(g_det)
det_ov=100*(t_det-t_base)/t_base
print(f"saturated decode step: base {t_base:.2f}ms ({B/(t_base/1e3):.0f} tok/s)  +detector {t_det:.2f}ms  serial overhead {det_ov:+.2f}%")

# overlap: replay base graph on stream A, fire detector kernels on stream B concurrently
sA=torch.cuda.Stream(); sB=torch.cuda.Stream()
# Overlap: time base-only wall vs (base on sA + detector-step on sB) wall, host-timed
def wall(fn, iters=40, warm=12):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
    return statistics.median(ts)
def base_only(): g_base.replay()
def serial_both(): g_det.replay()
def concurrent():
    e=torch.cuda.Event(); torch.cuda.current_stream().record_event(e)
    sA.wait_event(e); sB.wait_event(e)
    with torch.cuda.stream(sA): g_base.replay()
    with torch.cuda.stream(sB): step(True, sB.cuda_stream)   # detector kernels only, on sB
    sA.synchronize(); sB.synchronize()
w_base=wall(base_only); w_serial=wall(serial_both); w_conc=wall(concurrent)
hidden=100*(w_serial-w_conc)/(w_serial-w_base) if w_serial>w_base else 0
print(f"overlap wall: base {w_base:.2f}ms  serial(base+det) {w_serial:.2f}ms  concurrent {w_conc:.2f}ms  hidden {hidden:.0f}% of detector")
json.dump(dict(B=B,t_base_ms=t_base,t_det_ms=t_det,det_serial_pct=det_ov,
               w_base_ms=w_base,w_serial_ms=w_serial,w_conc_ms=w_conc,hidden_pct=hidden,
               decode_toks_s=B/(t_base/1e3)),open(f'synth_decode_B{B}.json','w'),indent=1)
print(f"wrote synth_decode_B{B}.json")
