#!/usr/bin/env python3
# THROWAWAY: is cudaMallocAsync/cudaFreeAsync (stream-ordered) capture-legal inside a torch.cuda.graph,
# and does it conflict with torch's graph-capture memory pool? (the chosen capture-safe alloc mechanism)
# If this captures+replays clean, swapping Marlin's rt_malloc->cudaMallocAsync makes the cast capture-safe.
import ctypes, torch, traceback
import warnings; warnings.filterwarnings("ignore")
rt=ctypes.CDLL("libcudart.so")
rt.cudaMallocAsync.restype=ctypes.c_int;  rt.cudaMallocAsync.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_void_p]
rt.cudaFreeAsync.restype=ctypes.c_int;    rt.cudaFreeAsync.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
rt.cudaMemsetAsync.restype=ctypes.c_int;  rt.cudaMemsetAsync.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t,ctypes.c_void_p]
rt.cudaGetErrorString.restype=ctypes.c_char_p; rt.cudaGetErrorString.argtypes=[ctypes.c_int]
def ck(rc,what):
    if rc!=0: print(f"   {what} -> CUDA err {rc}: {rt.cudaGetErrorString(rc).decode()}")
    return rc

torch.cuda.init(); x=torch.ones(256,device="cuda")  # ensure ctx + default pool exist
SZ=1<<20  # 1 MiB

# baseline: cudaMallocAsync works eagerly?
p=ctypes.c_void_p(); st=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
rc=ck(rt.cudaMallocAsync(ctypes.byref(p),SZ,st),"eager mallocAsync");
ck(rt.cudaMemsetAsync(p,0,SZ,st),"eager memsetAsync"); ck(rt.cudaFreeAsync(p,st),"eager freeAsync")
torch.cuda.synchronize(); print("eager cudaMallocAsync/freeAsync: OK" if rc==0 else "eager FAILED")

# THE TEST: cudaMallocAsync INSIDE torch.cuda.graph capture (does it invalidate capture / pool-conflict?)
print("\ncapture: torch.cuda.graph with a cudaMallocAsync+memset+freeAsync inside —")
g=torch.cuda.CUDAGraph(); ok=False; err=None
try:
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pp=ctypes.c_void_p(); stt=ctypes.c_void_p(s.cuda_stream)
        rt.cudaMallocAsync(ctypes.byref(pp),SZ,stt); rt.cudaMemsetAsync(pp,1,SZ,stt); rt.cudaFreeAsync(pp,stt)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    out=torch.empty(256,device="cuda")
    with torch.cuda.graph(g):
        cs=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        q=ctypes.c_void_p()
        rt.cudaMallocAsync(ctypes.byref(q),SZ,cs)      # alloc inside capture (stream-ordered -> should be a graph mem node)
        rt.cudaMemsetAsync(q,7,SZ,cs)                  # use it
        out.copy_(x*2.0)                                # a torch op too (mixed pool)
        rt.cudaFreeAsync(q,cs)
    ok=True
except Exception as e:
    err=repr(e); traceback.print_exc()
print(f"  capture inside-async-alloc succeeded: {ok}" + (f"  ERR={err}" if err else ""))
if ok:
    out.zero_(); g.replay(); torch.cuda.synchronize()
    print(f"  replay ok; out[0]={out[0].item()} (expect 2.0 => graph + async-alloc node replays clean)")
    print("  => cudaMallocAsync is CAPTURE-LEGAL on this cu13/torch; use it for the capture-safe cast.")
