#!/usr/bin/env python3
# GATE-1 g1.3 PROBE B (decisive, names the op): does the SHIPPED Koopman/EDMD path execute inside the engine's
# torch.cuda.graph static-KV capture, or fire a capture-illegal op? We call the EXACT shipped function the cublas
# shim calls on every GEMM -- cipher_edmd_live_collect (cipher_rt_cublas_shim.c:239-244) -- INSIDE a torch.cuda.graph
# capture region. NO rebuild -- deployed libcipher_rt.so. CIPHER_RT_DISABLE_AUTO_INIT=1.
#
# WITNESS FIX: cipher_edmd_live_collect ALWAYS returns false (edmd_live.cpp:687/693/751/753 -- no return true; shim
# ignores it). The real engagement witness is s->rows via cipher_edmd_live_get_stats. The shipped collect also has a
# DISTINCT-INPUT gate (DISTINCT_INPUT_THRESHOLD=8): no cudaMemcpyAsync until 8 distinct activation ptrs seen. So drive
# 8+ distinct activations OUTSIDE capture (rows should climb), THEN call inside capture and watch (a) rows climb =
# memcpy fired, (b) cudaPeekAtLastError = did the in-capture op error. Let the RUNTIME name which op trips.
import os, sys, ctypes
os.environ["CIPHER_RT_DISABLE_AUTO_INIT"]="1"; os.environ["CIPHER_RT_DISABLE_CUINIT_HOOK"]="1"
import torch
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; FP16=2
cudart=ctypes.CDLL("libcudart.so")
cudart.cudaPeekAtLastError.restype=ctypes.c_int; cudart.cudaGetLastError.restype=ctypes.c_int
cudart.cudaGetErrorString.restype=ctypes.c_char_p; cudart.cudaGetErrorString.argtypes=[ctypes.c_int]
def cerr(peek=True):
    c=cudart.cudaPeekAtLastError() if peek else cudart.cudaGetLastError(); return (c, cudart.cudaGetErrorString(c).decode())
lib=ctypes.CDLL(SO)
lib.cipher_edmd_live_collect.restype=ctypes.c_bool
lib.cipher_edmd_live_collect.argtypes=[ctypes.c_int]*3+[ctypes.c_int,ctypes.c_void_p]*3
lib.cipher_edmd_live_get_stats.restype=ctypes.c_bool
lib.cipher_edmd_live_get_stats.argtypes=[ctypes.c_int,ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int)]
def rows(K,N):
    rr=ctypes.c_float(); ee=ctypes.c_float(); md=ctypes.c_float(); rw=ctypes.c_int(-1)
    ok=lib.cipher_edmd_live_get_stats(K,N,ctypes.byref(rr),ctypes.byref(ee),ctypes.byref(md),ctypes.byref(rw))
    return (rw.value if ok else None)   # None = shape not registered/failed yet (get_stats gates on registered||failed)

K,N=2048,2048
A=torch.randn(N,K,dtype=torch.float16,device="cuda").contiguous()      # weight
C=torch.zeros(N,1,dtype=torch.float16,device="cuda").contiguous()      # output
acts=[torch.randn(K,1,dtype=torch.float16,device="cuda").contiguous() for _ in range(16)]  # distinct activations
def collect(Bt): return lib.cipher_edmd_live_collect(1,K,N, FP16,A.data_ptr(), FP16,Bt.data_ptr(), FP16,C.data_ptr())

for _ in range(5): _=torch.matmul(A,acts[0])   # warm cuBLAS
torch.cuda.synchronize(); _=cerr(peek=False)

# ---- drive distinct-input gate OUTSIDE capture; witness = does the shipped cudaMemcpyAsync land snapshots? ----
print("=== WARM (outside capture): drive 8+ distinct activations, witness s->rows via get_stats ===",flush=True)
for i,b in enumerate(acts):
    collect(b); torch.cuda.synchronize()
    # rows readable only once a result exists; use a parallel direct count by re-reading after each
print(f"  after 16 distinct collects: cudaErr={cerr()} (legal outside capture)",flush=True)
# Force a definitive witness: keep collecting until rows climbs OR many calls (rows visible after registered/failed;
# pre-registration get_stats returns None, so instead we detect the memcpy by a KNOWN side effect: the snapshot path
# requires the gate passed. We verify gate-pass differently below via the capture differential.)

# ---- CONTROL: capture real fp16 GEMM WITHOUT collect -> must succeed ----
print("\n=== CONTROL: torch.cuda.graph capture, NO collect ===",flush=True)
ctl=False
try:
    g0=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        Cc=torch.matmul(A,acts[0])
    g0.replay(); torch.cuda.synchronize(); ctl=True
    print("  CONTROL capture+replay SUCCEEDED",flush=True)
except Exception as e: print(f"  CONTROL FAILED (confound): {type(e).__name__}: {str(e)[:140]}",flush=True)

# ---- TEST warm: steady-state memcpy path inside capture (gate already passed via 16 distinct above) ----
print("\n=== TEST warm: shipped collect INSIDE capture (steady-state -> cudaMemcpyAsync D2H stream0 @ :217) ===",flush=True)
torch.cuda.synchronize(); _=cerr(peek=False)
warm_err=None; warm_inside=None
try:
    gt=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gt):
        Cc=torch.matmul(A,acts[1])
        collect(acts[2])                       # steady-state shipped snapshot
        warm_inside=cerr(peek=True)            # name the EXACT in-capture error (or 'no error' = memcpy not illegal)
    gt.replay(); torch.cuda.synchronize()
    print(f"  capture COMPLETED. in-capture cudaPeekErr={warm_inside}",flush=True)
except Exception as e:
    warm_err=f"{type(e).__name__}: {str(e)[:200]}"; print(f"  in-capture cudaPeekErr={warm_inside}",flush=True); print(f"  capture FAILED -> {warm_err}",flush=True)

# ---- TEST cold: 8th-distinct collect inside capture -> ensure_capture_buffers -> cudaHostAlloc(:171) ----
print("\n=== TEST cold: 8th-distinct collect inside capture (cold alloc path -> cudaHostAlloc :171) ===",flush=True)
cold_err=None; cold_inside=None
try:
    K2,N2=1536,1536
    A2=torch.randn(N2,K2,dtype=torch.float16,device="cuda").contiguous()   # weight (N2 x K2) -- correct GEMM shape
    C2=torch.zeros(N2,1,dtype=torch.float16,device="cuda").contiguous()
    acts2=[torch.randn(K2,1,dtype=torch.float16,device="cuda").contiguous() for _ in range(8)]
    for _ in range(3): _=torch.matmul(A2,acts2[0])
    torch.cuda.synchronize()
    for b in acts2[:7]:                        # 7 distinct OUTSIDE: gate NOT passed, NO cudaHostAlloc yet
        lib.cipher_edmd_live_collect(1,K2,N2,FP16,A2.data_ptr(),FP16,b.data_ptr(),FP16,C2.data_ptr())
    torch.cuda.synchronize(); _=cerr(peek=False)
    g2=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        Cc=torch.matmul(A2,acts2[7])
        lib.cipher_edmd_live_collect(1,K2,N2,FP16,A2.data_ptr(),FP16,acts2[7].data_ptr(),FP16,C2.data_ptr())  # 8th -> gate passes -> cudaHostAlloc :171
        cold_inside=cerr(peek=True)
    g2.replay(); torch.cuda.synchronize()
    print(f"  capture COMPLETED. in-capture cudaPeekErr={cold_inside}",flush=True)
except Exception as e:
    cold_err=f"{type(e).__name__}: {str(e)[:200]}"; print(f"  in-capture cudaPeekErr={cold_inside}",flush=True); print(f"  capture FAILED -> {cold_err}",flush=True)

print("\n=== g1.3 PROBE B SUMMARY ===",flush=True)
print(f"  CONTROL (no collect) captured: {ctl}",flush=True)
print(f"  TEST warm (memcpy path):   in-capture err={warm_inside}  capture_failed={warm_err is not None}  {warm_err or ''}",flush=True)
print(f"  TEST cold (hostAlloc path): in-capture err={cold_inside}  capture_failed={cold_err is not None}  {cold_err or ''}",flush=True)
print("  -> the op that ACTUALLY trips (nonzero cudaErr / capture_failed) is the named capture-illegal op; cite that line.",flush=True)
sys.stdout.flush(); os._exit(0)
