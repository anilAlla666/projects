import ctypes, os, statistics, json, math, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libkernel_dedup.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
lib.get_fire.restype=ctypes.c_uint; lib.reset_fire.restype=ctypes.c_int
BLK_M,BLK_N=128,256; NSM=132

def timed(fn, iters=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def run(M,K,N):
    R=N//BLK_N; row_tiles=math.ceil(M/BLK_M); total=row_tiles*math.ceil(N/BLK_N)
    waves=total/NSM
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def ck():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    lib.reset_fire(); ck(); torch.cuda.synchronize(); fires=lib.get_fire()
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=timed(ck)
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),2); t2=timed(ck)
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); t3=timed(ck)
    d2=(t2-t0)/t0*100; d3=(t3-t0)/t0*100
    print(f"  N={N:6d} K={K:6d} tiles={total:4d} waves={waves:5.2f}  fires={fires:3d}/{row_tiles:<3d} "
          f" base={t0*1000:7.1f}us  redund={d2:+6.2f}%  DEDUP={d3:+6.2f}%  {'PASS' if d3<3 else 'fail'}")
    return dict(M=M,K=K,N=N,total_tiles=total,waves=waves,fires=int(fires),row_tiles=row_tiles,
                base_us=t0*1000,redundant_pct=d2,dedup_pct=d3,pass3=bool(d3<3))

out={}
print("=== N-sweep @ K=4096 (short tiles, gate-like) — amortization vs waves ===")
out['k4096']=[run(2048,4096,N) for N in [1024,2048,3072,4096,6144,8192,12288,14336,20480,28672]]
print("\n=== N-sweep @ K=14336 (long tiles, down-like) — does K shift the threshold? ===")
out['k14336']=[run(2048,14336,N) for N in [4096,8192,14336,28672]]
json.dump(out,open('sweep_n_result.json','w'),indent=1)
print("\nwrote sweep_n_result.json")
