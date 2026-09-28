import ctypes, os, statistics, json, math, subprocess, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libkernel_dedup.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
lib.get_fire.restype=ctypes.c_uint; lib.reset_fire.restype=ctypes.c_int
BLK_M,BLK_N=128,256; NSM=132

def med(fn, iters=120):
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def clocks():
    o=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu",
                      "--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    return o

def study(name,M,K,N,reps=6):
    R=N//BLK_N; row_tiles=math.ceil(M/BLK_M); total=row_tiles*math.ceil(N/BLK_N); waves=total/NSM
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def ck():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    # verify gate
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    lib.reset_fire(); ck(); torch.cuda.synchronize(); fires=lib.get_fire()
    # warmup both modes
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0)
    for _ in range(80): ck()
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    for _ in range(80): ck()
    deltas=[]; b0s=[]
    for _ in range(reps):
        lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=med(ck)
        lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); t3=med(ck)
        deltas.append((t3-t0)/t0*100); b0s.append(t0*1000)
    mu=statistics.mean(deltas); sd=statistics.pstdev(deltas)
    lo,hi=min(deltas),max(deltas)
    verdict='PASS' if hi<3 else ('FAIL' if lo>=3 else 'STRADDLE-3%')
    print(f"  {name:16s} K={K:6d} N={N:6d} waves={waves:5.2f} fires={fires:3d}/{row_tiles:<3d} "
          f"base={statistics.mean(b0s):7.1f}us  dedup={mu:+5.2f}% ±{sd:.2f}  [{lo:+.2f}..{hi:+.2f}]  {verdict}")
    return dict(name=name,M=M,K=K,N=N,waves=waves,fires=int(fires),row_tiles=row_tiles,
                base_us=statistics.mean(b0s),dedup_mean_pct=mu,dedup_std_pct=sd,dedup_min=lo,dedup_max=hi,
                verdict=verdict,deltas=deltas)

print("clocks @start:",clocks())
print("=== REAL Mistral-7B prefill shapes (M=2048) + synthetic large-N — dedup floor, 6 reps ===")
shapes=[("k/v_proj",2048,4096,1024),
        ("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),
        ("gate/up_proj",2048,4096,14336),
        ("synthN=20480",2048,4096,20480),
        ("synthN=28672",2048,4096,28672)]
res=[study(*s) for s in shapes]
print("clocks @end:  ",clocks())
json.dump(res,open('stability_result.json','w'),indent=1)
print("wrote stability_result.json")
