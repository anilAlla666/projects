import ctypes, os, statistics, json, math, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libkernel_dedup.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
lib.get_fire.restype=ctypes.c_uint; lib.reset_fire.restype=ctypes.c_int
BLK_M,BLK_N=128,256; NSM=132; BIG=2_000_000_000   # R huge -> only seq=0 fires => machinery-only

def med(fn, iters=120):
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def study(name,M,K,N,reps=6):
    R=N//BLK_N; row_tiles=math.ceil(M/BLK_M); total=row_tiles*math.ceil(N/BLK_N); waves=total/NSM
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def ck():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    for _ in range(80): ck()  # warm
    dd=[]; mm=[]
    for _ in range(reps):
        lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=med(ck)
        lib.set_dedup(BIG); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); tm=med(ck)   # machinery only
        lib.set_dedup(R);   lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); t3=med(ck)   # dedup real
        mm.append((tm-t0)/t0*100); dd.append((t3-t0)/t0*100)
    mach=statistics.mean(mm); ded=statistics.mean(dd); corr=ded-mach
    cstd=statistics.pstdev([d-m for d,m in zip(dd,mm)])
    v='PASS' if corr<3 else 'FAIL'
    print(f"  {name:14s} N={N:6d} K={K:6d} waves={waves:5.2f}  machinery={mach:+5.2f}%  dedup_raw={ded:+5.2f}%  "
          f"-> TRUE_r_floor={corr:+5.2f}% ±{cstd:.2f}  {v}")
    return dict(name=name,M=M,K=K,N=N,waves=waves,machinery_pct=mach,dedup_raw_pct=ded,
                true_floor_pct=corr,true_floor_std=cstd,verdict=v)

print("=== machinery-subtracted TRUE dedup r-floor (zero-gate-cost impl), 6 reps ===")
print("    (machinery = atomic+threadfence+NamedBarrier on every tile; a real n_coord==0 gate pays 0)")
shapes=[("k/v_proj",2048,4096,1024),
        ("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),
        ("gate/up_proj",2048,4096,14336),
        ("synthN=20480",2048,4096,20480),
        ("synthN=28672",2048,4096,28672)]
res=[study(*s) for s in shapes]
json.dump(res,open('machinery_result.json','w'),indent=1)
print("wrote machinery_result.json")
