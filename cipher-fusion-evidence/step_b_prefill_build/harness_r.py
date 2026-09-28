import ctypes, os, statistics, json, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libkernel_r.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
base=ctypes.CDLL(os.path.abspath('libgemm_base.so'))
base.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; base.run_gemm.restype=ctypes.c_int

def timed(fn, iters=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

# ---------- correctness: single CTA (M=128,N=256,K=512) -> r exact, C unbroken ----------
print("=== correctness (single-CTA validation) ===")
M,K,N=128,512,256
W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
wref=W.float().sum(0).contiguous()                 # [K]
rout=torch.zeros(128,device=dev,dtype=torch.float32)
C=torch.empty(M,N,device=dev,dtype=torch.float16)
assert lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),1)==0
assert lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)==0
torch.cuda.synchronize()
ref=torch.nn.functional.linear(A,W)
crel=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
r_true=(A.float()@wref)
rrel=(rout-r_true).abs().max().item()/r_true.abs().max().item()
print(f"  C rel-err vs torch (mode1)   : {crel:.2e}   (GEMM must be unbroken)")
print(f"  r rel-err vs A@wref          : {rrel:.2e}   (override live + correct)")
print(f"  r_out[:3]={rout[:3].tolist()}  r_true[:3]={r_true[:3].tolist()}")
# mode0: C still correct, rout untouched
rout.zero_(); assert lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0)==0
lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None); torch.cuda.synchronize()
crel0=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
print(f"  C rel-err vs torch (mode0)   : {crel0:.2e}   rout sum (should be 0): {rout.sum().item()}")

# ---------- cost: r-only delta on real shapes ----------
print("\n=== r-only cost (mode1 vs mode0 on SAME binary; + mode0 vs stock baseline) ===")
def measure(name,M,K,N):
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def ck():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    def stock():
        if base.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=timed(ck)
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),1); t1=timed(ck)
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),2); t2=timed(ck)
    tb=timed(stock)
    d_r=(t1-t0)/t0*100; d_r2=(t2-t0)/t0*100; d_sub=(t0-tb)/tb*100
    print(f"\n  ==== {name}  M={M} K={K} N={N} ====")
    print(f"    mode0 (r-off)            : {t0*1000:8.2f} us")
    print(f"    mode1 (r naive scalar)   : {t1*1000:8.2f} us  -> {d_r:+.2f}%")
    print(f"    mode2 (r efficient vec)  : {t2*1000:8.2f} us  -> {d_r2:+.2f}%   [COST FLOOR, all 256 thr, vectorized]")
    print(f"    stock baseline           : {tb*1000:8.2f} us  (subclass regression {d_sub:+.2f}%)")
    return dict(name=name,M=M,K=K,N=N,mode0_us=t0*1000,mode1_us=t1*1000,mode2_us=t2*1000,stock_us=tb*1000,
                r_delta_naive_pct=d_r,r_delta_floor_pct=d_r2,subclass_regression_pct=d_sub)
res=[measure("down_proj",2048,14336,4096),
     measure("v_proj",   2048, 4096,1024),
     measure("gate_proj",2048, 4096,14336)]
res.append(dict(validation=dict(C_relerr_mode1=crel,r_relerr=rrel,C_relerr_mode0=crel0)))
json.dump(res,open('r_result.json','w'),indent=1)
print("\nwrote r_result.json")
