import ctypes, os, statistics, json, math, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libkernel_dedup.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
lib.get_fire.restype=ctypes.c_uint
lib.reset_fire.restype=ctypes.c_int

BLK_M, BLK_N = 128, 256

def timed(fn, iters=250, warm=60):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

# ---------- correctness: single CTA -> r exact, C unbroken (mode1) ----------
print("=== correctness (single-CTA validation, mode1) ===")
M,K,N=128,512,256
W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
wref=W.float().sum(0).contiguous()
rout=torch.zeros(128,device=dev,dtype=torch.float32)
C=torch.empty(M,N,device=dev,dtype=torch.float16)
assert lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),1)==0
assert lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)==0
torch.cuda.synchronize()
ref=torch.nn.functional.linear(A,W)
crel=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
r_true=(A.float()@wref)
rrel=(rout-r_true).abs().max().item()/r_true.abs().max().item()
print(f"  C rel-err (mode1): {crel:.2e}   r rel-err: {rrel:.2e}")

# ---------- cost: dedup (mode3) vs redundant (mode2) vs baseline (mode0) ----------
print("\n=== dedup floor: mode0 (off) / mode2 (redundant, every CTA) / mode3 (dedup, ~1/R tiles) ===")
def measure(name,M,K,N):
    R = N // BLK_N
    row_tiles = math.ceil(M/BLK_M); col_tiles = math.ceil(N/BLK_N)
    total_tiles = row_tiles*col_tiles
    expect_fire = total_tiles // R     # == row_tiles (one r per row-tile)
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def ck():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    # --- verify the gate fires exactly row_tiles times in ONE launch ---
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    lib.reset_fire(); ck(); torch.cuda.synchronize(); fires=lib.get_fire()
    # --- timings ---
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=timed(ck)
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),2); t2=timed(ck)
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); t3=timed(ck)
    d2=(t2-t0)/t0*100; d3=(t3-t0)/t0*100
    gate_ok = (fires==expect_fire)
    print(f"\n  ==== {name}  M={M} K={K} N={N}   R={R}  total_tiles={total_tiles}  waves~={total_tiles/132:.1f} ====")
    print(f"    gate fires (1 launch)    : {fires}   expect {expect_fire} (=row_tiles)   {'OK' if gate_ok else 'WRONG'}")
    print(f"    mode0 (r-off)            : {t0*1000:8.2f} us")
    print(f"    mode2 (redundant, x{R})  : {t2*1000:8.2f} us  -> {d2:+.2f}%   [recomputes r {R}x]")
    print(f"    mode3 (dedup, 1/{R})     : {t3*1000:8.2f} us  -> {d3:+.2f}%   [r once per row-tile]  {'<3% PASS' if d3<3 else '>=3% FAIL'}")
    return dict(name=name,M=M,K=K,N=N,R=R,total_tiles=total_tiles,waves=total_tiles/132,
                fires=int(fires),expect_fire=int(expect_fire),gate_ok=bool(gate_ok),
                mode0_us=t0*1000,mode2_us=t2*1000,mode3_us=t3*1000,
                redundant_pct=d2,dedup_pct=d3,pass3=bool(d3<3))

res=[measure("gate_proj",2048, 4096,14336),   # most waves -> most likely to pass: do first
     measure("down_proj",2048,14336, 4096),
     measure("v_proj",   2048, 4096, 1024)]
res.append(dict(validation=dict(C_relerr_mode1=crel,r_relerr=rrel)))
json.dump(res,open('dedup_result.json','w'),indent=1)
print("\nwrote dedup_result.json")
