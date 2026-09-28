# Track A keystone: fresh @1980 measurement of the fused-checksum WGMMA GEMM vs cuBLAS.
# Decomposes deployable cost: parity_gap (custom GEMM vs cuBLAS) x checksum_overhead (fused vs custom base) = total vs cuBLAS.
# Uses the existing dedup kernel (libkernel_dedup.so): mode0=GEMM-only, mode3=dedup fused row-sum checksum.
import ctypes, os, statistics, json, math, torch
torch.manual_seed(0); dev='cuda'
SO = os.path.abspath('/home/ubuntu/cipher-fusion-evidence/step_b_prefill_build/libkernel_dedup.so')
lib=ctypes.CDLL(SO)
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

def measure(name,M,K,N):
    R = N // BLK_N
    row_tiles = math.ceil(M/BLK_M); col_tiles = math.ceil(N/BLK_N)
    total_tiles = row_tiles*col_tiles
    expect_fire = total_tiles // R
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def cutlass():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    def cublas(): torch.nn.functional.linear(A,W)
    # correctness of custom GEMM vs cuBLAS
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); cutlass(); torch.cuda.synchronize()
    ref=torch.nn.functional.linear(A,W)
    crel=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
    # gate fires exactly row_tiles times
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
    lib.reset_fire(); cutlass(); torch.cuda.synchronize(); fires=lib.get_fire()
    # timings: cuBLAS, mode0 (custom GEMM checksum-off), mode3 (fused dedup checksum)
    t_cub=timed(cublas)
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); t0=timed(cutlass)
    lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3); t3=timed(cutlass)
    parity_gap = (t0-t_cub)/t_cub*100        # custom GEMM vs cuBLAS
    checksum   = (t3-t0)/t0*100              # fused checksum vs custom base
    total_vs_cublas = (t3-t_cub)/t_cub*100   # THE deployable per-step number
    gflop=2*M*K*N/1e9
    print(f"\n==== {name}  M={M} K={K} N={N}  R={R} waves~={total_tiles/132:.2f}  fires={fires}/{expect_fire} ====")
    print(f"  C rel-err vs cuBLAS  : {crel:.2e}")
    print(f"  cuBLAS               : {t_cub*1000:8.2f} us  ({gflop/t_cub/1e3:6.1f} TF)")
    print(f"  custom GEMM (mode0)  : {t0*1000:8.2f} us  ({gflop/t0/1e3:6.1f} TF)  parity_gap={parity_gap:+.1f}% (custom/{t0/t_cub*100:.0f}% of cuBLAS)")
    print(f"  fused checksum(mode3): {t3*1000:8.2f} us  checksum={checksum:+.1f}% vs custom base")
    print(f"  >>> TOTAL vs cuBLAS  : {total_vs_cublas:+.1f}%   {'<3% PASS' if total_vs_cublas<3 else '>=3% FAIL'}")
    return dict(name=name,M=M,K=K,N=N,R=R,waves=total_tiles/132,fires=int(fires),expect_fire=int(expect_fire),
                C_relerr=crel, cublas_us=t_cub*1000, mode0_us=t0*1000, mode3_us=t3*1000,
                parity_gap_pct=parity_gap, checksum_pct=checksum, total_vs_cublas_pct=total_vs_cublas,
                pass3=bool(total_vs_cublas<3))

shapes=[("k/v_proj",2048,4096,1024),
        ("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),
        ("gate/up_proj",2048,4096,14336)]
res=[measure(*s) for s in shapes]
import subprocess
clk=subprocess.run(["nvidia-smi","--query-gpu=clocks.gr","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
out=dict(clock=clk, shapes=res)
json.dump(out,open('/home/ubuntu/cipher-fusion-evidence/ra_keystone/trackA/base_vs_cublas_1980.json','w'),indent=1)
print(f"\nclock during run: {clk}")
print("wrote base_vs_cublas_1980.json")
