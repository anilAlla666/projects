# Track A: drift-canceling INTERLEAVED measure of cuBLAS / custom-GEMM(mode0) / fused-checksum(mode3).
# Per rep, time all three back-to-back so any clock drift hits all three equally; median over reps.
# Also samples the in-loop clock so the report states the real measurement clock (not the cooldown transient).
import ctypes, os, statistics, json, math, subprocess, torch
torch.manual_seed(0); dev='cuda'
SO='/home/ubuntu/cipher-fusion-evidence/step_b_prefill_build/libkernel_dedup.so'
lib=ctypes.CDLL(SO)
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; lib.run_gemm.restype=ctypes.c_int
lib.set_ck_ptrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]; lib.set_ck_ptrs.restype=ctypes.c_int
lib.set_dedup.argtypes=[ctypes.c_int]; lib.set_dedup.restype=ctypes.c_int
BLK_N=256
def clk(): return subprocess.run(["nvidia-smi","--query-gpu=clocks.gr","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()

def ev(fn):
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record(); fn(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)

def measure(name,M,K,N,reps=200,warm=60):
    R=N//BLK_N
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    wref=W.float().sum(0).contiguous(); rout=torch.zeros(128,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def cublas(): torch.nn.functional.linear(A,W)
    def cutlass():
        if lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    # correctness
    lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0); cutlass(); torch.cuda.synchronize()
    crel=(C.float()-torch.nn.functional.linear(A,W).float()).abs().max().item()/torch.nn.functional.linear(A,W).float().abs().max().item()
    for _ in range(warm): cublas(); cutlass()
    torch.cuda.synchronize()
    tb=[]; t0=[]; t3=[]; clocks=[]
    for i in range(reps):
        lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),0)
        tb.append(ev(cublas))
        t0.append(ev(cutlass))
        lib.set_dedup(R); lib.set_ck_ptrs(wref.data_ptr(),rout.data_ptr(),3)
        t3.append(ev(cutlass))
        if i%40==0: clocks.append(int(clk()))
    mb=statistics.median(tb); m0=statistics.median(t0); m3=statistics.median(t3)
    parity=(m0-mb)/mb*100; checksum=(m3-m0)/m0*100; total=(m3-mb)/mb*100
    print(f"{name:14s} N={N:5d} waves~{math.ceil(M/128)*math.ceil(N/BLK_N)/132:.2f}  clk{min(clocks)}-{max(clocks)}  "
          f"cuBLAS={mb*1000:7.1f}us custom={m0*1000:7.1f}us fused={m3*1000:7.1f}us | "
          f"parity={parity:+5.1f}% checksum={checksum:+5.1f}% TOTAL={total:+5.1f}% {'PASS' if total<3 else 'FAIL'}  Crel={crel:.0e}")
    return dict(name=name,M=M,K=K,N=N,reps=reps,clk_lo=min(clocks),clk_hi=max(clocks),
                cublas_us=mb*1000,custom_us=m0*1000,fused_us=m3*1000,C_relerr=crel,
                parity_gap_pct=parity,checksum_pct=checksum,total_vs_cublas_pct=total,pass3=bool(total<3))

shapes=[("k/v_proj",2048,4096,1024),("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),("gate/up_proj",2048,4096,14336)]
res=[measure(*s) for s in shapes]
out=dict(method="interleaved drift-canceling, median of 200 reps, clock locked 1980 (runs lower under load)",
         clock_idle=clk(), shapes=res)
json.dump(out,open('/home/ubuntu/cipher-fusion-evidence/ra_keystone/trackA/interleaved_1980.json','w'),indent=1)
print("wrote interleaved_1980.json")
