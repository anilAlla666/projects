import ctypes, json, os, statistics, torch
torch.manual_seed(0); dev='cuda'

lib=ctypes.CDLL(os.path.abspath('libgemv.so'))
lib.launch_ref_m.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
lib.launch_ref_m.restype=None

def timed(fn, iters=100, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

sB=torch.cuda.Stream()
def measure(name, M, K, N):
    W=(torch.randn(N,K,device=dev)*0.02).half()      # [N,K] (out,in)
    A=(torch.randn(M,K,device=dev)*0.1).half()        # [M,K]
    wref=W.float().sum(0).contiguous()                # [K] precomputed once (static W)
    refout=torch.zeros(M,device=dev,dtype=torch.float32)
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def gemm():      torch.nn.functional.linear(A, W)            # cuBLAS baseline
    def ref():       lib.launch_ref_m(A.data_ptr(),wref.data_ptr(),refout.data_ptr(),M,K,None)  # fp16-read fp32-acc
    def outred():    torch.nn.functional.linear(A,W,out=C); C.float().sum(1)   # separate-op output reduction
    def overlap():
        torch.nn.functional.linear(A, W)                        # GEMM default stream
        with torch.cuda.stream(sB):
            lib.launch_ref_m(A.data_ptr(),wref.data_ptr(),refout.data_ptr(),M,K,sB.cuda_stream)
        torch.cuda.current_stream().wait_stream(sB)
    # correctness of ref kernel
    ref(); torch.cuda.synchronize()
    ref_true=(A.float()@wref)
    rerr=(refout-ref_true).abs().max().item()/ref_true.abs().max().item()
    t_g=timed(gemm); t_r=timed(ref); t_or=timed(outred)-t_g; t_ov=timed(overlap)
    gflop=2*M*K*N/1e9
    print(f"\n==== {name}  M={M} K={K} N={N} ====  (ref kernel rel-err {rerr:.1e})")
    print(f"  GEMM baseline cuBLAS              : {t_g*1000:8.2f} us   ({gflop/t_g/1e3:.0f} TFLOP/s)")
    print(f"  reference A.w_ref STANDALONE      : {t_r*1000:8.2f} us = {t_r/t_g*100:+.2f}% of GEMM  (reads A={M*K*2/1e6:.0f}MB fp16)")
    print(f"  reference OVERLAPPED (side stream): {(t_ov-t_g)*1000:8.2f} us = {(t_ov-t_g)/t_g*100:+.2f}% of GEMM   <-- marginal when overlapped")
    print(f"  output reduction SEPARATE-op      : {t_or*1000:8.2f} us = {t_or/t_g*100:+.2f}% of GEMM  (fused-epilogue ~free, cf M=1 +0.55%)")
    chksum_overlap = (t_ov-t_g)/t_g*100                    # ref overlapped + output fused(~0)
    chksum_naive   = (t_r + t_or)/t_g*100                  # both as separate ops (no fusion/overlap)
    print(f"  >>> CHECKSUM overlapped+fused approx = {chksum_overlap:+.2f}% ; naive separate-ops = {chksum_naive:+.2f}%")
    return dict(name=name.strip(),M=M,K=K,N=N,gemm_us=t_g*1000,tflops=gflop/t_g/1e3,
                ref_standalone_pct=t_r/t_g*100, ref_overlap_pct=chksum_overlap,
                outred_separate_pct=t_or/t_g*100, checksum_naive_pct=chksum_naive, ref_relerr=rerr)

res=[measure("down_proj",2048,14336,4096),
     measure("v_proj",2048,4096,1024),       # worst checksum:GEMM ratio (smallest N)
     measure("gate_proj",2048,4096,14336)]
json.dump(res, open('prefill_result.json','w'), indent=1)
print("\nwrote prefill_result.json")
