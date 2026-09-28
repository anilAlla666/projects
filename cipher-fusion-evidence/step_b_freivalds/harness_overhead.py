#!/usr/bin/env python3
# STEP B FREIVALDS -- overhead measurement.
# Baseline GEMM = real CUTLASS sm90 WGMMA kernel (libgemm_base.so) AND cuBLAS (torch.linear).
# Check matvecs measured as fp16-read gemvs (deployable byte traffic; fp32 accum is for the
# threshold/coverage compute, NOT the bandwidth timing -- upcasting inputs would 2x the bytes).
#   verify C[M,N]=A[M,K]@W[N,K]^T via random g[N]:   u=W^T@g [K] ; v=A@u [M] ; Cg=C@g [M] ; v?=Cg
# Layouts composed from primitives:
#   v-only            = best-case fully-fused (Cg free in epilogue, u free/amortized)  -> LOWER bound
#   spec-(i)          = custom-GEMM + u + v   (Cg free epilogue; cuBLAS can't host it, Finding 1)
#   all-separate (UB) = GEMM + u + v + Cg     (all three as standalone gemvs)            -> UPPER bound
#   overlap           = GEMM(stream1) || (u+v)(stream2)  -- can u,v hide under the compute-bound GEMM?
import ctypes, os, statistics, json, subprocess, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libgemm_base.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.run_gemm.restype=ctypes.c_int

def clocks():
    o=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw,temperature.gpu","--format=csv,noheader"],
                     capture_output=True,text=True).stdout.strip()
    return o

def med(fn, iters=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)*1000.0   # us

def study(name,M,K,N):
    W=(torch.randn(N,K,device=dev)*0.02).half()   # Linear weight [out=N, in=K]
    A=(torch.randn(M,K,device=dev)*0.1).half()
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    g=torch.randn(N,device=dev).half()            # random probe (fp16 for byte-faithful timing)
    # build a correct C first (used by Cg and by correctness checks)
    def gemm_so(stream=None):
        rc=lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,
                        (stream.cuda_stream if stream is not None else None))
        if rc!=0: raise RuntimeError(f"run_gemm rc={rc}")
    gemm_so(); torch.cuda.synchronize()
    # correctness of the CUTLASS GEMM vs cuBLAS
    ref=torch.nn.functional.linear(A,W)
    gemm_relerr=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()

    # ---- primitives ----
    u=torch.matmul(g, W)            # [N]@[N,K] -> u[K] = W^T@g ; reads all of W
    def f_u():  return torch.matmul(g, W)
    def f_v():  return torch.mv(A, u)      # reads all of A
    def f_cg(): return torch.mv(C, g)      # reads all of C
    def f_gemm_cublas(): return torch.nn.functional.linear(A,W)

    t_gemm_cut = med(gemm_so)
    t_gemm_cub = med(f_gemm_cublas)
    t_u = med(f_u); t_v = med(f_v); t_cg = med(f_cg)
    # achieved bandwidth of each gemv (fp16 bytes moved / time)
    bw_u = (N*K*2)/(t_u*1e-6)/1e12; bw_v=(M*K*2)/(t_v*1e-6)/1e12; bw_cg=(M*N*2)/(t_cg*1e-6)/1e12

    # ---- serial composite layouts (default stream, back-to-back) ----
    def serial_all():  gemm_so(); torch.matmul(g,W); torch.mv(A,u); torch.mv(C,g)
    def serial_uv():   gemm_so(); torch.matmul(g,W); torch.mv(A,u)
    def serial_v():    gemm_so(); torch.mv(A,u)
    def serial_cg():   gemm_so(); torch.mv(C,g)
    t_all = med(serial_all); t_uv = med(serial_uv); t_vonly = med(serial_v); t_cgonly = med(serial_cg)

    # ---- concurrent overlap: GEMM on s1, (u then v) on s2 ----
    s1=torch.cuda.Stream(); s2=torch.cuda.Stream()
    def overlap_uv():
        # GEMM on s1; u,v on s2 (independent of GEMM output). measure wall until both done.
        with torch.cuda.stream(s1):
            gemm_so(stream=s1)
        with torch.cuda.stream(s2):
            uu=torch.matmul(g,W); torch.mv(A,uu)
        s1.synchronize(); s2.synchronize()
    # for fair wall timing, time on default stream waiting for both
    def overlap_timed():
        ev0=torch.cuda.Event(True); ev1=torch.cuda.Event(True)
        torch.cuda.synchronize()
        ev0.record()
        with torch.cuda.stream(s1): gemm_so(stream=s1)
        with torch.cuda.stream(s2):
            uu=torch.matmul(g,W); torch.mv(A,uu)
        ev1.record(s1)  # not enough; sync both then measure host? use synchronize-based
        s1.synchronize(); s2.synchronize()
    # robust wall measure: median of host-timed full-sync regions
    import time
    def wall(fn, iters=120, warm=40):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); ts=[]
        for _ in range(iters):
            torch.cuda.synchronize(); t0=time.perf_counter()
            fn(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e6)
        return statistics.median(ts)
    w_overlap = wall(overlap_uv)
    # serial baseline measured the same host-timed way (apples-to-apples vs overlap)
    def serial_uv_sync(): gemm_so(); torch.matmul(g,W); torch.mv(A,u); torch.cuda.synchronize()
    w_serial_uv = wall(lambda: (gemm_so(), torch.matmul(g,W), torch.mv(A,u)))
    w_gemm_only = wall(lambda: gemm_so())

    # ---- overhead numbers (vs CUTLASS base and vs cuBLAS base) ----
    def ov(t_comp, t_base): return 100.0*(t_comp - t_base)/t_base
    out=dict(name=name,M=M,K=K,N=N,gemm_relerr=gemm_relerr,
             t_gemm_cutlass_us=t_gemm_cut, t_gemm_cublas_us=t_gemm_cub,
             t_u_us=t_u,t_v_us=t_v,t_cg_us=t_cg, bw_u=bw_u,bw_v=bw_v,bw_cg=bw_cg,
             # vs CUTLASS base (spec's stated baseline; OPTIMISTIC -- slow denominator)
             ov_all_cut=ov(t_all,t_gemm_cut), ov_uv_cut=ov(t_uv,t_gemm_cut),
             ov_vonly_cut=ov(t_vonly,t_gemm_cut), ov_cgonly_cut=ov(t_cgonly,t_gemm_cut),
             # vs cuBLAS base (deployable reality; CONSERVATIVE)
             ov_all_cub=100.0*(t_u+t_v+t_cg)/t_gemm_cub, ov_uv_cub=100.0*(t_u+t_v)/t_gemm_cub,
             ov_vonly_cub=100.0*t_v/t_gemm_cub, ov_cgonly_cub=100.0*t_cg/t_gemm_cub,
             # overlap
             wall_gemm_only_us=w_gemm_only, wall_serial_uv_us=w_serial_uv, wall_overlap_uv_us=w_overlap,
             overlap_benefit_us=w_serial_uv-w_overlap,
             )
    print(f"\n==== {name}  M={M} K={K} N={N}  (gemm rel-err {gemm_relerr:.1e}) ====")
    print(f"  GEMM   CUTLASS {t_gemm_cut:8.2f}us   cuBLAS {t_gemm_cub:8.2f}us  ({t_gemm_cut/t_gemm_cub*100:.0f}% of cuBLAS)")
    print(f"  u=W^Tg {t_u:7.2f}us ({bw_u:4.2f}TB/s)  v=Au {t_v:7.2f}us ({bw_v:4.2f}TB/s)  Cg=Cg {t_cg:7.2f}us ({bw_cg:4.2f}TB/s)")
    print(f"  OVERHEAD vs cuBLAS base : v-only {out['ov_vonly_cub']:5.2f}%  | u+v(spec-i) {out['ov_uv_cub']:5.2f}%  | u+v+Cg(UB) {out['ov_all_cub']:5.2f}%   [Cg-only {out['ov_cgonly_cub']:5.2f}%]")
    print(f"  OVERHEAD vs CUTLASS base: v-only {out['ov_vonly_cut']:5.2f}%  | u+v(spec-i) {out['ov_uv_cut']:5.2f}%  | u+v+Cg(UB) {out['ov_all_cut']:5.2f}%")
    print(f"  OVERLAP: gemm {w_gemm_only:7.1f}us  serial(gemm+u+v) {w_serial_uv:7.1f}us  concurrent {w_overlap:7.1f}us  benefit {out['overlap_benefit_us']:+.1f}us")
    return out

print("clocks @start:", clocks())
SHAPES=[("k/v_proj",2048,4096,1024),
        ("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),
        ("gate/up_proj",2048,4096,14336)]
res=[study(*s) for s in SHAPES]
print("\nclocks @end:  ", clocks())
json.dump(res,open('overhead_result.json','w'),indent=1)
print("wrote overhead_result.json")
