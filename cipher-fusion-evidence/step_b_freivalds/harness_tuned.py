#!/usr/bin/env python3
# Tuned-gemv lower bound: read achievable peak BW at 1200MHz, then time the coalesced row-reduction
# gemv for the two Freivalds terms that gate the best case (v=A@u, Cg=C@g). Report achieved BW and
# overhead vs cuBLAS/CUTLASS base. u=W^T@g (strided direction) reported as torch + hard byte-floor.
import ctypes, os, statistics, json, subprocess, torch
torch.manual_seed(0); dev='cuda'
gemm=ctypes.CDLL(os.path.abspath('libgemm_base.so'))
gemm.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; gemm.run_gemm.restype=ctypes.c_int
tk=ctypes.CDLL(os.path.abspath('libtuned_gemv.so'))
tk.rowgemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; tk.rowgemv.restype=ctypes.c_int
tk.read_bw.argtypes=[ctypes.c_void_p,ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p]; tk.read_bw.restype=ctypes.c_int

def clocks():
    return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader"],
                          capture_output=True,text=True).stdout.strip()
def med(fn, iters=300, warm=80):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)*1000.0  # us

# ---- peak read bandwidth at locked 1200MHz ----
NBIG=256*1024*1024  # 256M fp16 = 512MB
big=torch.randn(NBIG,device=dev).half(); out=torch.zeros(1024,device=dev,dtype=torch.float32)
def f_read(): tk.read_bw(big.data_ptr(),NBIG,out.data_ptr(),None)
t_read=med(f_read, iters=100, warm=30)
BW_PEAK=(NBIG*2)/(t_read*1e-6)/1e12
print(f"clocks @start: {clocks()}")
print(f"PEAK READ BW @1200MHz: {NBIG*2/1e6:.0f}MB in {t_read:.1f}us = {BW_PEAK:.2f} TB/s\n")

def study(name,M,K,N):
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    C=torch.empty(M,N,device=dev,dtype=torch.float16); g=torch.randn(N,device=dev).half()
    def gemm_so():
        if gemm.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)!=0: raise RuntimeError
    gemm_so(); torch.cuda.synchronize()
    u=torch.matmul(g,W).half()                         # [K]
    yv=torch.zeros(M,device=dev,dtype=torch.float32)   # v=A@u
    yc=torch.zeros(M,device=dev,dtype=torch.float32)   # Cg=C@g
    # correctness of tuned rowgemv (fp32 out) vs torch fp32
    tk.rowgemv(A.data_ptr(),u.data_ptr(),yv.data_ptr(),M,K,None)
    tk.rowgemv(C.data_ptr(),g.data_ptr(),yc.data_ptr(),M,N,None); torch.cuda.synchronize()
    v_ref=(A.float()@u.float()); c_ref=(C.float()@g.float())
    v_err=(yv-v_ref).abs().max().item()/v_ref.abs().max().item()
    c_err=(yc-c_ref).abs().max().item()/c_ref.abs().max().item()
    # timings
    t_gemm_cut=med(gemm_so); t_gemm_cub=med(lambda: torch.nn.functional.linear(A,W))
    t_v=med(lambda: tk.rowgemv(A.data_ptr(),u.data_ptr(),yv.data_ptr(),M,K,None))
    t_cg=med(lambda: tk.rowgemv(C.data_ptr(),g.data_ptr(),yc.data_ptr(),M,N,None))
    t_u_torch=med(lambda: torch.matmul(g,W))
    bw_v=(M*K*2)/(t_v*1e-6)/1e12; bw_cg=(M*N*2)/(t_cg*1e-6)/1e12
    # hard byte-floors (bytes/BW_PEAK) in us -- unattackable lower bound. BW_PEAK is TB/s.
    fl_v=(M*K*2)/(BW_PEAK*1e12)*1e6; fl_cg=(M*N*2)/(BW_PEAK*1e12)*1e6; fl_u=(N*K*2)/(BW_PEAK*1e12)*1e6
    def p(t,b): return 100.0*t/b
    o=dict(name=name,M=M,K=K,N=N,v_err=v_err,c_err=c_err,
           t_gemm_cutlass_us=t_gemm_cut,t_gemm_cublas_us=t_gemm_cub,
           t_v_tuned_us=t_v,t_cg_tuned_us=t_cg,t_u_torch_us=t_u_torch,
           bw_v_tuned=bw_v,bw_cg_tuned=bw_cg,
           floor_v_us=fl_v,floor_cg_us=fl_cg,floor_u_us=fl_u,
           # v-only (irreducible, best-case fused) overhead
           ov_vonly_tuned_cub=p(t_v,t_gemm_cub), ov_vonly_tuned_cut=p(t_v,t_gemm_cut),
           ov_vonly_floor_cub=p(fl_v,t_gemm_cub), ov_vonly_floor_cut=p(fl_v,t_gemm_cut),
           # Cg-only (overlap lower bound: Cg strictly post-GEMM)
           ov_cgonly_tuned_cub=p(t_cg,t_gemm_cub), ov_cgonly_floor_cub=p(fl_cg,t_gemm_cub),
           # u byte-floor (the W-read that must be paid if g not fixed/not mainloop-fused)
           ov_u_floor_cub=p(fl_u,t_gemm_cub))
    print(f"==== {name} M={M} K={K} N={N}  (rowgemv rel-err v={v_err:.1e} Cg={c_err:.1e}) ====")
    print(f"  GEMM CUTLASS {t_gemm_cut:7.1f}us  cuBLAS {t_gemm_cub:7.1f}us")
    print(f"  v=A@u  tuned {t_v:6.2f}us ({bw_v:4.2f}TB/s, {bw_v/BW_PEAK*100:3.0f}% peak)  floor {fl_v:5.2f}us")
    print(f"  Cg=C@g tuned {t_cg:6.2f}us ({bw_cg:4.2f}TB/s, {bw_cg/BW_PEAK*100:3.0f}% peak)  floor {fl_cg:5.2f}us")
    print(f"  u=W^Tg torch {t_u_torch:6.2f}us  | u byte-floor {fl_u:5.2f}us ({o['ov_u_floor_cub']:.2f}% of cuBLAS)")
    print(f"  IRREDUCIBLE v-only: tuned {o['ov_vonly_tuned_cub']:5.2f}%  floor {o['ov_vonly_floor_cub']:5.2f}%  (vs cuBLAS)   {'PASS' if o['ov_vonly_floor_cub']<3 else 'FAIL'}@floor / {'PASS' if o['ov_vonly_tuned_cub']<3 else 'FAIL'}@tuned")
    print(f"  overlap-LB Cg-only: tuned {o['ov_cgonly_tuned_cub']:5.2f}%  floor {o['ov_cgonly_floor_cub']:5.2f}%  (vs cuBLAS)\n")
    return o

SHAPES=[("k/v_proj",2048,4096,1024),("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),("gate/up_proj",2048,4096,14336)]
res=[study(*s) for s in SHAPES]
print("clocks @end:", clocks())
json.dump(dict(bw_peak_tbs=BW_PEAK, shapes=res),open('tuned_result.json','w'),indent=1)
print("wrote tuned_result.json")
