#!/usr/bin/env python3
# THE SPINE: detector per-GEMM overhead f(M) vs row-count M, real Mistral shapes, production clock.
# A decode GEMM at batch B is a GEMM at M=B -> decode-checksum-cost = f(M=B). One curve settles
# configs 1 (decode-only) and 2 (prefill) per-GEMM cost and exposes the headline: "+1.1% decode" is
# a B=1 number; under throughput batching the cost rises with B toward the prefill floor.
# Check = v=A@u + Cg=C@g (the fixed-g Freivalds/ABFT check; Cg is a free epilogue byproduct, so we
# report BOTH v+Cg (separate-kernel deployable) and v-only (Cg-fused) ). cuBLAS GEMM baseline.
import ctypes, os, statistics, json, subprocess, torch
torch.manual_seed(0); dev='cuda'
tk=ctypes.CDLL('/home/ubuntu/cipher-fusion-evidence/step_b_freivalds/libtuned_gemv.so')
tk.rowgemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; tk.rowgemv.restype=ctypes.c_int
tk.read_bw.argtypes=[ctypes.c_void_p,ctypes.c_long,ctypes.c_void_p,ctypes.c_void_p]; tk.read_bw.restype=ctypes.c_int
def clk(): return subprocess.run(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
def med(fn, iters=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)*1000.0
# peak read BW at this clock
NBIG=256*1024*1024; big=torch.randn(NBIG,device=dev).half(); ob=torch.zeros(1024,device=dev,dtype=torch.float32)
t_read=med(lambda: tk.read_bw(big.data_ptr(),NBIG,ob.data_ptr(),None), iters=100, warm=30)
BW=(NBIG*2)/(t_read*1e-6)/1e12
del big; torch.cuda.empty_cache()
print(f"clock under load: {clk()}   peak read BW = {BW:.2f} TB/s")
SHAPES=[("k/v",4096,1024),("q/o",4096,4096),("down",14336,4096),("gate/up",4096,14336)]
MS=[1,8,16,32,64,128,256,512,1024,2048]
print(f"\n{'shape':8s} {'M':>5s} {'GEMM us':>9s} | {'v us':>7s} {'Cg us':>7s} | "
      f"{'v+Cg meas%':>10s} {'v-only meas%':>12s} | {'v+Cg floor%':>11s} {'v-only floor%':>13s}")
res=[]
for name,K,N in SHAPES:
    W=(torch.randn(N,K,device=dev)*0.02).half()
    for M in MS:
        A=(torch.randn(M,K,device=dev)*0.1).half()
        g=torch.randn(N,device=dev).half(); u=torch.matmul(g,W).half()
        C=torch.nn.functional.linear(A,W)
        yv=torch.zeros(M,device=dev,dtype=torch.float32); yc=torch.zeros(M,device=dev,dtype=torch.float32)
        t_g=med(lambda: torch.nn.functional.linear(A,W))
        t_v=med(lambda: tk.rowgemv(A.data_ptr(),u.data_ptr(),yv.data_ptr(),M,K,None))
        t_c=med(lambda: tk.rowgemv(C.data_ptr(),g.data_ptr(),yc.data_ptr(),M,N,None))
        meas_vc=100*(t_v+t_c)/t_g; meas_v=100*t_v/t_g
        fl_v=(M*K*2)/(BW*1e12)*1e6; fl_c=(M*N*2)/(BW*1e12)*1e6
        floor_vc=100*(fl_v+fl_c)/t_g; floor_v=100*fl_v/t_g
        res.append(dict(shape=name,K=K,N=N,M=M,gemm_us=t_g,v_us=t_v,cg_us=t_c,
                        meas_vc_pct=meas_vc,meas_v_pct=meas_v,floor_vc_pct=floor_vc,floor_v_pct=floor_v))
        mark=lambda x:'<3' if x<3 else '  '
        print(f"{name:8s} {M:5d} {t_g:9.2f} | {t_v:7.2f} {t_c:7.2f} | "
              f"{meas_vc:9.2f}{mark(meas_vc)} {meas_v:11.2f}{mark(meas_v)} | {floor_vc:10.2f}{mark(floor_vc)} {floor_v:12.2f}{mark(floor_v)}")
    print()
json.dump(dict(bw_tbs=BW,clock=clk(),rows=res),open('f_of_M.json','w'),indent=1)
# crossover: for each shape, smallest M where v-only floor crosses 3%
print("CROSSOVER (smallest M where v-only floor >=3%, i.e. fully-fused best case fails):")
for name,K,N in SHAPES:
    rs=[r for r in res if r['shape']==name]
    cx=next((r['M'] for r in rs if r['floor_v_pct']>=3), None)
    cxm=next((r['M'] for r in rs if r['meas_v_pct']>=3), None)
    print(f"  {name:8s}: v-only floor>=3% at M={cx}   v-only measured>=3% at M={cxm}")
print("wrote f_of_M.json")
