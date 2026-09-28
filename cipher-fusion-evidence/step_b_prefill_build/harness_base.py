import ctypes, os, statistics, json, torch
torch.manual_seed(0); dev='cuda'
lib=ctypes.CDLL(os.path.abspath('libgemm_base.so'))
lib.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.run_gemm.restype=ctypes.c_int

def timed(fn, iters=100, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def run(name, M, K, N):
    W=(torch.randn(N,K,device=dev)*0.02).half()     # [N,K] (out,in) -- the Linear weight
    A=(torch.randn(M,K,device=dev)*0.1).half()       # [M,K]
    C=torch.empty(M,N,device=dev,dtype=torch.float16)
    def cutlass():
        r=lib.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,None)
        if r!=0: raise RuntimeError(f"run_gemm rc={r}")
    def cublas(): torch.nn.functional.linear(A,W)
    # correctness
    cutlass(); torch.cuda.synchronize()
    ref=torch.nn.functional.linear(A,W)
    rerr=(C.float()-ref.float()).abs().max().item()/ref.float().abs().max().item()
    t_cut=timed(cutlass); t_cub=timed(cublas)
    gflop=2*M*K*N/1e9
    tf_cut=gflop/t_cut/1e3; tf_cub=gflop/t_cub/1e3
    print(f"\n==== {name}  M={M} K={K} N={N} ====")
    print(f"  rel-err vs cuBLAS         : {rerr:.2e}")
    print(f"  cuBLAS  : {t_cub*1000:8.2f} us  ({tf_cub:6.1f} TFLOP/s)")
    print(f"  CUTLASS : {t_cut*1000:8.2f} us  ({tf_cut:6.1f} TFLOP/s)  = {tf_cut/tf_cub*100:.1f}% of cuBLAS")
    return dict(name=name,M=M,K=K,N=N,relerr=rerr,
                cublas_us=t_cub*1000,cutlass_us=t_cut*1000,
                cublas_tflops=tf_cub,cutlass_tflops=tf_cut,
                frac_of_cublas=tf_cut/tf_cub)

res=[run("down_proj",2048,14336,4096),
     run("v_proj",   2048, 4096,1024),
     run("gate_proj",2048, 4096,14336)]
json.dump(res,open('base_result.json','w'),indent=1)
print("\nwrote base_result.json")
