#!/usr/bin/env python3
# Rigorous event-based concurrent-overlap test: can the check matvecs (u=W^Tg, v=A@u) HIDE under
# the compute-bound GEMM by running on a separate stream?  u,v depend only on W,A (not the GEMM
# output), so they CAN in principle co-execute with the GEMM. If they hide, the only exposed cost
# is Cg (strictly post-GEMM) -> down would flip to PASS (Cg-only ~1.4%). If the GEMM saturates all
# SMs, the gemvs are starved and wall ~ serial -> the serial number is the deployable one.
import ctypes, os, statistics, json, subprocess, torch
torch.manual_seed(0); dev='cuda'
gemm=ctypes.CDLL(os.path.abspath('libgemm_base.so'))
gemm.run_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]; gemm.run_gemm.restype=ctypes.c_int
tk=ctypes.CDLL(os.path.abspath('libtuned_gemv.so'))
tk.rowgemv.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; tk.rowgemv.restype=ctypes.c_int

def study(name,M,K,N,reps=8):
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    C=torch.empty(M,N,device=dev,dtype=torch.float16); g=torch.randn(N,device=dev).half()
    u=torch.matmul(g,W).half(); yv=torch.zeros(M,device=dev,dtype=torch.float32)
    s1=torch.cuda.Stream(); s2=torch.cuda.Stream()
    def G(st):
        if gemm.run_gemm(A.data_ptr(),W.data_ptr(),C.data_ptr(),M,N,K,st.cuda_stream)!=0: raise RuntimeError
    def CHK(st):
        torch.matmul(g,W)                       # u (torch, strided)
        tk.rowgemv(A.data_ptr(),u.data_ptr(),yv.data_ptr(),M,K,st.cuda_stream)  # v (tuned)
    def t_gemm_only():
        st=torch.cuda.Event(True);en=torch.cuda.Event(True)
        torch.cuda.synchronize(); st.record(s1)
        with torch.cuda.stream(s1): G(s1)
        en.record(s1); torch.cuda.synchronize(); return st.elapsed_time(en)*1000
    def t_serial():
        st=torch.cuda.Event(True);en=torch.cuda.Event(True)
        torch.cuda.synchronize(); st.record(s1)
        with torch.cuda.stream(s1): G(s1);
        with torch.cuda.stream(s1):
            torch.matmul(g,W); tk.rowgemv(A.data_ptr(),u.data_ptr(),yv.data_ptr(),M,K,s1.cuda_stream)
        en.record(s1); torch.cuda.synchronize(); return st.elapsed_time(en)*1000
    def t_concurrent():
        st=torch.cuda.Event(True);eg=torch.cuda.Event(True);ec=torch.cuda.Event(True)
        torch.cuda.synchronize(); st.record(s1)
        s2.wait_event(st)                       # both streams start together
        with torch.cuda.stream(s1): G(s1)
        with torch.cuda.stream(s2): CHK(s2)
        eg.record(s1); ec.record(s2); torch.cuda.synchronize()
        return max(st.elapsed_time(eg), st.elapsed_time(ec))*1000
    for _ in range(40): t_concurrent()          # warm
    go=statistics.median([t_gemm_only() for _ in range(reps)])
    se=statistics.median([t_serial()    for _ in range(reps)])
    co=statistics.median([t_concurrent()for _ in range(reps)])
    uv=se-go                                     # serial cost of u+v
    hidden=se-co                                 # how much of u+v got hidden by overlap
    frac_hidden=hidden/uv*100 if uv>0 else 0
    print(f"  {name:13s} gemm={go:7.1f}us  serial(+u+v)={se:7.1f}us  concurrent={co:7.1f}us  "
          f"u+v={uv:5.1f}us  hidden={hidden:+5.1f}us ({frac_hidden:+4.0f}%)  "
          f"{'OVERLAP HELPS' if frac_hidden>20 else 'NO OVERLAP (GEMM saturates SMs)'}")
    return dict(name=name,M=M,K=K,N=N,gemm_us=go,serial_us=se,concurrent_us=co,
                uv_serial_us=uv,hidden_us=hidden,frac_hidden_pct=frac_hidden)

print("clocks:",subprocess.run(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader"],
      capture_output=True,text=True).stdout.strip())
print("=== concurrent overlap: do u+v hide under the GEMM? ===")
SHAPES=[("k/v_proj",2048,4096,1024),("q/o_proj",2048,4096,4096),
        ("down_proj",2048,14336,4096),("gate/up_proj",2048,4096,14336)]
res=[study(*s) for s in SHAPES]
json.dump(res,open('overlap_result.json','w'),indent=1)
print("wrote overlap_result.json")
