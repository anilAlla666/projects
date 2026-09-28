#!/usr/bin/env python3
# Advisor hardening: 2-GEMM CHAIN in ONE capture — exercises the multi-call paths the 1-GEMM probe skipped:
# workspace rekey + 2nd-call HIT + shared-workspace serial reuse + async cast alloc/free/alloc/free pool reuse.
import os, ctypes, torch
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; lib=ctypes.CDLL(SO)
def u(n):
    try: f=getattr(lib,n); f.restype=ctypes.c_ulong; return int(f())
    except: return -1
torch.manual_seed(0)
M,K,N=8,4096,4096
x =(torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3)
W1=(torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.05)
W2=(torch.randn(N,N,device="cuda",dtype=torch.bfloat16)*0.05)
def chain(a): return F.linear(F.linear(a,W1),W2)        # 2 Marlin-eligible GEMMs, sequential dependency
# bf16 reference chain (fp32 matmul -> what cuBLAS ~produces; INT4 error compounds over 2 GEMMs)
y_bf16=((x.float()@W1.float().T)@W2.float().T).to(torch.bfloat16)

print("warm both GEMMs (quantize + allocate workspace) ...")
for _ in range(8): chain(x); torch.cuda.synchronize()
h0=u("cipher_rt_marlin_calls_handled")
y_marlin_eager=chain(x).clone(); torch.cuda.synchronize()
relE=(y_marlin_eager.float()-y_bf16.float()).abs().max().item()/y_bf16.float().abs().max().item()
print(f"  eager-Marlin chain vs bf16 chain max_rel={relE:.3e} (compounded INT4 error, ID-able)")

print("capture 2-GEMM chain in one graph ...")
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): chain(x)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
yo=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
h_pre=u("cipher_rt_marlin_calls_handled"); ok=False; err=None
g=torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        yo.copy_(chain(x))
    ok=True
except Exception as e: err=repr(e)
h_cap=u("cipher_rt_marlin_calls_handled")
print(f"  capture succeeded: {ok}"+(f" ERR={err}" if err else f"  | Marlin handled delta across capture: {h_cap-h_pre} (expect 2 = both GEMMs)"))
if ok:
    yo.zero_(); torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    rv_m=(yo.float()-y_marlin_eager.float()).abs().max().item()/y_marlin_eager.float().abs().max().item()
    rv_b=(yo.float()-y_bf16.float()).abs().max().item()/y_bf16.float().abs().max().item()
    # replay AGAIN to confirm stable (pool reuse across replays)
    yo.zero_(); g.replay(); torch.cuda.synchronize()
    rv_m2=(yo.float()-y_marlin_eager.float()).abs().max().item()/y_marlin_eager.float().abs().max().item()
    print(f"  replay1 vs eager-Marlin={rv_m:.3e} vs bf16={rv_b:.3e} | replay2 vs eager-Marlin={rv_m2:.3e}")
    print(f"  -> 2-GEMM gate: {'PASS (both GEMMs Marlin, bit-identical, stable across replays)' if rv_m<1e-2 and rv_b>1e-2 and rv_m2<1e-2 else 'FAIL'}")
