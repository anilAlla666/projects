#!/usr/bin/env python3
# THROWAWAY PROBE (no .so change): does Marlin's GOT-patch substitution survive being INSIDE a torch.cuda.graph
# capture (mechanism B = capture-time substitution)? Discriminator (advisor): (a) did Marlin fire DURING the
# capture block (handled-counter delta), (b) is the replayed output == eager-Marlin-INT4 AND != cuBLAS-bf16
# (positive ID that Marlin specifically is the captured node, not cuBLAS faithfully replaying).
import os, ctypes, torch, traceback
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
lib=ctypes.CDLL(SO)
def u(n):
    try: f=getattr(lib,n); f.restype=ctypes.c_ulong; return int(f())
    except: return -1
def H(): return dict(handled=u("cipher_rt_marlin_calls_handled"), bf16=u("cipher_rt_marlin_calls_bf16_substituted"),
                     total=u("cipher_rt_matmul_calls_total"), mm_handled=u("cipher_rt_matmul_calls_handled"),
                     skipped=u("cipher_rt_marlin_calls_skipped"))

torch.manual_seed(0)
M,K,N=8,4096,4096
x=(torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.3)
W=(torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.05)   # Marlin-eligible: M=8<=64, N,K=4096>=1024, %128/64==0
y_bf16=(x.float()@W.float().T).to(torch.bfloat16)              # clean bf16 reference (what cuBLAS ~produces)

print("marlin_active=", u("cipher_rt_marlin_is_active"), " (need 1; CIPHER_MARLIN=on)")
print("warm (eager) — drive Marlin to quantize + substitute:")
h0=H()
for i in range(8):
    y=F.linear(x,W); torch.cuda.synchronize()
h_warm=H()
y_marlin_eager=F.linear(x,W).clone(); torch.cuda.synchronize()
print(f"  handled {h0['handled']}->{h_warm['handled']} (Marlin fired eagerly: {h_warm['handled']>h0['handled']})")
relM=(y_marlin_eager.float()-y_bf16.float()).abs().max().item()/y_bf16.float().abs().max().item()
print(f"  eager-Marlin vs bf16-ref max_rel={relM:.3e} (INT4 lossy -> should be measurably >0, makes the kernel ID-able)")

print("\nCAPTURE: torch.cuda.graph of F.linear(x,W) with Marlin armed —")
h_pre=H()
g=torch.cuda.CUDAGraph(); captured=False; err=None
try:
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): F.linear(x,W)        # capture-pool warmup on side stream
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    yo=torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
    with torch.cuda.graph(g):
        yo.copy_(F.linear(x,W))
    captured=True
except Exception as e:
    err=repr(e); traceback.print_exc()
h_cap=H()
print(f"  capture succeeded: {captured}" + (f"  ERROR={err}" if err else ""))
print(f"  (a) Marlin handled delta ACROSS capture block: {h_cap['handled']-h_pre['handled']}  bf16sub delta: {h_cap['bf16']-h_pre['bf16']}  skipped delta: {h_cap['skipped']-h_pre['skipped']}")
print(f"      -> Marlin fired DURING capture: {h_cap['handled']>h_pre['handled']}  (if 0, it PASSED THROUGH to cuBLAS = bypass)")

if captured:
    yo.zero_(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    rel_vs_marlin=(yo.float()-y_marlin_eager.float()).abs().max().item()/y_marlin_eager.float().abs().max().item()
    rel_vs_bf16 =(yo.float()-y_bf16.float()).abs().max().item()/y_bf16.float().abs().max().item()
    print(f"  (b) replay(zeroed) output: vs eager-Marlin rel={rel_vs_marlin:.3e} | vs bf16 rel={rel_vs_bf16:.3e}")
    is_marlin = rel_vs_marlin<1e-2 and rel_vs_bf16>1e-2
    is_bf16   = rel_vs_bf16<1e-2 and rel_vs_marlin>1e-2
    print(f"      -> captured node is: {'MARLIN-INT4 (gate works via mechanism B!)' if is_marlin else ('cuBLAS-bf16 (Marlin bypassed at capture)' if is_bf16 else 'AMBIGUOUS')}")
