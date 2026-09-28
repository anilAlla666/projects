#!/usr/bin/env python3
# Freivalds COVERAGE replay of Step A's harmful single-bit-flip set.
# Step A injected a single bit flip at output element C[m, chan] (delta = abs_delta, or NaN/inf).
#   - row-sum check (Step A): signal = delta            (caught if > T_rowsum)
#   - Freivalds check (this): signal = delta * g[chan]  (caught if > T_freivalds), g random per step
# T_freivalds = per-shape max clean |v - Cg| over rows (fp32 accum), over calibration g's
#   -> 0 clean false positives by construction (Step A discipline). SAME g used for T and signal.
# Compounded coverage over s independent g vectors: caught if ANY of s catches (or NaN/inf).
import json, statistics, torch
torch.manual_seed(1234); dev='cuda'

A1=json.load(open('/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a1_results.json'))
harm=[r for r in A1['results'] if r['harmful']]
# shape per ptype (K=in, N=out)
SHP={'q_proj':(4096,4096),'o_proj':(4096,4096),'k_proj':(4096,1024),'v_proj':(4096,1024),
     'gate_proj':(4096,14336),'up_proj':(4096,14336),'down_proj':(14336,4096),'lm_head':(4096,32000)}
M=2048
SMAX=4
# ---- calibrate T_freivalds per distinct (K,N) shape on a synthetic clean GEMM ----
shapes=sorted(set(SHP.values()))
G={}; T={}   # per shape: list of SMAX g-vectors (cpu), and fixed threshold
print("=== calibrating T_freivalds (max clean |v-Cg| over rows, fp32 accum), M=%d ===" % M)
for (K,N) in shapes:
    W=(torch.randn(N,K,device=dev)*0.02).half(); A=(torch.randn(M,K,device=dev)*0.1).half()
    C=torch.nn.functional.linear(A,W)                 # fp16 output (cuBLAS), as deployed
    gs=[torch.randn(N,device=dev) for _ in range(SMAX)]
    tmax=0.0
    for g in gs:
        u=(W.float().t()@g)                            # [K] fp32  (W^T@g)
        v=A.float()@u                                  # [M] fp32
        cg=C.float()@g                                 # [M] fp32 (reduces fp16 C)
        res=(v-cg).abs().max().item()
        tmax=max(tmax,res)
    G[(K,N)]=[g.cpu() for g in gs]; T[(K,N)]=tmax
    print(f"  K={K:5d} N={N:5d}  T_freivalds={tmax:.4e}")

# ---- replay harmful flips ----
print("\n=== Freivalds detection of Step A harmful class (200 harmful flips) ===")
def caught_by(g_idx, r):
    if r['naninf']: return True
    K,N=SHP[r['ptype']]; g=G[(K,N)][g_idx]
    chan=r['chan']; sig=abs(r['abs_delta']*g[chan].item())
    return sig>T[(K,N)]
naninf=sum(1 for r in harm if r['naninf']); finite=len(harm)-naninf
print(f"  total harmful={len(harm)}  naninf(caught by any check)={naninf}  finite={finite}")
# per-step (s=1) and compounded coverage
rows=[]
for s in [1,2,4]:
    caught=[any(caught_by(i,r) for i in range(s)) for r in harm]
    rate=100*sum(caught)/len(harm)
    # finite-only per-step margin: min finite signal / T  (s=1, g_idx 0)
    rows.append((s,rate,sum(caught)))
    print(f"  s={s}: compounded coverage = {sum(caught)}/{len(harm)} = {rate:.2f}%")
# margin on the finite (non-trivial) flips at s=1
fin=[r for r in harm if not r['naninf']]
fin_sig=[]
for r in fin:
    K,N=SHP[r['ptype']]; g=G[(K,N)][0]
    fin_sig.append((abs(r['abs_delta']*g[r['chan']].item()), T[(K,N)], r['ptype'], r['abs_delta'], g[r['chan']].item()))
print(f"\n  finite flips ({len(fin)}): per-step(s=1) signals vs T:")
for sig,t,pt,d,gc in fin_sig:
    print(f"    {pt:10s} delta={d:10.3f}  g[chan]={gc:+.4f}  signal={sig:10.4f}  T={t:.3e}  margin={sig/t:8.1f}x  {'CAUGHT' if sig>t else 'MISS'}")
# probability a finite flip is missed by one random g: P(|g[chan]| < T/delta)
import math
def miss_prob(delta, T):  # g~N(0,1): P(|g|<x)=erf(x/sqrt2)
    x=T/abs(delta); return math.erf(x/math.sqrt(2))
print("\n  per-flip miss prob (one g, g~N(0,1)) for the finite flips:")
for r in fin:
    K,N=SHP[r['ptype']]; mp=miss_prob(r['abs_delta'],T[(K,N)])
    print(f"    {r['ptype']:10s} delta={r['abs_delta']:10.3f}  P(miss|1g)={mp:.2e}  P(miss|s=2)={mp**2:.2e}")
json.dump(dict(T={f'{k[0]}x{k[1]}':v for k,v in T.items()},
               naninf=naninf,finite=finite,total=len(harm),
               coverage=[{'s':s,'rate':rate,'caught':c} for s,rate,c in rows]),
          open('coverage_result.json','w'),indent=1)
print("\nwrote coverage_result.json")
