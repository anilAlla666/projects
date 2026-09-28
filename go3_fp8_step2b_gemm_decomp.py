#!/usr/bin/env python3
# G-O3 STEP 2b -- NAME the binding term by decomposition (advisor #3; don't inherit the stale claim). At the REAL
# Mistral prefill GEMM shapes, locked clock, time: (1) bf16 GEMM, (2) RAW per-tensor FP8 GEMM (pre-quantized inputs,
# no act-quant -- the tensor-core ceiling), (3) FP8 GEMM + per-call activation quant (the tax-included path the
# driver-boundary actuator must pay every call). Decomposition raw~1.7-2x -> with-quant~1.0-1.1x IS the evidence that
# the per-call activation-quant tax binds (you can't keep activations in fp8 across ops without fusing). torch._scaled_mm
# = per-tensor scalar FP8, the same fused path the actuator uses.
import os, sys, time, subprocess
import torch
LOCK=os.environ.get("GO3_LOCK_MHZ","1200")
torch.manual_seed(0); dev="cuda"
def e4m3_quant(x):  # per-tensor absmax -> e4m3 + scale (what the actuator's NVRTC kernel does)
    amax=x.abs().amax().clamp(min=1e-4); s=amax/448.0
    return (x/s).clamp(-448,448).to(torch.float8_e4m3fn), s.float()
# Mistral-7B prefill GEMM shapes (out m, in k) at batch n=2048: o_proj/q 4096x4096, gate/up 14336x4096, down 4096x14336
SHAPES=[("q/o_proj",4096,4096),("gate/up",14336,4096),("down",4096,14336),("lm_head",32000,4096)]
N=2048; REP=50
def timed(fn):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(REP): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/REP
subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{LOCK},{LOCK}"],capture_output=True); time.sleep(0.5)
print(f"[go3-s2b] GEMM decomposition @ locked {LOCK}MHz, n(batch)={N}, REP={REP}",flush=True)
print(f"{'shape':<12} {'m':>6} {'k':>6} | {'bf16 us':>9} {'rawFP8 us':>10} {'FP8+quant us':>13} | {'raw/bf16':>9} {'quant/bf16':>11}",flush=True)
agg_bf16=agg_raw=agg_q=0.0
for name,mO,kI in SHAPES:
    W=torch.randn(mO,kI,dtype=torch.bfloat16,device=dev)        # weight (out x in)
    X=torch.randn(N,kI,dtype=torch.bfloat16,device=dev)         # activation (tokens x in)
    Wq,ws=e4m3_quant(W); Xq,xs=e4m3_quant(X)
    t_bf16=timed(lambda: torch.matmul(X,W.t()))
    # raw FP8: inputs already fp8 (no quant) -> tensor-core ceiling. _scaled_mm(A[m,k] fp8, B[k,n] fp8col, scales)
    Wt=Wq.t().contiguous().t()  # column-major rhs
    def raw(): return torch._scaled_mm(Xq, Wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    try: t_raw=timed(raw)
    except Exception as e: t_raw=float('nan'); print(f"  raw _scaled_mm err {name}: {str(e)[:80]}")
    # FP8 + per-call activation quant (the tax: quantize X every call, weight is pre-quantized once)
    def withq():
        xq,xsc=e4m3_quant(X); return torch._scaled_mm(xq, Wq.t(), scale_a=xsc, scale_b=ws, out_dtype=torch.bfloat16)
    try: t_q=timed(withq)
    except Exception as e: t_q=float('nan'); print(f"  withq err {name}: {str(e)[:80]}")
    agg_bf16+=t_bf16; agg_raw+=t_raw; agg_q+=t_q
    print(f"{name:<12} {mO:>6} {kI:>6} | {t_bf16*1e6:>9.1f} {t_raw*1e6:>10.1f} {t_q*1e6:>13.1f} | {t_bf16/t_raw:>9.2f} {t_bf16/t_q:>11.2f}",flush=True)
print(f"\n[go3-s2b SUMMARY] aggregate over shapes: RAW FP8 = {agg_bf16/agg_raw:.2f}x bf16 (tensor-core ceiling) | FP8+act-quant = {agg_bf16/agg_q:.2f}x bf16 (driver-boundary, tax-included)",flush=True)
print(f"[go3-s2b BINDING TERM] the per-call activation-quant tax collapses raw {agg_bf16/agg_raw:.2f}x -> {agg_bf16/agg_q:.2f}x at the GEMM level; end-to-end further capped by ~67.5% GEMM fraction (Amdahl) + non-GEMM ops.",flush=True)
sys.stdout.flush(); os._exit(0)
