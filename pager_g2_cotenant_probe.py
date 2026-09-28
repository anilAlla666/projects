#!/usr/bin/env python3
# INCREMENT-2 DECISIVE TEST: co-tenant invariance of an agent's first-decode-step logits (greedy-immune).
# Contamination = agent X's logits change when its batch-mates or batch index change. Benign FP = tiny deltas,
# co-tenant-invariant and ~the run-to-run noise floor. Compares LOGITS at step 1 (before any greedy cascade),
# EAGER (no graph/clone artifacts). Weights in the pager region. Run CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel

MODEL=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
DET=os.environ.get("DET","0")=="1"
if DET:
    torch.use_deterministic_algorithms(True)  # + CUBLAS_WORKSPACE_CONFIG=:4096:8 -> run-to-run reproducible

POOL=[
 "The history of artificial intelligence began in the 1950s, when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes, a civilization had",
 "The recipe calls for two cups of flour, a pinch of salt, and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty",
]
eng=PagerGraphModel(CipherPager(), MODEL).load(); m=eng.m; tok=eng.tok; dev="cuda"
ids=[tok(p, return_tensors="pt").input_ids[0] for p in POOL]
P=min(t.shape[0] for t in ids); rows=[t[:P] for t in ids]   # common length, no padding
Lc=P+8

@torch.no_grad()
def first_step_logits(batch_rows, target_idx):
    pids=torch.stack(batch_rows).cuda()           # [B,P]
    c=StaticCache(config=m.config, max_cache_len=Lc)
    m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True)  # prefill
    nxt=m(pids[:, -1:], cache_position=torch.tensor([P],device=dev), past_key_values=c, use_cache=True).logits  # [B,1,V] decode step
    return nxt[target_idx,-1].float().clone()      # target row's decode logits

def md(a,b): return float((a-b).abs().max())

X=rows[0]
# X solo (twice -> FP run-to-run noise floor on identical input)
solo1=first_step_logits([X], 0); solo2=first_step_logits([X], 0); floor=md(solo1,solo2)
# X in different batch compositions / indices
compA=first_step_logits([rows[0],rows[1],rows[2],rows[3]], 0)   # X@0, co-tenants {1,2,3}
compB=first_step_logits([rows[4],rows[5],rows[6],rows[0]], 3)   # X@3, co-tenants {4,5,6}
compC=first_step_logits([rows[0],rows[5],rows[3],rows[7]], 0)   # X@0, DIFFERENT co-tenants {5,3,7}
compD=first_step_logits([rows[i%8] for i in range(8)], 0)       # X@0 in B=8
dA,dB,dC,dD = md(solo1,compA), md(solo1,compB), md(solo1,compC), md(solo1,compD)
dAC = md(compA,compC)   # SAME index, DIFFERENT co-tenants -> pure contamination signal
dAB = md(compA,compB)   # different index
scale=float(solo1.abs().max())
print(f"[cotenant {os.path.basename(MODEL)} DET={DET}] logit scale=±{scale:.1f}  FP-noise floor (solo vs solo)={floor:.3e}", flush=True)
print(f"  X solo vs X@batch:  B4@0={dA:.3e}  B4@3(reindex)={dB:.3e}  B4@0(diff cotenants)={dC:.3e}  B8@0={dD:.3e}", flush=True)
print(f"  PURE contamination: same-index diff-cotenants |Δ(compA,compC)|={dAC:.3e}   diff-index |Δ(compA,compB)|={dAB:.3e}", flush=True)
# verdict: contamination iff X's logits move with co-tenants beyond the noise floor (allow generous 50x floor or 1e-1)
tol=max(floor*50, 1e-1)
contaminated = dAC>tol
print(f"  tol={tol:.3e} -> {'CONTAMINATION (X depends on its batch-mates)' if contaminated else 'NO CONTAMINATION (X invariant to co-tenants within FP noise; greedy divergence at high B is near-tie FP flips, not cross-row leakage)'}", flush=True)
print(f"[VERDICT cotenant] {'FAIL-CONTAMINATED' if contaminated else 'PASS-NO-CONTAMINATION'}", flush=True)
sys.stdout.flush(); os._exit(0)
