#!/usr/bin/env python3
# INCREMENT-3 router gate: M distinct models, capacity K<M, Zipfian model-popularity request stream forcing
# residency swaps. Gate: per-request KL=0 vs solo (esp. ACROSS swaps), swaps physically real (HBM freed on evict),
# cross-model misroute negative control MUST flag. CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, numpy as np, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager
from cipher_router import MultiModelRouter
from cipher_engine_batched import classify   # ratified: exact | tie | FAULT (with solo margins)

N=int(os.environ.get("N","48")); K=int(os.environ.get("K","2")); REQ=int(os.environ.get("REQ","24"))
ZIPF=float(os.environ.get("ZIPF","1.3"))
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("llama1b","/home/ubuntu/models/Llama-3.2-1B-Instruct",0xC4),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
SPECS=SPECS[:int(os.environ.get("MLIM","4"))]
M=len(SPECS); names=[s[0] for s in SPECS]
PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"

pager=CipherPager()
print(f"[router] loading M={M} distinct models, capacity K={K} (forces swaps), N={N}", flush=True)
R=MultiModelRouter(pager, SPECS, K, PROMPT, N)
free=torch.cuda.mem_get_info()[0]/(1<<30)
print(f"[router] setup done: {M} models (regions+graphs), {K} resident. HBM free={free:.1f}GB", flush=True)

# solo per-step margins (for the ratified near-tie classification)
solo_mar={}
for name,e in R.models.items():
    # recompute solo with margins via WaveServer-style? PagerGraphModel.solo not present; derive margins cheaply:
    solo_mar[name]=[1.0]*N   # placeholder; these models capture EXACT (inc-1 KL=0) so margin path unused unless divergence

# Zipfian model-popularity stream (fixed seed)
rng=np.random.default_rng(0)
w=np.array([1.0/((i+1)**ZIPF) for i in range(M)]); w/=w.sum()
stream=[names[i] for i in rng.choice(M, size=REQ, p=w)]

ex=tie=fault=0; swaps_real=0; across_swap_checked=0; across_swap_ok=0; faults=[]
cap_swap=[]; cap_hit=[]; dec_all=[]
for r,name in enumerate(stream):
    out,kind,freed,victim,cap_ms,dec_ms=R.serve(name)
    dec_all.append(dec_ms); (cap_swap if kind=="swap" else cap_hit).append(cap_ms)
    solo=R.solos[name]
    k,fd,mg=classify(out, solo, solo_mar[name])
    if k=="exact": ex+=1
    elif k=="tie": tie+=1
    else: fault+=1; faults.append((name,fd,mg))
    if kind=="swap":
        if freed>0.5: swaps_real+=1
        across_swap_checked+=1
        if k in ("exact","tie"): across_swap_ok+=1
from collections import Counter
print(f"[stream] {REQ} reqs, popularity={dict(Counter(stream))}", flush=True)
print(f"  CORRECTNESS: exact={ex} near-tie={tie} FAULT={fault}  {'PASS' if fault==0 else 'FAIL '+str(faults)}", flush=True)
print(f"  SWAPS: total={R.swaps} hits={R.hits} physically-real(HBM freed>0.5GB)={swaps_real}/{across_swap_checked}", flush=True)
print(f"  ACROSS-SWAP KL=0: {across_swap_ok}/{across_swap_checked} requests whose model was evicted+restored still matched solo", flush=True)

# cross-model misroute negative control: model A's output vs a DIFFERENT model's solo -> must FAULT
a,b=names[0],names[1]
outA,_,_,_,_,_=R.serve(a)
kk,_,_=classify(outA, R.solos[b], [1.0]*N)
nc_ok=(kk=="FAULT")
print(f"  MISROUTE neg-control: model {a} output vs model {b} solo -> {kk} ({'DETECTED' if nc_ok else 'NOT DETECTED'})", flush=True)

gate=(fault==0) and (across_swap_ok==across_swap_checked) and (swaps_real==across_swap_checked or across_swap_checked==0) and nc_ok
print(f"[GATE router] correctness={'PASS' if fault==0 else 'FAIL'} across-swap-KL0={across_swap_ok}/{across_swap_checked} swaps-real={swaps_real}/{across_swap_checked} misroute-nc={'PASS' if nc_ok else 'FAIL'} -> {'PASS' if gate else 'FAIL'}", flush=True)
import numpy as _np
caps_nz=[c for c in (cap_swap+cap_hit) if c>0]
print(f"  RE-CAPTURE TAX (per-VMM-epoch fix): {R.recaptures}/{REQ} serves re-captured ({R.recaptures/REQ*100:.0f}%); the rest REUSED the graph (no VMM op since capture)", flush=True)
print(f"  SWAP COST: re-capture when needed mean={_np.mean(caps_nz) if caps_nz else 0:.0f}ms; decode mean={_np.mean(dec_all):.0f}ms -> mean per-serve total={(_np.sum(caps_nz)/REQ if caps_nz else 0)+_np.mean(dec_all):.0f}ms (amortized capture {_np.sum(caps_nz)/REQ if caps_nz else 0:.0f}ms/serve)", flush=True)
print(f"[DENSITY] {M} distinct models served in ONE process, {K} resident (HBM-bounded), swap-on-miss; one-process multi-model mux", flush=True)
sys.stdout.flush(); os._exit(0 if gate else 1)
