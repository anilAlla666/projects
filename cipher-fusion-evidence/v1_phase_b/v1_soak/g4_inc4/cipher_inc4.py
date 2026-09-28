#!/usr/bin/env python3
"""
G-O1 ENGINE INCREMENT 4 -- the both-regimes 100-agent real-workload gate, CIPHER end-to-end.
Composes the proven bricks: multi-model pager residency (inc-3, swap-on-miss) + batched lockstep coalescing
(inc-2 WaveServer, re-captures per wave => the inc-3b swap fix is satisfied for free) + graph-decode over pager
(inc-1). CIPHER owns allocator+dispatch+graphs; NO vLLM in the serving path. CIPHER_RT_DISABLE_AUTO_INIT=1
(=> CIPHER's compute stack Marlin/FP8/fusion/DVFS is OFF -- this proves the multi-model fp16 SUBSTRATE, not the
integrated product; engine (+) compute-actuators do not yet compose).

WORKLOAD (real-workload mock, reproducible fixed trace; NOT live-asyncio): 100 agents = 50 short-burst (10-80 tok)
+ 50 long-burst (128-160 tok), Zipfian model-popularity over the CAPTURE-CORRECT set {Qwen2-7B, Llama-3.1-8B,
Llama-3.2-1B, TinyLlama} (Mistral EXCLUDED -- its SWA static-KV capture is a separate open binding term), gamma
bursty arrivals, growing context. Coalesce only within (model, gen-bucket) cohorts (never short+long together --
that would inflate short-burst P99 to the long wave time). P99 from MEASURED wave serve times.

SAFEGUARD (non-negotiable, carried inc-3b debt): LIVE per-agent KL=0 across ALL 100 agents (every served output vs
its batch-1 solo, greedy-modulo-sub-delta-near-ties), real served output never a re-run; a hard device-assert is
caught and reported as 'corruption re-emerged at agent N', not a silent crash. Gate = 0 FAULT + 0 contamination.
"""
import os, sys, time, numpy as np, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer, classify

K     = int(os.environ.get("K","4"))      # resident-model budget; 4 = all fit (swap INACTIVE), <4 = swap stress
NAGENT= int(os.environ.get("NAGENT","100"))
BMAX  = int(os.environ.get("BMAX","8"))
ZIPF  = float(os.environ.get("ZIPF","1.1"))
RATE  = float(os.environ.get("RATE","0.15"))  # mean gamma inter-arrival (s)
BURST = float(os.environ.get("BURST","0.5"))
P     = int(os.environ.get("P","16"))
SEED  = int(os.environ.get("SEED","0"))
GB=1<<30
SPECS=[("qwen2","/home/ubuntu/models/Qwen2-7B",0xC2),
       ("llama","/home/ubuntu/models/Llama-3.1-8B",0xC3),
       ("llama1b","/home/ubuntu/models/Llama-3.2-1B-Instruct",0xC4),
       ("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
names=[s[0] for s in SPECS]; M=len(SPECS)
POOL=["The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty"]

# ---- engine: M WaveServers + residency (K resident, LRU, swap-on-miss). solo precomputed while resident. ----
pager=CipherPager(); ws={}; rows={}; solos={}; ck_drift=0
print(f"[inc4] loading M={M} models, K={K} resident ({'ALL FIT -> swap INACTIVE' if K>=M else 'forced budget -> swap ACTIVE'})", flush=True)
MAXG=160
for name,path,key in SPECS:
    w=WaveServer(pager, path, key); ws[name]=w
    rr=[w.tok(p, return_tensors="pt").input_ids[0][:P] for p in POOL]
    Pc=min(r.shape[0] for r in rr); rows[name]=[r[:Pc] for r in rr]
    for pi in range(len(POOL)):                       # solo per (model,prompt) at MAXG, while resident
        solos[(name,pi)]=w.solo(rows[name][pi], MAXG)
    w.base.evict()                                    # evict after solo precompute
    print(f"  loaded+solo {name} (evicted)", flush=True)
resident=[]; vmm=0
def ensure_resident(name):
    global vmm
    if name in resident: resident.remove(name); resident.append(name); return False
    if len(resident)>=K:
        v=resident.pop(0); ws[v].base.evict(); vmm+=1
    ws[name].base.restore(); vmm+=1; resident.append(name); return True
for name in names[:K]: ws[name].base.restore(); resident.append(name)   # warm
print(f"[inc4] warm: {resident} resident. HBM free={torch.cuda.mem_get_info()[0]/GB:.1f}GB", flush=True)

# ---- reproducible trace: 100 agents, 50 short + 50 long, Zipfian model, gamma arrivals ----
rng=np.random.default_rng(SEED)
w_model=np.array([1.0/((i+1)**ZIPF) for i in range(M)]); w_model/=w_model.sum()
inter=rng.gamma(shape=BURST, scale=RATE/BURST, size=NAGENT); arr=np.cumsum(inter)
agents=[]
for i in range(NAGENT):
    regime="short" if i<NAGENT//2 else "long"
    gen=int(rng.integers(10,81)) if regime=="short" else int(rng.integers(128,MAXG+1))
    mi=int(rng.choice(M,p=w_model)); pi=int(rng.integers(0,len(POOL)))
    agents.append({"id":i,"arr":float(arr[i]),"model":names[mi],"regime":regime,"pi":pi,"gen":gen,"served":False,"lat":None})
agents.sort(key=lambda a:a["arr"])
from collections import Counter
print(f"[inc4] trace: {NAGENT} agents, model-mix={dict(Counter(a['model'] for a in agents))}, regimes={dict(Counter(a['regime'] for a in agents))}", flush=True)

# ---- RATIFIED ORACLE (Anil): teacher-forced per-step logit/argmax closeness + co-tenant invariance. Free-running
# token-match-vs-batch-1-solo over 48-160 tok over-flags benign near-tie CASCADES (one sub-delta flip -> valid-but-
# different greedy continuation); proven via graph teacher-forced: per-step logits within ~0.039 batch-shape delta,
# co-tenant-invariant. TF kills the cascade (identical context each step) so a mismatch is an ISOLATED step whose
# solo top1-2 margin tells benign (sub-delta near-tie) from real (corruption/contamination, margin>>delta). ----
from transformers import StaticCache
TFTHRESH=0.10   # solo margin above this at a TF mismatch => REAL divergence (batch-shape delta ~0.04 can't flip it)
@torch.no_grad()
def verify_wave(w, cohort_rows, cohort_solos, gens):
    m=w.m; dev="cuda"; B=len(cohort_rows); Pp=cohort_rows[0].shape[0]; Gm=max(gens); Lc=Pp+Gm+8
    def cl(t): return int(max(0,min(w.V-1,int(t))))
    pids=torch.stack([r.to(dev) for r in cohort_rows])
    cache=StaticCache(config=m.config,max_cache_len=Lc)
    first=m(pids,cache_position=torch.arange(Pp,device=dev),past_key_values=cache,use_cache=True).logits[:,-1].argmax(dim=-1)
    sin=torch.zeros(B,1,dtype=torch.long,device=dev); sin.copy_(first.view(B,1)); spos=torch.tensor([Pp],device=dev)
    torch.cuda.synchronize(); tc=time.time(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
    torch.cuda.synchronize(); capm=(time.time()-tc)*1000
    miss=[[] for _ in range(B)]; first_ok=[cl(first[b])==cl(cohort_solos[b][0][0]) for b in range(B)]
    sin.copy_(first.view(B,1)); spos.fill_(Pp); td=time.time()
    for i in range(Gm-1):
        g.replay(); torch.cuda.synchronize(); arg=slog[:,-1].argmax(dim=-1)
        for b in range(B):
            if i+1 < gens[b] and cl(arg[b])!=cl(cohort_solos[b][0][i+1]):
                miss[b].append((i+1, cohort_solos[b][1][i+1]))     # (step, solo margin) -- isolated under TF
        nxt=torch.tensor([cl(cohort_solos[b][0][i+1]) for b in range(B)],device=dev)   # teacher-force ALL to solo
        sin.copy_(nxt.view(B,1)); spos.add_(1)
    decm=(time.time()-td)*1000; del g,cache,slog
    res=[]
    for b in range(B):
        big=[mm for mm in miss[b] if mm[1]>=TFTHRESH]
        if not first_ok[b]: res.append(("FAULT",0,9.9))
        elif big: res.append(("FAULT",big[0][0],big[0][1]))
        elif miss[b]: res.append(("tie",miss[b][0][0],miss[b][0][1]))
        else: res.append(("exact",-1,0.0))
    return res, {"cap_ms":capm,"dec_ms":decm,"B":B,"G":Gm,"waste_steps":sum(Gm-gl for gl in gens),"tot_steps":B*Gm}

# ---- scheduler: single GPU, FCFS-by-cohort; coalesce same (model,regime) up to BMAX; MEASURED wave times ----
vclock=0.0; done=0; waves=0; swaps=0; recaps=0; cap_tot=0.0; dec_tot=0.0; waste=0; totsteps=0
ex=tie=fault=0; faults=[]; lat_short=[]; lat_long=[]; wave_sizes=[]; fault_hit=False
def pending(): return [a for a in agents if not a["served"] and a["arr"]<=vclock]
while done<NAGENT and not fault_hit:
    pend=pending()
    if not pend:
        nxt=min((a["arr"] for a in agents if not a["served"]), default=None)
        if nxt is None: break
        vclock=max(vclock,nxt); continue
    head=min(pend, key=lambda a:a["arr"])             # oldest waiting -> bound head-of-line latency
    cohort=[a for a in pend if a["model"]==head["model"] and a["regime"]==head["regime"]][:BMAX]
    name=head["model"]
    swapped=ensure_resident(name);  swaps+= 1 if swapped else 0
    try:
        res,info=verify_wave(ws[name], [rows[name][a["pi"]] for a in cohort], [solos[(name,a["pi"])] for a in cohort], [a["gen"] for a in cohort])
    except Exception as exn:
        fault_hit=True
        print(f"\n[!!! CORRUPTION RE-EMERGED] wave for model={name} cohort={[a['id'] for a in cohort]} FAULTED: {type(exn).__name__} -- live safeguard CAUGHT it, NOT silently passed", flush=True)
        break
    recaps+=1; cap_tot+=info["cap_ms"]; dec_tot+=info["dec_ms"]; waste+=info["waste_steps"]; totsteps+=info["tot_steps"]
    Tw=(info["cap_ms"]+info["dec_ms"])/1000.0; vclock+=Tw; waves+=1; wave_sizes.append(len(cohort))
    for k,a in enumerate(cohort):
        a["served"]=True; done+=1; a["lat"]=vclock-a["arr"]
        kind,fd,mg=res[k]
        if kind=="exact": ex+=1
        elif kind=="tie": tie+=1
        else: fault+=1; faults.append((a["id"],name,fd,round(mg,3)))
        (lat_short if a["regime"]=="short" else lat_long).append(a["lat"])

# ---- cross-model misroute negative control: a0's model teacher-forced to a DIFFERENT model's solo -> must FAULT ----
nc="n/a"
if done>2 and not fault_hit:
    a0=next(a for a in agents if a["served"]); other=next(nm for nm in names if nm!=a0["model"])
    ensure_resident(a0["model"])
    rnc,_=verify_wave(ws[a0["model"]], [rows[a0["model"]][a0["pi"]]], [solos[(other,a0["pi"])]], [a0["gen"]])
    nc=rnc[0][0]   # a0-model fed other-model solo tokens -> argmax mismatches with LARGE margins -> FAULT

def pctl(x,p): return float(np.percentile(x,p)) if x else 0.0
print(f"\n=== INC-4 RESULT (K={K}, {'swap INACTIVE/all-resident' if K>=M else 'swap ACTIVE/forced-budget'}) ===", flush=True)
print(f"  CORRECTNESS (live, all {done} served agents): exact={ex} near-tie={tie} FAULT={fault}  {'PASS (0 FAULT)' if fault==0 and not fault_hit else 'FAIL '+str(faults[:5])}", flush=True)
print(f"  MISROUTE neg-control: {nc} ({'DETECTED' if nc=='FAULT' else 'NOT DETECTED'})", flush=True)
print(f"  AGENTS-PER-GPU: {done}/{NAGENT} served, {M} distinct models, {K} resident; HBM free={torch.cuda.mem_get_info()[0]/GB:.1f}GB", flush=True)
print(f"  COALESCING: {waves} waves, mean wave={np.mean(wave_sizes) if wave_sizes else 0:.2f}/{BMAX}; SWAPS={swaps} ({'INACTIVE' if K>=M else 'active'}); re-captures={recaps}", flush=True)
print(f"  P99 LATENCY: short p50={pctl(lat_short,50)*1000:.0f}ms p99={pctl(lat_short,99)*1000:.0f}ms (n={len(lat_short)}) | long p50={pctl(lat_long,50)*1000:.0f}ms p99={pctl(lat_long,99)*1000:.0f}ms (n={len(lat_long)})", flush=True)
print(f"  COST: re-capture {cap_tot/max(1,recaps):.0f}ms/wave; decode {dec_tot/max(1,waves):.0f}ms/wave; lockstep waste={waste}/{totsteps} ({waste/max(1,totsteps)*100:.0f}%)", flush=True)
gate=(fault==0) and (not fault_hit) and (nc=="FAULT") and (done==NAGENT)
print(f"[GATE inc4 K={K}] correctness={'PASS' if fault==0 and not fault_hit else 'FAIL'} misroute={'PASS' if nc=='FAULT' else 'FAIL'} served={done}/{NAGENT} -> {'PASS' if gate else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if gate else 1)
