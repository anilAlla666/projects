#!/usr/bin/env python3
# SCALED real-engine smoke for the `agent` regime -- a miniature cipher_inc4.
# Reuses the VALIDATED engine (cipher_engine.CipherPager + cipher_engine_batched
# .WaveServer/solo/serve_wave/classify) UNCHANGED. Drives 2 SMALL models
# (TinyLlama + Llama-3.2-1B) with N agents (short+long mix), gamma-ish arrivals,
# same-(model,regime) coalescing into lockstep waves, and MEASURES per-agent
# latency -> real p99, plus real tok/W from nvml. Cheap (both models ~1-2GB).
import os, sys, json, time
sys.path.insert(0, "/home/ubuntu")
import numpy as np, torch
torch.manual_seed(0)
torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN MHA breaks CUDA-graph capture on torch 2.11 (Llama-3.2-1B); flash/mem-eff/math are capture-safe
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer, classify

regime = sys.argv[1] if len(sys.argv) > 1 else "agent"
job = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
N    = int(job.get("agents", 12)); BMAX = int(job.get("bmax", 4)); P = 12; SEED = 0
seen = {k: os.environ[k] for k in ("CIPHER_RT_DISABLE_AUTO_INIT","CUDA_INJECTION64_PATH",
                                   "CIPHER_FP8","CIPHER_VOLT","K") if k in os.environ}
# NOTE: agent lane uses TinyLlama only. Llama-3.2-1B's rope_scaling="llama3" does a host-side op
# mid-forward that is illegal under CUDA-graph capture on torch 2.11 (cudaErrorStreamCaptureInvalidated;
# a failed capture corrupts the context uncatchably). The validated inc-4 multi-model engine ran on an
# older torch. Multi-model realness is carried by the density lane (smoke_density_real.py). Real p99 +
# tok/W here come from 12 agents across multiple coalesced waves on one model.
SPECS = [("tiny","/home/ubuntu/models/TinyLlama-1.1B",0xC5)]
names = [s[0] for s in SPECS]; M = len(SPECS)
POOL = ["The history of artificial intelligence began in the 1950s when researchers first",
        "In a distant galaxy far beyond the reach of human telescopes a civilization had",
        "The recipe calls for two cups of flour a pinch of salt and three large",
        "Quantum computing promises to revolutionize cryptography by factoring large numbers in"]

class PowerSampler:
    def __init__(s): s.on=False; s.s=[]; s.c=[]
    def _loop(s):
        import pynvml; pynvml.nvmlInit(); d=pynvml.nvmlDeviceGetHandleByIndex(0)
        while s.on:
            try: s.s.append(pynvml.nvmlDeviceGetPowerUsage(d)/1000.0); s.c.append(pynvml.nvmlDeviceGetClockInfo(d,0))
            except Exception: pass
            time.sleep(0.05)
    def start(s):
        import threading; s.on=True; s.t=threading.Thread(target=s._loop,daemon=True); s.t.start()
    def stop(s): s.on=False; s.t.join(timeout=1)
    def avg(s): return sum(s.s)/len(s.s) if s.s else 0.0

# ---- load both models into pager regions (real); WARMUP-capture each to prime allocator pools (capture-safety);
#      gracefully EXCLUDE any model whose capture won't prime on this torch build (honest partial coverage) ----
t0=time.time(); pager=CipherPager(); ws={}; rows={}; solos={}
for nm,path,key in SPECS:
    w=WaveServer(pager, path, key); ws[nm]=w
    rr=[w.tok(p, return_tensors="pt").input_ids[0][:P] for p in POOL]
    Pc=min(r.shape[0] for r in rr); rows[nm]=[r[:Pc] for r in rr]
load_s=time.time()-t0

# ---- reproducible trace: N agents, half short(8) half long(24), round-robin model, gamma arrivals ----
rng=np.random.default_rng(SEED)
inter=rng.gamma(shape=0.5, scale=0.1/0.5, size=N); arr=np.cumsum(inter)
agents=[]
for i in range(N):
    reg="short" if i%2==0 else "long"; gen=8 if reg=="short" else 24
    nm=names[i%M]; pi=int(rng.integers(0,len(POOL)))
    agents.append({"id":i,"arr":float(arr[i]),"model":nm,"regime":reg,"pi":pi,"gen":gen,"served":False,"lat":None})
agents.sort(key=lambda a:a["arr"])

# precompute solos for the (model,prompt,gen) used
for a in agents:
    k=(a["model"],a["pi"],a["gen"])
    if k not in solos: solos[k]=ws[a["model"]].solo(rows[a["model"]][a["pi"]], a["gen"])

# ---- scheduler: FCFS-by-cohort, coalesce same (model,regime) up to BMAX; MEASURED wave times ----
vclock=0.0; done=0; waves=0; ex=tie=fault=0; faults=[]; lat_short=[]; lat_long=[]; served_tok=0
ps=PowerSampler(); ps.start(); wall0=time.time()
def pending(): return [a for a in agents if not a["served"] and a["arr"]<=vclock]
while done<N:
    pend=pending()
    if not pend:
        nxt=min((a["arr"] for a in agents if not a["served"]), default=None)
        if nxt is None: break
        vclock=max(vclock,nxt); continue
    head=min(pend, key=lambda a:a["arr"])
    cohort=[a for a in pend if a["model"]==head["model"] and a["regime"]==head["regime"]][:BMAX]
    nm=head["model"]; gens=[a["gen"] for a in cohort]
    out,info=ws[nm].serve_wave([rows[nm][a["pi"]] for a in cohort], gens)
    Tw=(info["cap_ms"]+info["dec_ms"])/1000.0; vclock+=Tw; waves+=1
    for k,a in enumerate(cohort):
        sk=(a["model"],a["pi"],a["gen"]); kind,fk,mg=classify(out[k], solos[sk][0], solos[sk][1])
        if kind=="exact": ex+=1
        elif kind=="tie": tie+=1
        else: fault+=1; faults.append((a["id"],nm,fk,round(mg,3)))
        a["served"]=True; done+=1; a["lat"]=vclock-a["arr"]; served_tok+=a["gen"]
        (lat_short if a["regime"]=="short" else lat_long).append(a["lat"])
ps.stop(); serve_wall=time.time()-wall0; avgW=ps.avg()
# negative-control misroute on agent 0: feed its served tokens against a WRONG reference -> must FAULT.
# (>1 model: a different model's solo; 1 model: a different prompt's solo -- both are genuine misroutes.)
a0=agents[0]; so=ws[a0["model"]].serve_wave([rows[a0["model"]][a0["pi"]]],[a0["gen"]])[0][0]
if M>1:
    other=next(nm for nm in names if nm!=a0["model"]); wk=(other,a0["pi"],a0["gen"])
    if wk not in solos: solos[wk]=ws[other].solo(rows[other][a0["pi"]], a0["gen"])
else:
    op=(a0["pi"]+1)%len(POOL); wk=(a0["model"],op,a0["gen"])
    if wk not in solos: solos[wk]=ws[a0["model"]].solo(rows[a0["model"]][op], a0["gen"])
nc,_,_=classify(so, solos[wk][0], solos[wk][1])

def pctl(x,p): return round(float(np.percentile(x,p))*1000,1) if x else None
thru=served_tok/serve_wall if serve_wall else 0; tokW=thru/avgW if avgW else 0
metrics={"agents_per_gpu":done,"models":M,"fault":fault,"exact":ex,"near_tie":tie,
         "misroute_negctrl":nc,"waves":waves,
         "p99_short_ms":pctl(lat_short,99),"p99_long_ms":pctl(lat_long,99),
         "p50_short_ms":pctl(lat_short,50),"p50_long_ms":pctl(lat_long,50),
         "fleet_tok_w":round(tokW,4),"thru_tok_s":round(thru,1),"avg_power_w":round(avgW,1),
         "served_tok":served_tok,"serve_wall_s":round(serve_wall,2),"load_s":round(load_s,1),
         "hbm_free_gb":round(torch.cuda.mem_get_info()[0]/(1<<30),1)}
print("CIPHER_MOCK_RESULT "+json.dumps({"regime":regime,"name":job.get("name"),
      "injection_state_seen":seen,"metrics":metrics,"mock":False,"real_engine":True}),flush=True)
sys.stdout.flush(); os._exit(0)
