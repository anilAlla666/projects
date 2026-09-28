#!/usr/bin/env python3
"""
INCREMENT-2 CIPHER-owned async dispatch loop (reproducible).  Drives WaveServer over a PRECOMPUTED gamma arrival
trace (deterministic event order; real GPU replays timed; the virtual clock advances by MEASURED wave time). The
scheduling POLICY is the contribution: coalesce concurrently-pending same-model requests into one lockstep batched
replay (re-capture per wave). This is CIPHER's scheduler at the driver boundary -- NOT vLLM's per-step scheduler.

HEADLINE (advisor): the realized COALESCING RATE under realistic bursty arrivals -- not "batched is faster when
batched" (that's the B-sweep ceiling). Reports realized wave size vs ceiling, throughput vs serial(B=1), and the
work-proportional WASTE (lockstep early-finishers). Correctness = no-contamination + greedy-modulo-near-ties.
Reproducible (fixed trace) by design -- the #1 defense against timing-noise artifacts.  CIPHER_RT_DISABLE_AUTO_INIT=1.
"""
import os, sys, time, numpy as np, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer, classify

MODEL=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
NREQ =int(os.environ.get("NREQ","48"))
BMAX =int(os.environ.get("BMAX","8"))
BURST=float(os.environ.get("BURST","0.3"))   # gamma shape: <1 clustered/bursty, >=1 spread; the coalescing knob
RATE =float(os.environ.get("RATE","0.05"))   # mean inter-arrival (s); smaller => more concurrency
GMIN,GMAX=int(os.environ.get("GMIN","24")),int(os.environ.get("GMAX","64"))
P    =int(os.environ.get("P","16"))
SEED =int(os.environ.get("SEED","0"))

POOL=["The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty",
 "The detective examined the muddy footprints leading away from the abandoned warehouse near",
 "Climate scientists warned that rising ocean temperatures could accelerate the melting of polar",
 "He picked up the worn paperback and began to read the opening lines aloud that",
 "The orchestra tuned their instruments as the conductor raised his baton in the still"]

srv=WaveServer(CipherPager(), MODEL); tok=srv.tok
rows=[tok(p, return_tensors="pt").input_ids[0][:P] for p in POOL]
Pc=min(r.shape[0] for r in rows); rows=[r[:Pc] for r in rows]   # common length (lockstep; left-pad for variable = noted extension)

# ---- precomputed reproducible arrival trace (gamma inter-arrivals; variable gen_len = growing-context/waste) ----
rng=np.random.default_rng(SEED)
inter=rng.gamma(shape=BURST, scale=RATE/BURST, size=NREQ)   # mean=RATE; shape<1 => bursty
arr=np.cumsum(inter)
gens=rng.integers(GMIN,GMAX+1,size=NREQ)
pidx=rng.integers(0,len(rows),size=NREQ)
trace=[{"id":i,"arr":float(arr[i]),"row":rows[pidx[i]],"gen":int(gens[i])} for i in range(NREQ)]

# MODELED wave occupancy advances the virtual clock -> scheduling is DETERMINISTIC (coalescing exactly reproducible,
# decoupled from GPU-timing jitter). Real GPU replays still run+timed for throughput/latency/correctness reporting.
STEP_MODEL_S=0.006; CAP_MODEL_S=0.015
def run(bmax):
    """greedy wave scheduler; virtual clock advances by MODELED wave time (reproducible). real GPU work measured."""
    served={}; waves=[]; vclock=0.0; done=0
    pend=sorted(trace, key=lambda r:(r["arr"], r["id"]))
    while done<NREQ:
        ready=[r for r in pend if r["arr"]<=vclock and r["id"] not in served]
        if not ready:
            nxt=min((r["arr"] for r in pend if r["id"] not in served), default=None)
            if nxt is None: break
            vclock=max(vclock,nxt); continue
        wave=ready[:bmax]
        out,info=srv.serve_wave([w["row"] for w in wave], [w["gen"] for w in wave])
        vclock += max(w["gen"] for w in wave)*STEP_MODEL_S + CAP_MODEL_S   # modeled occupancy (deterministic)
        for k,w in enumerate(wave): served[w["id"]]=out[k]; done+=1
        waves.append({"size":len(wave),"info":info,"ids":[w["id"] for w in wave]})
    return served, waves

# ===== batched run (the scheduler) =====
t0=time.time(); served,waves=run(BMAX); wall=time.time()-t0
sizes=[w["size"] for w in waves]
gpu_ms=sum(w["info"]["cap_ms"]+w["info"]["dec_ms"] for w in waves)
useful_tok=sum(t["gen"] for t in trace)
waste=sum(w["info"]["waste_steps"] for w in waves); totsteps=sum(w["info"]["tot_steps"] for w in waves)
cap_tot=sum(w["info"]["cap_ms"] for w in waves); dec_tot=sum(w["info"]["dec_ms"] for w in waves)
# per-step decode latency at the realized batch sizes
perstep=[w["info"]["dec_ms"]/max(1,w["info"]["G"]-1) for w in waves]

# ===== serial baseline (B=1) on the SAME trace =====
served1,waves1=run(1)
gpu1_ms=sum(w["info"]["cap_ms"]+w["info"]["dec_ms"] for w in waves1)

print(f"[sched {os.path.basename(MODEL)}] NREQ={NREQ} BMAX={BMAX} burst={BURST} rate={RATE} gen={GMIN}-{GMAX}", flush=True)
print(f"  COALESCING: waves={len(waves)} mean_wave={np.mean(sizes):.2f}/{BMAX} (max={max(sizes)}) "
      f"frac_in_batch>1={sum(s for s in sizes if s>1)/NREQ:.2f}  realized_fill={np.mean(sizes)/BMAX:.2f}", flush=True)
print(f"  THROUGHPUT: useful_tok={useful_tok} gpu_time batched={gpu_ms:.0f}ms serial(B=1)={gpu1_ms:.0f}ms "
      f"-> speedup={gpu1_ms/gpu_ms:.2f}x  ({useful_tok/(gpu_ms/1000):.0f} tok/s batched vs {useful_tok/(gpu1_ms/1000):.0f} serial)", flush=True)
print(f"  WASTE(lockstep early-finish): {waste}/{totsteps} slot-steps = {waste/totsteps*100:.0f}%  "
      f"capture_overhead={cap_tot/(cap_tot+dec_tot)*100:.0f}% of GPU time", flush=True)
print(f"  LATENCY: per-step decode mean={np.mean(perstep):.2f}ms (bandwidth-floored; batching buys THROUGHPUT not per-step latency)", flush=True)

# ===== CORRECTNESS: per-agent KL=0 (ratified) on a sample, vs solo =====
import random; random.seed(1)
sample=random.sample(range(NREQ), min(8,NREQ)); ex=tie=fault=0; faults=[]
for rid in sample:
    tr=trace[rid]; sr,mar=srv.solo(tr["row"], tr["gen"]); kind,fd,mg=classify(served[rid], sr, mar)
    if kind=="exact": ex+=1
    elif kind=="tie": tie+=1
    else: fault+=1; faults.append((rid,fd,mg))
print(f"  CORRECTNESS(sample={len(sample)}): exact={ex} near-tie={tie} FAULT={fault}  {'PASS' if fault==0 else 'FAIL '+str(faults)}", flush=True)

# ===== ROUTING NEGATIVE CONTROL: feed agent A's served output against a DIFFERENT-prompt agent's solo -> MUST FAULT =====
nc_ok=False; k_swapped="n/a"
a=sample[0]
b=next((j for j in range(NREQ) if j!=a and not torch.equal(trace[j]["row"], trace[a]["row"])), None)  # any different-prompt agent
if b is not None:
    sr_b,mar_b=srv.solo(trace[b]["row"], trace[a]["gen"])
    k_swapped,_,_=classify(served[a], sr_b, mar_b)   # A's output judged against B's solo -> should be FAULT
    nc_ok=(k_swapped=="FAULT")
print(f"  ROUTING neg-control: agent-A output vs different-agent-B solo -> {k_swapped} "
      f"({'DETECTED (check discriminates slot->agent routing)' if nc_ok else 'NOT DETECTED -- routing check has no power'})", flush=True)

gate = (fault==0) and nc_ok
print(f"[GATE sched] correctness={'PASS' if fault==0 else 'FAIL'} routing-nc={'PASS' if nc_ok else 'FAIL'} -> {'PASS' if gate else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if gate else 1)
