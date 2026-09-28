#!/usr/bin/env python3
# REAL-ENGINE SMOKE for the `agent` regime. Imports the VALIDATED engine modules
# UNCHANGED (cipher_engine.CipherPager, cipher_engine_batched.WaveServer/classify)
# and drives them with a genuinely small Class-D workload: TinyLlama only, B=4
# agents in one lockstep wave, G=8 tokens. Returns MEASURED numbers (real graph
# capture+replay over the pager, real per-agent correctness vs solo, real power).
#
# This is the cheap hardware proof that the router dispatches into the real
# substrate. It is NOT the 100-agent gate -- it's a 4-agent smoke of the same code.
import os, sys, json, time
sys.path.insert(0, "/home/ubuntu")  # import the validated engine modules in place

regime = sys.argv[1] if len(sys.argv) > 1 else "agent"
job = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
B = int(job.get("agents", 4)); G = int(job.get("tokens", 8)); P = 12

import torch
torch.manual_seed(0)
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer, classify

seen = {k: os.environ[k] for k in ("CIPHER_RT_DISABLE_AUTO_INIT", "CUDA_INJECTION64_PATH",
                                   "CIPHER_FP8", "CIPHER_VOLT", "K") if k in os.environ}

POOL = ["The history of artificial intelligence began in the 1950s when researchers first",
        "In a distant galaxy far beyond the reach of human telescopes a civilization had",
        "The recipe calls for two cups of flour a pinch of salt and three large",
        "Quantum computing promises to revolutionize cryptography by factoring large numbers in"]

class PowerSampler:
    def __init__(s): s.on=False; s.s=[]
    def _loop(s):
        import pynvml; pynvml.nvmlInit(); d=pynvml.nvmlDeviceGetHandleByIndex(0)
        while s.on:
            try: s.s.append(pynvml.nvmlDeviceGetPowerUsage(d)/1000.0)
            except Exception: pass
            time.sleep(0.05)
    def start(s):
        import threading; s.on=True; s.t=threading.Thread(target=s._loop,daemon=True); s.t.start()
    def stop(s): s.on=False; s.t.join(timeout=1)
    def avg(s): return sum(s.s)/len(s.s) if s.s else 0.0

t0=time.time()
pager=CipherPager()
w=WaveServer(pager, "/home/ubuntu/models/TinyLlama-1.1B", 0xC5)
load_s=time.time()-t0

# B prompt rows, all length P (lockstep requirement), G tokens each
rows=[w.tok(POOL[i%len(POOL)], return_tensors="pt").input_ids[0][:P] for i in range(B)]
gens=[G]*B
solos=[w.solo(rows[b], G) for b in range(B)]   # batch-1 references + margins

ps=PowerSampler(); ps.start(); ts=time.time()
out, info = w.serve_wave(rows, gens)            # REAL batched graph-decode over the pager
serve_s=time.time()-ts; ps.stop()

# per-agent correctness vs solo (the ratified exact|tie|FAULT gate)
ex=tie=fault=0; verdicts=[]
for b in range(B):
    kind, k, m = classify(out[b], solos[b][0], solos[b][1])
    verdicts.append(f"{kind}@{k}" if k>=0 else kind)
    if kind=="exact": ex+=1
    elif kind=="tie": tie+=1
    else: fault+=1

# NEGATIVE CONTROL (misroute): agent 0's tokens vs a DIFFERENT prompt's solo -> must FAULT
nc_kind,_,_ = classify(out[0], solos[1][0], solos[1][1]) if B>1 else ("n/a",0,0)

served_tok=B*G; avgW=ps.avg(); thru=served_tok/serve_s if serve_s else 0
tokW=thru/avgW if avgW else 0
metrics={"agents_per_gpu": B, "models": 1, "tokens_each": G,
         "fault": fault, "exact": ex, "near_tie": tie,
         "verdicts": verdicts, "misroute_negctrl": nc_kind,
         "fleet_tok_w": round(tokW,4), "thru_tok_s": round(thru,1), "avg_power_w": round(avgW,1),
         "serve_ms": round(serve_s*1000,1), "cap_ms": round(info["cap_ms"],1),
         "dec_ms": round(info["dec_ms"],1), "load_s": round(load_s,1),
         "hbm_free_gb": round(torch.cuda.mem_get_info()[0]/(1<<30),1)}

print("CIPHER_MOCK_RESULT " + json.dumps({
    "regime": regime, "name": job.get("name"), "injection_state_seen": seen,
    "metrics": metrics, "mock": False, "real_engine": True}), flush=True)
sys.stdout.flush(); os._exit(0)
