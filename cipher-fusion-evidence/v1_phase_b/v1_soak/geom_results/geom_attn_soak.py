#!/usr/bin/env python3
# Geometry-attention detection — 30-min sustained soak on the NEW binary (7cdc68ef),
# CIPHER_GEOM_ATTN=on. The counter fires on the UNIVERSAL CUPTI launch callback, so
# prove (by measurement) no crash / no leak / no NaN under sustained load AND that the
# counter is actually exercised (increments). Single-process torch (ctypes-readable
# counter; reliable). Result: geom_attn_soak.json.
import os, time, json, ctypes, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# DECODE workload (TinyLlama generate) — the regime that fires the ATTENTION
# classifier (validated: 10656). Sustained decode exercises BOTH the universal
# launch path AND the geom-attn counter.
M = "/home/ubuntu/models/TinyLlama-1.1B"
DUR = float(os.environ.get("D9_SOAK_SEC", "1800"))
so = os.environ["MARLIN_SO"]; lib = ctypes.CDLL(so)
lib.cipher_rt_geom_attn_intercepts.restype = ctypes.c_ulong
lib.cipher_rt_geom_launches.restype = ctypes.c_ulong
def gattn(): return int(lib.cipher_rt_geom_attn_intercepts())
def glaunch(): return int(lib.cipher_rt_geom_launches())
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = tok("The history of computing spans several distinct eras, each defined by", return_tensors="pt").input_ids.cuda()
def gen():
    with torch.no_grad():
        return m.generate(ids, max_new_tokens=32, do_sample=False)[0, ids.shape[1]:].tolist()
for _ in range(3): gen()   # warmup → ATTENTION classification engages
torch.cuda.reset_peak_memory_stats()
base_mem = torch.cuda.memory_allocated()
attn0, lau0 = gattn(), glaunch()
t0 = time.time(); nfwd = 0; nan = 0; incoh = 0; ref = None; cps = []; nextcp = t0 + 60
while time.time() - t0 < DUR:
        toks = gen(); nfwd += 1
        if len(set(toks)) <= 1: nan += 1          # all-identical => garbage/corruption
        if ref is None: ref = toks
        elif toks != ref: incoh += 1               # deterministic decode => must match
        torch.cuda.synchronize()
        if time.time() >= nextcp:
            mem = torch.cuda.memory_allocated()
            cps.append({"t": round(time.time()-t0), "nfwd": nfwd, "free_mib": round(torch.cuda.mem_get_info()[0]/2**20,1),
                        "geom_attn": gattn(), "geom_launches": glaunch(), "nan": nan, "incoh": incoh})
            print("[geom-soak t=%ds] fwd=%d geom_attn=%d launches=%d nan=%d incoh=%d"
                  % (cps[-1]["t"], nfwd, cps[-1]["geom_attn"], cps[-1]["geom_launches"], nan, incoh), flush=True)
            nextcp += 60
attn1, lau1 = gattn(), glaunch()
frees = [c["free_mib"] for c in cps]
spread = (max(frees)-min(frees)) if frees else 0
res = {"so": so, "soak_sec": round(time.time()-t0,1), "forwards": nfwd, "nan": nan, "incoherent": incoh,
       "geom_attn_delta": attn1-attn0, "geom_launches_delta": lau1-lau0,
       "counter_exercised": (attn1-attn0) > 0,
       "checkpoint_free_spread_mib": round(spread,1), "checkpoints": cps,
       "PASS": bool(nan==0 and incoh==0 and nfwd>0 and (attn1-attn0)>0 and spread<2048)}
json.dump(res, open("/home/ubuntu/geom_attn_soak.json","w"), indent=2)
print("GEOM-SOAK %s: %d fwd / %.0fs, geom_attn+%d launches+%d nan=%d incoh=%d spread=%.1fMiB"
      % ("PASS" if res["PASS"] else "FAIL", nfwd, res["soak_sec"], res["geom_attn_delta"],
         res["geom_launches_delta"], nan, incoh, spread), flush=True)
