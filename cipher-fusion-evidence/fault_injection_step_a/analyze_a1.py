#!/usr/bin/env python3
import json, math
from collections import defaultdict
d=json.load(open("/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a1_results.json"))
R=d["results"]
def band(b): return "sign" if b==15 else ("exp" if 10<=b<=14 else "mant")
print("== A1 GATES ==", json.dumps(d["gates"]))
print(f"== {len(R)} injections, clean_top1={d['meta']['clean_top1']} ==\n")

# harm-by-bit table
print("bit  band   n    naninf%  top1flip%  HARMFUL%   med|d|        med_dmax(finite)")
byb=defaultdict(list)
for r in R: byb[r["bit"]].append(r)
for b in range(16):
    rs=byb[b]; n=len(rs)
    nf=sum(r["naninf"] for r in rs)/n*100
    tf=sum(r["top1_flip"] for r in rs)/n*100
    hm=sum(r["harmful"] for r in rs)/n*100
    fin=[abs(r["delta"]) for r in rs if r["delta"] is not None and math.isfinite(r["delta"])]
    fin.sort(); medd=fin[len(fin)//2] if fin else float('nan')
    dmx=[r["dmax"] for r in rs if math.isfinite(r["dmax"])]; dmx.sort()
    meddmax=dmx[len(dmx)//2] if dmx else float('nan')
    print(f"{b:3d}  {band(b):4s}  {n:4d}  {nf:6.1f}  {tf:8.1f}  {hm:8.1f}   {medd:10.4g}   {meddmax:10.4g}")

# by band
print("\n== by significance band ==")
bb=defaultdict(list)
for r in R: bb[band(r["bit"])].append(r)
for bn in ["sign","exp","mant"]:
    rs=bb[bn]; n=len(rs)
    print(f"{bn:5s} n={n:4d} naninf={sum(r['naninf'] for r in rs)/n*100:5.1f}% "
          f"top1flip={sum(r['top1_flip'] for r in rs)/n*100:5.1f}% HARMFUL={sum(r['harmful'] for r in rs)/n*100:5.1f}%")

# by op type (harmful rate)
print("\n== by op type (HARMFUL%) ==")
bo=defaultdict(list)
for r in R: bo[r["ptype"]].append(r)
for p in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]:
    rs=bo[p];
    if not rs: continue
    n=len(rs); print(f"{p:10s} n={n:4d} HARMFUL={sum(r['harmful'] for r in rs)/n*100:5.1f}% "
          f"naninf={sum(r['naninf'] for r in rs)/n*100:5.1f}% top1flip={sum(r['top1_flip'] for r in rs)/n*100:5.1f}%")

# the coverage-relevant distribution: harmful flips, their |delta|
harm=[r for r in R if r["harmful"]]
beni=[r for r in R if not r["harmful"]]
hf=[abs(r["delta"]) for r in harm if r["delta"] is not None and math.isfinite(r["delta"])]
hinf=sum(1 for r in harm if r["delta"] is None or not math.isfinite(r["delta"]))
bf=[abs(r["delta"]) for r in beni if r["delta"] is not None and math.isfinite(r["delta"])]
hf.sort(); bf.sort()
def pct(x,p):
    if not x: return float('nan')
    return x[min(len(x)-1,int(p*len(x)))]
print(f"\n== HARMFUL flips n={len(harm)} ({len(harm)/len(R)*100:.1f}%) ; finite|d|: "
      f"min={hf[0] if hf else float('nan'):.4g} p10={pct(hf,.1):.4g} p50={pct(hf,.5):.4g} "
      f"p90={pct(hf,.9):.4g} ; inf/nan-delta count={hinf}")
print(f"== BENIGN  flips n={len(beni)} ({len(beni)/len(R)*100:.1f}%) ; finite|d|: "
      f"max={bf[-1] if bf else float('nan'):.4g} p90={pct(bf,.9):.4g} p50={pct(bf,.5):.4g}")
# save summary
json.dump(dict(
  by_bit={b:dict(n=len(byb[b]),naninf=sum(r['naninf'] for r in byb[b]),
                 top1=sum(r['top1_flip'] for r in byb[b]),harmful=sum(r['harmful'] for r in byb[b])) for b in range(16)},
  by_band={bn:dict(n=len(bb[bn]),harmful=sum(r['harmful'] for r in bb[bn])) for bn in ["sign","exp","mant"]},
  harmful_n=len(harm), benign_n=len(beni), harmful_finite_delta=hf, benign_finite_delta=bf,
  harmful_inf_delta=hinf),
  open("/home/ubuntu/cipher-fusion-evidence/fault_injection_step_a/a1_summary.json","w"))
print("\nwrote a1_summary.json")
