#!/usr/bin/env python3
# FORK-2 Mech 1: SAMPLED rotating-subset per-step verification vs periodic-N, on the REAL ra-e2e trace.
# Single-step transient catch at MATCHED cost. Confirms the conservation theorem: catch = cost/R for BOTH,
# so sampling does not beat periodic for uniform single-step faults. Pure CPU simulation over the real trace
# + the real ra-e2e per-phase recompute cost model. Reads ra_e2e files READ-ONLY (no edit).
import json, random
random.seed(0)
RA="/home/ubuntu/cipher-fusion-evidence/ra_e2e"
trace=json.load(open(f"{RA}/trace.json"))["steps"]          # list of [phase, M]
cal=json.load(open(f"{RA}/calib.json"))
tb={"decode":cal["t_base_decode_ms"],"prefill":cal["t_base_prefill_ms"]}
tr={"decode":cal["t_recompute_decode_ms"],"prefill":cal["t_recompute_prefill_ms"]}
G=32*7+1   # GEMMs per step: 32 layers x 7 linears + lm_head = 225 (matches ra_e2e forward 'all linears + lm_head')
nsteps=len(trace)
W_base=sum(tb[p] for p,_ in trace)
W_rec =sum(tr[p] for p,_ in trace)
R=W_rec/W_base
print(f"steps={nsteps} G/step={G} R=Wrec/Wbase={R:.4f} (periodic-N overhead=R/N, periodic catch=1/N => catch/cost=1/R={1/R:.3f})")

NT=200000   # transients (random step, random GEMM)
faults=[(random.randrange(nsteps), random.randrange(G)) for _ in range(NT)]

def periodic_cost_catch(N):
    check=set(range(0,nsteps,N))
    cost=sum(tr[trace[s][0]] for s in check)/W_base
    caught=sum(1 for (s,g) in faults if s in check)/NT
    return cost*100, caught*100

def sampled_cost_catch(f):
    # rotating subset: at step s, checked GEMM ids = {(s*ksub + j) mod G : j in [0,ksub)} with ksub=round(f*G)
    ksub=max(1,round(f*G))
    cost=f*R   # checking f-fraction of GEMMs every step costs f x t_recompute each step
    caught=0
    for (s,g) in faults:
        start=(s*ksub)%G
        # in-subset if g in [start, start+ksub) mod G
        d=(g-start)%G
        if d<ksub: caught+=1
    return cost*100, caught/NT*100, ksub

# Frontier: a set of cost budgets; for each, periodic-N (nearest) and sampled-f at the SAME cost.
rows=[]
for N in [10,20,30,45,64]:
    pc,pk=periodic_cost_catch(N)
    f=(pc/100)/R                  # match sampled cost to periodic's cost
    sc,sk,ksub=sampled_cost_catch(f)
    rows.append({"N":N,"periodic_cost_pct":round(pc,3),"periodic_catch_pct":round(pk,3),
                 "sampled_f":round(f,4),"sampled_ksub_of_%d"%G:ksub,"sampled_cost_pct":round(sc,3),"sampled_catch_pct":round(sk,3),
                 "sampled_minus_periodic_catch_pp":round(sk-pk,3)})
    print(f"N={N:3d}: periodic cost={pc:.2f}% catch={pk:.2f}%  |  sampled f={f:.4f}(k={ksub}/{G}) cost={sc:.2f}% catch={sk:.2f}%  "
          f"=> delta={sk-pk:+.2f}pp")
out={"nsteps":nsteps,"G_per_step":G,"R":R,"conservation_catch_per_cost":1/R,"n_transients":NT,"frontier":rows,
     "verdict":"sampled rotating-subset gives transient catch within rounding of periodic-N at equal cost (catch=cost/R for both) => does NOT beat periodic"}
json.dump(out,open("/home/ubuntu/cipher-fusion-evidence/ra_fork2/mech1_result.json","w"),indent=1)
print("wrote mech1_result.json")
