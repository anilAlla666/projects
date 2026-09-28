#!/usr/bin/env python3
# Assemble the E2E verdict from measured primitives (f_of_M.json + saturated graph times).
# Decode config-1 E2E = aggregate fused-check fraction of the decode GEMM time at batch B = f(B).
# Prefill config-2 E2E = aggregate fused-check fraction x prefill-wall-share (serving-mix amortized).
# Config-3 periodic recompute = check_cost/N (latency N steps). All fractions weighted by GEMM time.
import json
F=json.load(open('f_of_M.json'))['rows']
# per-layer GEMM multiplicity by shape
MULT={'k/v':2,'q/o':2,'down':1,'gate/up':2}   # k,v / q,o / down / gate,up
def agg(M, key):  # GEMM-time-weighted aggregate of floor_{key}_pct across the per-layer linears at row-count M
    num=den=0.0
    for sh in ['k/v','q/o','down','gate/up']:
        r=next(x for x in F if x['shape']==sh and x['M']==M)
        num+=MULT[sh]*r[key]*r['gemm_us']; den+=MULT[sh]*r['gemm_us']
    return num/den
print("="*78)
print("DECODE config-1 E2E (fused-check fraction of decode GEMM time = f(B)), per batch B:")
print(f"  {'B':>5s} {'v-only%':>9s} {'v+Cg%':>9s}   (v-only=Cg-epilogue-fused; v+Cg=Cg separate read)")
dec={}
for B in [1,8,32,64,128,256,512]:
    vo=agg(B,'floor_v_pct'); vc=agg(B,'floor_vc_pct'); dec[B]=(vo,vc)
    mk=lambda x:'  <3' if x<3 else ' >=3'
    print(f"  {B:5d} {vo:8.2f}{mk(vo)} {vc:8.2f}{mk(vc)}")
# crossover
def cross(idx):
    Bs=sorted(dec);
    for B in Bs:
        if dec[B][idx]>=3: return B
    return '>512'
print(f"  -> decode crosses 3%: v-only at B={cross(0)}, v+Cg at B={cross(1)}")
pref_vo=agg(2048,'floor_v_pct'); pref_vc=agg(2048,'floor_vc_pct')
print(f"\nPREFILL per-GEMM aggregate fused-check (M=2048): v-only {pref_vo:.2f}%  v+Cg {pref_vc:.2f}%")

# serving mix: c_p (prefill ms/token), c_d (decode ms/token) from saturated graph
SD=json.load(open('synth_decode_B128.json')); SP=json.load(open('synth_decode_B2048.json'))
c_p = SP['t_base_ms']/2048.0          # prefill ms per token (M=2048 tile)
c_d = SD['t_base_ms']/128.0           # decode ms per token-step (B=128)
print(f"\nserving cost/token: prefill {c_p*1000:.1f}us/tok   decode {c_d*1000:.1f}us/tok (B=128)")
def pshare(P,D): return (P*c_p)/(P*c_p+D*c_d)
print("\n"+"="*78)
print("CONFIG-2 prefill-checksum-every-step E2E = prefill_agg x prefill_wall_share, by workload:")
print(f"  {'P':>5s} {'D':>5s} {'pref_share':>11s} {'cfg2 v-only%':>13s} {'cfg2 v+Cg%':>11s} {'cfg1+2 v+Cg%':>13s}")
for P,D in [(2048,128),(512,128),(512,512),(128,512),(256,1024),(1024,2048)]:
    s=pshare(P,D); ds=1-s
    c2vo=pref_vo*s; c2vc=pref_vc*s
    # combined both-regimes (every step), v+Cg, decode at B=128
    comb=ds*dec[128][1]+s*pref_vc
    mk=lambda x:'  <3' if x<3 else ' >=3'
    print(f"  {P:5d} {D:5d} {s*100:10.1f}% {c2vo:12.2f}{mk(c2vo)} {c2vc:10.2f}{mk(c2vc)} {comb:12.2f}{mk(comb)}")

print("\n"+"="*78)
print("CONFIG-3 periodic recompute (prefill), overhead vs N and detection latency:")
print(f"  {'N':>4s} {'checksum/N (of prefill)':>24s} {'full-recompute/N':>18s} {'latency':>10s}")
for N in [1,4,16,64]:
    chk=pref_vc/N; full=100.0/N
    print(f"  {N:4d} {chk:22.2f}% {full:16.1f}% {N:8d} steps")
print("  (E2E = column x prefill_wall_share; checksum=Step-A-class 100% cov, full-recompute=bit-exact all-SDC)")

# overlap per-phase (from MEM% + measured)
print("\n"+"="*78)
print("OVERLAP under saturation (per-phase): decode MEM~87% (no spare BW), prefill MEM~48% (spare).")
print("  deployable FUSED check is bandwidth-bound -> hides only where HBM has headroom = PREFILL,")
print("  not the decode phase that dominates serving wall-time. So overlap does not rescue E2E decode.")
json.dump(dict(decode_f=dec,pref_vo=pref_vo,pref_vc=pref_vc,c_p_ms=c_p,c_d_ms=c_d,
    decode_cross_vonly=cross(0),decode_cross_vc=cross(1)),open('synthesis_e2e.json','w'),indent=1)
print("\nwrote synthesis_e2e.json")
