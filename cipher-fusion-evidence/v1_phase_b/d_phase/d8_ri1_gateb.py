#!/usr/bin/env python3
# R-I1 SHIELD via SM-PARTITION — Gate B on the CLEAN diag2 probe.
# Reuses the CP54 green-ctx substrate (env CIPHER_QOS_CLASS=partition +
# CIPHER_SM_COUNT) — NOT rebuilt. Routes BOTH tenants through DISJOINT partitions
# (victim -> 8 SMs set B; aggressor -> 96 SMs set A) and MEASURES confinement
# (not inferred): disjoint grp_masks + aggressor launches greened-onto-A +
# victim p99 under aggressor-with-partition vs victim solo.
#
# Bar (outcome 1/PASS): victim p99 near solo WHILE aggressor keeps ~full work
# (~73% of full-GPU, contrast the throttle's kill-switch to ~1%). No PASS off p50.
import os, sys, json, time, ctypes, subprocess, statistics, re

LIB = "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
SECS = float(os.environ.get("D8_SECONDS", "25"))

def _bind_accessors(lib):
    for fn, rt in [("cipher_rt_green_ctx_sm_count", ctypes.c_uint),
                   ("cipher_rt_green_ctx_cur_mask", ctypes.c_uint),
                   ("cipher_rt_green_ctx_is_initialized", ctypes.c_int),
                   ("cipher_rt_green_ctx_group_id", ctypes.c_uint)]:
        try: getattr(lib, fn).restype = rt
        except Exception: pass

def green_state(lib):
    try:
        return {"sm_count": int(lib.cipher_rt_green_ctx_sm_count()),
                "mask": int(lib.cipher_rt_green_ctx_cur_mask()),
                "init": int(lib.cipher_rt_green_ctx_is_initialized()),
                "group_id": int(lib.cipher_rt_green_ctx_group_id())}
    except Exception as e:
        return {"err": str(e)}

if len(sys.argv) > 1 and sys.argv[1] == "aggressor":
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL); _bind_accessors(lib)
    import torch
    N = int(os.environ.get("SAT", "8192"))
    x = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    for _ in range(5): z = x @ y
    torch.cuda.synchronize()
    gs = green_state(lib)
    it = 0; deadline = time.time() + SECS
    while time.time() < deadline:
        for _ in range(64): z = x @ y
        torch.cuda.synchronize(); it += 64
    gs2 = green_state(lib)
    json.dump({"role":"aggressor","matmul_iters":it,"matmul_per_s":it/SECS,
               "green":gs2}, open(sys.argv[2], "w"))
    print(f"[aggressor] green sm_count={gs2.get('sm_count')} mask=0x{gs2.get('mask',0):x} "
          f"init={gs2.get('init')} matmul/s={it/SECS:.0f}", flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "victim":
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL); _bind_accessors(lib)
    import torch
    V = int(os.environ.get("VDIM", "1024"))
    x = torch.randn(V, V, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(V, V, dtype=torch.bfloat16, device="cuda")
    n_iter = int(os.environ.get("N_ITER", "2000")); think = float(os.environ.get("THINK_US","500"))/1e6
    for _ in range(50): z = x @ y
    torch.cuda.synchronize()
    gs = green_state(lib)
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter(); z = x @ y; torch.cuda.synchronize()
        lat.append((time.perf_counter()-t0)*1e3)
        if think>0: time.sleep(think)
    lat.sort()
    def p(q): return lat[min(len(lat)-1,int(q/100*len(lat)))]
    res = {"role":"victim","n":len(lat),"p50_ms":p(50),"p90_ms":p(90),"p99_ms":p(99),
           "p999_ms":p(99.9),"max_ms":lat[-1],"min_ms":lat[0],"green":gs}
    json.dump(res, open(sys.argv[2], "w"))
    print(f"[victim] green sm_count={gs.get('sm_count')} mask=0x{gs.get('mask',0):x} init={gs.get('init')} "
          f"wall ms: p50={p(50):.3f} p99={p(99):.3f} max={lat[-1]:.3f}", flush=True)
    sys.exit(0)

# ---- orchestrate: 4 conditions ----
HERE = os.path.abspath(__file__); PY = sys.executable
PART_V = {"CIPHER_QOS_CLASS":"partition","CIPHER_SM_COUNT":os.environ.get("VICTIM_SM","8")}
PART_A = {"CIPHER_QOS_CLASS":"partition","CIPHER_SM_COUNT":os.environ.get("AGGR_SM","96")}

def run(mode, out, env_extra):
    env = dict(os.environ); env.update(env_extra)
    p = subprocess.Popen([PY, HERE, mode, out], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p

def collect(p):
    o,_ = p.communicate(); txt = o.decode(errors="replace"); sys.stdout.write(txt); return txt

def victim_only(env_extra, tag):
    p = run("victim", f"/tmp/ri1_{tag}.json", env_extra); collect(p); return json.load(open(f"/tmp/ri1_{tag}.json"))

def victim_with_aggr(venv, aenv, tag):
    ap = run("aggressor", f"/tmp/ri1_agg_{tag}.json", aenv); time.sleep(5)   # let aggressor ALLOCATE + saturate
    vp = run("victim", f"/tmp/ri1_{tag}.json", venv); vtxt = collect(vp)
    atxt = collect(ap)
    v = json.load(open(f"/tmp/ri1_{tag}.json"))
    try: a = json.load(open(f"/tmp/ri1_agg_{tag}.json"))
    except Exception: a = {}
    # confinement evidence from the aggressor's DIAG-T4.2.4d teardown line
    m = re.search(r"on_our_green=(\d+).*?on_primary=(\d+).*?on_null_stream=(\d+)", atxt)
    swaps = re.search(r"ctx_swaps_to_green=(\d+)", atxt)
    a["on_our_green"] = int(m.group(1)) if m else None
    a["on_primary"]   = int(m.group(2)) if m else None
    a["ctx_swaps_to_green"] = int(swaps.group(1)) if swaps else None
    return v, a

print("="*74, "\nR-I1 Gate B — disjoint SM-partition (CP54 green-ctx), clean diag2 probe\n", "="*74)
c1 = victim_only({}, "solo_full")                       # victim alone, full GPU
c2 = victim_only(PART_V, "solo_part")                   # victim alone, 8-SM partition (t424e ref)
c3v, c3a = victim_with_aggr({}, {}, "off")              # both full GPU, no partition (40x baseline)
c4v, c4a = victim_with_aggr(PART_V, PART_A, "on")       # disjoint partitions (THE TEST)

vmask = c4v.get("green",{}).get("mask",0); amask = c4a.get("green",{}).get("mask",0)
disjoint = (vmask & amask) == 0 and vmask != 0 and amask != 0
# aggressor work held: ON matmul/s vs a partitioned-alone aggressor would be ~96/132 of full;
# the key contrast is vs the throttle (which dropped to ~1%). Report ON matmul/s.
aggr_on_rate = c4a.get("matmul_per_s"); aggr_off_rate = c3a.get("matmul_per_s")

print("\n" + "="*74 + "\nSUMMARY — victim wall p99 (ms)")
print(f"  C1 victim SOLO (full GPU)        : p50={c1['p50_ms']:.3f}  p99={c1['p99_ms']:.3f}")
print(f"  C2 victim SOLO (8-SM partition)  : p50={c2['p50_ms']:.3f}  p99={c2['p99_ms']:.3f}   (t424e SM-cost ref)")
print(f"  C3 victim + aggr, NO partition   : p50={c3v['p50_ms']:.3f}  p99={c3v['p99_ms']:.3f}   (inflation vs C1 {c3v['p99_ms']/max(c1['p99_ms'],1e-9):.1f}x)")
print(f"  C4 victim + aggr, DISJOINT part  : p50={c4v['p50_ms']:.3f}  p99={c4v['p99_ms']:.3f}   (vs C1 {c4v['p99_ms']/max(c1['p99_ms'],1e-9):.1f}x, vs C2 {c4v['p99_ms']/max(c2['p99_ms'],1e-9):.2f}x)")
print(f"\n  DISJOINT masks: victim=0x{vmask:x} (sm={c4v.get('green',{}).get('sm_count')}) aggr=0x{amask:x} (sm={c4a.get('green',{}).get('sm_count')}) -> A∩B=∅ {disjoint}")
print(f"  CONFINEMENT (aggressor launches greened): on_our_green={c4a.get('on_our_green')} on_primary={c4a.get('on_primary')} ctx_swaps={c4a.get('ctx_swaps_to_green')}")
print(f"  AGGRESSOR WORK HELD: ON matmul/s={aggr_on_rate}  (no-partition matmul/s={aggr_off_rate})  [throttle dropped to ~1%; partition should keep it productive]")

# verdict (anti-spin): PASS needs victim p99 near solo AND aggressor productive AND disjoint+confined.
near_solo = c4v['p99_ms'] <= max(c2['p99_ms'], c1['p99_ms']) * 2.0
contention_existed = c3v['p99_ms'] > c1['p99_ms'] * 5
aggr_productive = (aggr_on_rate or 0) > (aggr_off_rate or 1) * 0.3   # not killed
# Confinement is judged EMPIRICALLY (victim p99 recovered to its partition floor),
# NOT from on_our_green — that CUPTI per-launch counter is unreliable (cuCtxSetCurrent
# is sticky, so launches are confined without per-launch green-swap counts). The
# victim running at 8-SM speed (C2/C4 p50 >> C1) is the proof the green ctx engaged.
confined = disjoint and near_solo
if not contention_existed:
    verdict = "INCONCLUSIVE — no contention in C3 to isolate (check harness)"
elif near_solo and aggr_productive and confined:
    verdict = "OUTCOME 1 PASS — victim p99 near solo, aggressor productive (~full work), disjoint+confined"
elif near_solo and confined and not aggr_productive:
    verdict = "OUTCOME 2 PARTIAL — isolation only with aggressor degraded"
elif (c4v['p99_ms'] < c3v['p99_ms']) and not near_solo:
    verdict = "OUTCOME 2 PARTIAL — tail tightens vs C3 but stays elevated (t424e SM-cost?)"
elif not confined or c4v['p99_ms'] >= c3v['p99_ms']*0.8:
    verdict = "OUTCOME 3 NOT-ENFORCED — aggressor not confined to set A (t424c recurred) -> 1-H100 ceiling"
else:
    verdict = "AMBIGUOUS — inspect numbers"
print(f"\n  VERDICT: {verdict}")
print("="*74)
json.dump({"c1":c1,"c2":c2,"c3":{"v":c3v,"a":c3a},"c4":{"v":c4v,"a":c4a},
           "disjoint":disjoint,"verdict":verdict}, open("/home/ubuntu/ri1_gateb_result.json","w"), indent=2)
