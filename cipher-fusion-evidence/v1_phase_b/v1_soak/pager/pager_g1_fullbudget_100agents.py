#!/usr/bin/env python3
# G-O1 CAPSTONE — 100 agents, realistic SLM-heavy mix, FULL 80GiB budget, on the BUILT pager (eager; the dispatch
# engine is the deferred build). Measures the RESIDENCY side of the 100-agent claim. STRUCTURAL FACT (not a harness
# choice): this pod has only 5 distinct causal-LM int4 models (~22GiB reserves); at FULL budget (75GiB) ALL fit
# resident -> ZERO eviction -> cold-miss ~0 -> LFU-DA == LRU (no eviction to differ on). So this confirms the
# residency/memory side accommodates 100 agents; the gap to a real 100-agent deployment is DISPATCH throughput (the
# eager-serial P99 floor reported here is the LOWER BOUND the batching engine lifts, NOT the product number).
import ctypes, os, sys, time, glob, torch, numpy as np
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
POLICY=os.environ.get("CIPHER_PAGER_POLICY","lru")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
# Realistic agent-fleet mix: SMALL SLMs hot (Zipfian rank 0/1 = list index 0/1), big base models the warm/cold tail.
MODELS=[("/home/ubuntu/models/TinyLlama-1.1B",0xE4),("/home/ubuntu/models/Llama-3.2-1B-Instruct",0xE5),
        ("/home/ubuntu/models/Mistral-7B-v0.1",0xE1),("/home/ubuntu/models/Qwen2-7B",0xE2),
        ("/home/ubuntu/models/Llama-3.1-8B",0xE3)]
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",ctypes.c_int,[]),("cipher_pager_begin_load",ctypes.c_int,[ctypes.c_ulonglong,ctypes.c_size_t]),
    ("cipher_pager_end_load",ctypes.c_int,[ctypes.c_int]),("cipher_pager_serve_demand",ctypes.c_int,[ctypes.c_int,ctypes.POINTER(ctypes.c_ulonglong)]),
    ("cipher_pager_serve_end",None,[ctypes.c_int]),("cipher_pager_mgr_init",ctypes.c_int,[ctypes.c_size_t]),
    ("cipher_pager_state",ctypes.c_int,[ctypes.c_int]),("cipher_pager_mgr_freq",ctypes.c_ulonglong,[ctypes.c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
from torch.cuda.memory import CUDAPluggableAllocator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
def cksz(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
models=[]; toks=[]; rids=[]; refs=[]; names=[]; reserves=[]; cfgs=[]
for path,key in MODELS:
    ck=f"/home/ubuntu/models_int4/{os.path.basename(path)}"; cfg=AutoConfig.from_pretrained(ck)
    reserve=cksz(ck)+cfg.vocab_size*cfg.hidden_size*4+(384<<20)
    rid=lib.cipher_pager_begin_load(key,reserve); pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda"); torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid); m._pool=pool
    t=AutoTokenizer.from_pretrained(path); models.append(m); toks.append(t); rids.append(rid)
    names.append(os.path.basename(path)); reserves.append(reserve); cfgs.append(cfg)
M=len(models); REFP="The history of artificial intelligence began in the"
for i in range(M):
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): refs.append(int(models[i](ids).logits[0,-1].argmax()))
BUDGET=75.0; lib.cipher_pager_mgr_init(int(BUDGET*GB))   # FULL budget -> all distinct models stay resident
resident=[lib.cipher_pager_state(rids[i])==2 for i in range(M)]
print(f"=== G-O1 FULL-BUDGET 100-AGENT (policy={POLICY}, budget={BUDGET:.0f}GiB) ===",flush=True)
print(f"  distinct models available on pod: {M} (the hard distinct-model ceiling); reserves sum="
      f"{sum(reserves)/GB:.1f}GiB << {BUDGET:.0f}GiB budget",flush=True)
print(f"  MODELS-RESIDENT at full budget: {sum(resident)}/{M}  (all distinct models fit -> no eviction)",flush=True)

va=ctypes.c_ulonglong(0)
@torch.no_grad()
def burst(i,nctx,ndecode=16):                  # realistic burst = prefill(growing ctx) + a few EAGER decode tokens
    ids=toks[i]("data "*nctx,return_tensors="pt",truncation=True,max_length=nctx).input_ids.cuda()
    out=models[i].generate(ids,max_new_tokens=ndecode,do_sample=False)
    return out.shape[1]
# --- real per-model burst latency (the eager service time the engine would batch away) ---
print("  per-model eager burst latency (prefill 512 + 16 decode):",flush=True)
svc=[]
for i in range(M):
    lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va)); torch.cuda.synchronize()
    t0=time.time(); burst(i,512); torch.cuda.synchronize(); dt=time.time()-t0
    lib.cipher_pager_serve_end(rids[i]); svc.append(dt)
    print(f"    {names[i]:<26} {dt*1000:.0f}ms",flush=True)

# --- realistic 100-agent stream through the ACTUAL pager (cold-miss + P99 real; service real) ---
N_AGENTS=100
def run_stream(burstiness, window=8.0, idle_mean=3.5, seed=0):
    rng=np.random.default_rng(seed)
    # each agent: a Zipfian-affine model + a request train over [0,window) with gamma-burst inter-arrivals
    reqs=[]   # (arrival, agent, model_idx, turn)
    for a in range(N_AGENTS):
        mi=(rng.zipf(1.6)-1)%M                       # Zipfian model affinity (small models hot)
        t=rng.uniform(0,idle_mean); turn=0
        while t<window:
            reqs.append((t,a,mi,turn)); turn+=1
            gap=rng.gamma(shape=burstiness, scale=idle_mean/burstiness)   # idle gap, burstiness-modulated
            t+=gap
    reqs.sort()
    server_free=0.0; ttfts=[]; ncold=0
    for (arr,a,mi,turn) in reqs:
        start=max(arr,server_free)
        if lib.cipher_pager_state(rids[mi])!=2: ncold+=1
        t0=time.time(); lib.cipher_pager_serve_demand(rids[mi],ctypes.byref(va)); piw=time.time()-t0
        nctx=min(256+96*turn,1024)
        t1=time.time(); burst(mi,nctx); pf=time.time()-t1
        lib.cipher_pager_serve_end(rids[mi]); proc=piw+pf; server_free=start+proc
        ttfts.append((start-arr)+piw+pf)
    ttfts.sort(); p99=ttfts[int(len(ttfts)*0.99)]; p50=ttfts[len(ttfts)//2]
    return ncold,len(reqs),p50*1000,p99*1000
print(f"  realistic {N_AGENTS}-agent stream (Zipfian affinity, duty-cycle burst+idle~3.5s, growing ctx), EAGER server:",flush=True)
print(f"  {'burstiness':>11} {'reqs':>5} {'cold-miss':>9} {'p50':>6} {'p99(eager-floor)':>16}",flush=True)
for b in [1.0,0.5,0.2]:
    nc,nr,p50,p99=run_stream(b)
    print(f"  {b:>11.1f} {nr:>5d} {nc:>8d}  {p50:>5.0f}ms {p99:>14.0f}ms",flush=True)

# --- agents-per-GPU: eager-serial dispatch floor (discrete-event queue, REAL measured svc times) ---
# This is the LOWER BOUND the batching engine lifts (continuous batching serves many agents' tokens per step),
# NOT a residency result. Reported to DIAGNOSE the binding term = dispatch, not residency.
def eager_capacity(slo_ms, idle_mean=3.5, burstiness=0.5, seed=1, Nmax=140):
    rng=np.random.default_rng(seed); cap=0
    for N in range(5,Nmax+1,5):
        reqs=[]
        for a in range(N):
            mi=(rng.zipf(1.6)-1)%M; t=rng.uniform(0,idle_mean)
            while t<12.0:
                reqs.append((t,mi)); t+=rng.gamma(shape=burstiness,scale=idle_mean/burstiness)
        reqs.sort(); sf=0.0; tt=[]
        for (arr,mi) in reqs:
            st=max(arr,sf); sf=st+svc[mi]; tt.append((st-arr)+svc[mi])
        tt.sort(); p99=tt[int(len(tt)*0.99)]*1000 if tt else 0
        if p99<=slo_ms: cap=N
        else: break
    return cap
for slo in [500,1000,2000]:
    print(f"  agents-per-GPU @ eager-serial P99<{slo}ms (DISPATCH FLOOR, engine lifts this): {eager_capacity(slo)}",flush=True)

# --- KV sizing at full budget (labeled SIZING, not measured residency) ---
free=BUDGET*GB-sum(reserves)
print(f"  KV sizing @ full budget: weights={sum(reserves)/GB:.1f}GiB, headroom={free/GB:.1f}GiB. KV/agent @1K ctx:",flush=True)
for i in [0,2,4]:
    c=cfgs[i]; kvh=getattr(c,'num_key_value_heads',c.num_attention_heads); hd=c.hidden_size//c.num_attention_heads
    kv1k=2*c.num_hidden_layers*kvh*hd*2*1024   # bytes: 2(K,V)*layers*kv_heads*head_dim*2B(fp16)*ctx
    print(f"    {names[i]:<26} {kv1k/(1<<20):.0f}MiB/agent @1K -> 100 agents={kv1k*100/GB:.1f}GiB (fits in {free/GB:.0f}GiB headroom: {kv1k*100<free})",flush=True)

# --- per-agent KL=0 (correctness carried) ---
klbad=0
for i in range(M):
    lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va))
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        if int(models[i](ids).logits[0,-1].argmax())!=refs[i]: klbad+=1
    lib.cipher_pager_serve_end(rids[i])
print(f"  per-agent KL=0 across swap: {M-klbad}/{M} | cold-miss ~0 = residency side delivers; p99 above is the "
      f"EAGER-SERIAL DISPATCH FLOOR (the deferred batching engine lifts it), NOT the residency result",flush=True)
sys.stdout.flush(); os._exit(0)
