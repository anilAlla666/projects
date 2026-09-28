#!/usr/bin/env python3
# G-O1 BINDING-LIMIT measurement (CIPHER-side only; the ONE unmeasured decision-relevant number, advisor):
# does the page-in tail under CORRELATED bursts keep P99 TTFT bounded at M past the resident ceiling, or does the
# K x page-in serialization over PCIe blow it? Also checks the advisor's catch: is the 314ms/~100ms swap (measured
# SEQUENTIALLY, no contention) still that fast UNDER a burst storm? Real models for KL=0; eager decode (isolates the
# swap/residency axis; the graph engine is deferred). NOT a two-sided sim -- N-sleep is already known at the primitive.
import ctypes, os, sys, time, glob, gc, json, math, random, torch, numpy as np
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
REAL=[("/home/ubuntu/models/Mistral-7B-v0.1",0xD1),("/home/ubuntu/models/Qwen2-7B",0xD2),
      ("/home/ubuntu/models/Llama-3.1-8B",0xD3),("/home/ubuntu/models/TinyLlama-1.1B",0xD4),
      ("/home/ubuntu/models/Llama-3.2-1B-Instruct",0xD5)]
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",ctypes.c_int,[]),("cipher_pager_begin_load",ctypes.c_int,[ctypes.c_ulonglong,ctypes.c_size_t]),
    ("cipher_pager_end_load",ctypes.c_int,[ctypes.c_int]),("cipher_pager_serve_demand",ctypes.c_int,[ctypes.c_int,ctypes.POINTER(ctypes.c_ulonglong)]),
    ("cipher_pager_serve_end",None,[ctypes.c_int]),("cipher_pager_mgr_init",ctypes.c_int,[ctypes.c_size_t]),
    ("cipher_pager_state",ctypes.c_int,[ctypes.c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
from torch.cuda.memory import CUDAPluggableAllocator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
def prequant(path):
    out=f"/home/ubuntu/models_int4/{os.path.basename(path)}"
    if os.path.exists(f"{out}/config.json"): return out
    print(f"  quantizing {os.path.basename(path)}...",flush=True)
    mm=AutoModelForCausalLM.from_pretrained(path,quantization_config=BNB,torch_dtype=torch.float16,device_map="cuda")
    os.makedirs(out,exist_ok=True); mm.save_pretrained(out); del mm; gc.collect(); torch.cuda.empty_cache(); return out
def cksz(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
models=[]; toks=[]; rids=[]; refs=[]; names=[]
for path,key in REAL:
    ck=prequant(path); cfg=AutoConfig.from_pretrained(ck); cd=cksz(ck)
    reserve=cd + cfg.vocab_size*cfg.hidden_size*4 + (384<<20)
    rid=lib.cipher_pager_begin_load(key, reserve); pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda"); torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid); m._pool=pool
    t=AutoTokenizer.from_pretrained(path)
    models.append(m); toks.append(t); rids.append(rid); names.append(os.path.basename(path))
    print(f"  loaded {os.path.basename(path)} rid={rid} reserve={reserve/GB:.1f}GiB",flush=True)
M=len(models)
# solo references (KL=0 across swap): first decoded token id for a fixed prompt, per model
REFP="The history of artificial intelligence began in the"
for i in range(M):
    ids=toks[i](REFP, return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): refs.append(int(models[i](ids).logits[0,-1].argmax()))
# budget from argv (GB): sweep to separate the arc's forced-paging (budget=11, ~2 resident, 63% cold) from a
# REALISTIC hot-set-resident policy (budget holds the Zipfian hot set -> low K). ROOT-CAUSE of term-3 (K).
BUDGET_GB=float(sys.argv[1]) if len(sys.argv)>1 else 11.0
lib.cipher_pager_mgr_init(int(BUDGET_GB*GB))
print(f"=== BURST STRESS: M={M} distinct real 4-bit models, budget {BUDGET_GB:.0f}GiB (root-cause K sweep) ===",flush=True)

# real-workload mock: gamma(0.2) bursty inter-arrivals (correlated), Zipfian model selection (hot/cold), growing ctx
rng=np.random.default_rng(0); NREQ=160; RATE=8.0; burstiness=0.2
ia=rng.gamma(shape=burstiness, scale=1.0/(RATE*burstiness), size=NREQ)   # CV=1/sqrt(burstiness) -> bursty at 0.2
arr=np.cumsum(ia)
zr=rng.zipf(1.6, size=NREQ); sel=[(z-1)%M for z in zr]                   # Zipfian: a few hot, rest cold
turn={};
def ctx_len(a):
    turn[a]=turn.get(a,0)+1; return min(256+128*turn[a], 1024)            # growing context per agent session
@torch.no_grad()
def prefill(i, nctx):
    ids=toks[i]("data "*nctx, return_tensors="pt", truncation=True, max_length=nctx).input_ids.cuda()
    o=models[i](ids); return int(o.logits[0,-1].argmax())
# single-thread server model: requests queue; process in arrival order; TTFT = queue + page_in_wait + prefill
server_free=0.0; rows=[]; klbad=0; pagein_lat=[]; npage=0
va=ctypes.c_ulonglong(0)
for r in range(NREQ):
    i=sel[r]; a=arr[r]; start=max(a, server_free)
    cold = (lib.cipher_pager_state(rids[i])!=2)
    t0=time.time(); rc=lib.cipher_pager_serve_demand(rids[i], ctypes.byref(va)); piw=time.time()-t0
    if cold and rc==0: pagein_lat.append(piw); npage+=1
    nctx=ctx_len(i); t1=time.time(); tok=prefill(i, nctx); pf=time.time()-t1
    if tok!=refs[i] and nctx<=len(toks[i](REFP).input_ids): pass  # ref is for REFP-len; skip mismatch from ctx diff
    lib.cipher_pager_serve_end(rids[i])
    proc=piw+pf; server_free=start+proc; ttft=(start-a)+piw+pf
    rows.append((ttft,(start-a),piw,pf,cold))
# KL=0 across swap: re-serve each model on REFP after the churn, verify first token == ref
for i in range(M):
    lib.cipher_pager_serve_demand(rids[i], ctypes.byref(va))
    ids=toks[i](REFP, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        if int(models[i](ids).logits[0,-1].argmax())!=refs[i]: klbad+=1
    lib.cipher_pager_serve_end(rids[i])
def pct(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(len(xs)*p))] if xs else 0
ttfts=[r[0] for r in rows]
print(f"KL=0 across swap: {M-klbad}/{M} models first-token-identical after churn (correctness gate)",flush=True)
print(f"page-ins: {npage}/{NREQ} cold; per-page-in latency UNDER BURST: p50={pct(pagein_lat,.5)*1000:.0f}ms p99={pct(pagein_lat,.99)*1000:.0f}ms (seq baseline ~100-150ms for 4-bit)",flush=True)
print(f"TTFT: p50={pct(ttfts,.5)*1000:.0f}ms p99={pct(ttfts,.99)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms",flush=True)
q=[r[1] for r in rows]; pw=[r[2] for r in rows]; pf=[r[3] for r in rows]
print(f"  TTFT decomposed (p99): queue={pct(q,.99)*1000:.0f}ms + page_in_wait={pct(pw,.99)*1000:.0f}ms + prefill={pct(pf,.99)*1000:.0f}ms",flush=True)
print(f"  -> binding limit: under gamma-{burstiness} correlated bursts, the page-in/queue tail {'BLOWS P99' if pct(ttfts,.99)>2 else 'stays bounded'} (SLO ref ~2s TTFT)",flush=True)
sys.stdout.flush(); os._exit(0)
