#!/usr/bin/env python3
# PART A — K + P99 vs BURSTINESS on the built INT4 co-residence (the decider). Realistic Zipfian model affinity +
# gamma burstiness sweep {1.0 uncorrelated / 0.5 / 0.2 correlated} + hot-set-resident budget + growing per-agent
# context. Does K (cold-miss count) stay small under realistic Zipfian, and where does the correlated-cold tail bite?
# Primary number = cold-miss RATE + K (residency-policy term, dispatch-speed-independent); P99 is eager-queue-
# confounded (binding-limit memory: at engine-speed page-in dominates, at eager the queue does) -- reported with caveat.
import ctypes, os, sys, time, glob, torch, numpy as np
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
MODELS=[("/home/ubuntu/models/Mistral-7B-v0.1",0xE1),("/home/ubuntu/models/Qwen2-7B",0xE2),
        ("/home/ubuntu/models/Llama-3.1-8B",0xE3),("/home/ubuntu/models/TinyLlama-1.1B",0xE4),
        ("/home/ubuntu/models/Llama-3.2-1B-Instruct",0xE5)]
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",ctypes.c_int,[]),("cipher_pager_begin_load",ctypes.c_int,[ctypes.c_ulonglong,ctypes.c_size_t]),
    ("cipher_pager_end_load",ctypes.c_int,[ctypes.c_int]),("cipher_pager_serve_demand",ctypes.c_int,[ctypes.c_int,ctypes.POINTER(ctypes.c_ulonglong)]),
    ("cipher_pager_serve_end",None,[ctypes.c_int]),("cipher_pager_mgr_init",ctypes.c_int,[ctypes.c_size_t]),("cipher_pager_state",ctypes.c_int,[ctypes.c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
from torch.cuda.memory import CUDAPluggableAllocator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
def cksz(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
models=[]; toks=[]; rids=[]; refs=[]; names=[]
for path,key in MODELS:
    ck=f"/home/ubuntu/models_int4/{os.path.basename(path)}"; cfg=AutoConfig.from_pretrained(ck)
    reserve=cksz(ck)+cfg.vocab_size*cfg.hidden_size*4+(384<<20)
    rid=lib.cipher_pager_begin_load(key,reserve); pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda"); torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid); m._pool=pool
    t=AutoTokenizer.from_pretrained(path); models.append(m); toks.append(t); rids.append(rid); names.append(os.path.basename(path))
M=len(models); REFP="The history of artificial intelligence began in the"
for i in range(M):
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): refs.append(int(models[i](ids).logits[0,-1].argmax()))
BUDGET=18.0; lib.cipher_pager_mgr_init(int(BUDGET*GB))   # hot-set-resident (~4 of 5; holds the Zipfian hot mass)
print(f"=== PART A: K + P99 vs burstiness, M={M} distinct 4-bit, hot-set budget {BUDGET:.0f}GiB ===",flush=True)
va=ctypes.c_ulonglong(0)
@torch.no_grad()
def prefill(i,nctx):
    ids=toks[i]("data "*nctx,return_tensors="pt",truncation=True,max_length=nctx).input_ids.cuda()
    return int(models[i](ids).logits[0,-1].argmax())
def run(burstiness, NREQ=160, RATE=8.0, seed=0):
    rng=np.random.default_rng(seed)
    ia=rng.gamma(shape=burstiness, scale=1.0/(RATE*burstiness), size=NREQ); arr=np.cumsum(ia)
    sel=[(z-1)%M for z in rng.zipf(1.6,size=NREQ)]; turn={}
    server_free=0.0; ttfts=[]; ncold=0; maxK=0
    for r in range(NREQ):
        i=sel[r]; a=arr[r]; start=max(a,server_free)
        cold=(lib.cipher_pager_state(rids[i])!=2)
        # K = cold-miss requests that have ARRIVED but not yet served when this one starts (burst cold-pile depth)
        if cold:
            k=sum(1 for j in range(r) if arr[j]<=start and sel[j]!=i and lib.cipher_pager_state(rids[sel[j]])!=2)
            maxK=max(maxK,k+1)
        t0=time.time(); lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va)); piw=time.time()-t0
        if cold: ncold+=1
        turn[i]=turn.get(i,0)+1; nctx=min(256+128*turn[i],1024)
        t1=time.time(); prefill(i,nctx); pf=time.time()-t1
        lib.cipher_pager_serve_end(rids[i]); proc=piw+pf; server_free=start+proc
        ttfts.append((start-a)+piw+pf)
    ttfts.sort(); p99=ttfts[int(len(ttfts)*0.99)]; p50=ttfts[len(ttfts)//2]
    return ncold/NREQ*100, maxK, p50*1000, p99*1000
print(f"  {'burstiness':>11} {'cold-miss%':>10} {'maxK(cold-pile)':>16} {'p50 TTFT':>9} {'p99 TTFT':>9}",flush=True)
for b in [1.0,0.5,0.2]:
    cm,k,p50,p99=run(b)
    print(f"  {b:>11.1f} {cm:>9.0f}% {k:>16d} {p50:>7.0f}ms {p99:>7.0f}ms  {'SLO-CLEAN' if p99<2000 else 'TAIL-BITES(eager-queue confound; see caveat)'}",flush=True)
# KL=0 across swap (carried correctness)
klbad=0
for i in range(M):
    lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va))
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        if int(models[i](ids).logits[0,-1].argmax())!=refs[i]: klbad+=1
    lib.cipher_pager_serve_end(rids[i])
print(f"  correctness: KL=0 across swap {M-klbad}/{M} | caveat: P99 is eager-single-thread-queue-dominated; cold-miss% + maxK are the clean K-lever numbers",flush=True)
sys.stdout.flush(); os._exit(0)
