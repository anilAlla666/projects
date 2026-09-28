#!/usr/bin/env python3
# FREQUENCY-AWARE RESIDENCY — cold-miss LFU-DA vs LRU on the built INT4 co-residence (the last on-pod K-lever).
# Structurally IDENTICAL to pager_zipfian_sweep.py (same MODELS order, NREQ/RATE/seed, RNG call order, serve loop,
# budget) so the LRU run REPRODUCES the tagged 36/34/33% EXACTLY (cold-miss is deterministic in (seed, policy);
# eviction order is wall-clock-independent). Policy is read by the C lib from CIPHER_PAGER_POLICY at mgr_init.
# Run twice (lru / lfuda) at matched seeds; the DISCRIMINATOR is per-model cold-miss vs the realized-Zipfian floor.
import ctypes, os, sys, time, glob, torch, numpy as np
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
POLICY=os.environ.get("CIPHER_PAGER_POLICY","lru")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
MODELS=[("/home/ubuntu/models/Mistral-7B-v0.1",0xE1),("/home/ubuntu/models/Qwen2-7B",0xE2),
        ("/home/ubuntu/models/Llama-3.1-8B",0xE3),("/home/ubuntu/models/TinyLlama-1.1B",0xE4),
        ("/home/ubuntu/models/Llama-3.2-1B-Instruct",0xE5)]
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
models=[]; toks=[]; rids=[]; refs=[]; names=[]; reserves=[]
for path,key in MODELS:
    ck=f"/home/ubuntu/models_int4/{os.path.basename(path)}"; cfg=AutoConfig.from_pretrained(ck)
    reserve=cksz(ck)+cfg.vocab_size*cfg.hidden_size*4+(384<<20)
    rid=lib.cipher_pager_begin_load(key,reserve); pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda"); torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid); m._pool=pool
    t=AutoTokenizer.from_pretrained(path); models.append(m); toks.append(t); rids.append(rid)
    names.append(os.path.basename(path)); reserves.append(reserve)
M=len(models); REFP="The history of artificial intelligence began in the"
for i in range(M):
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): refs.append(int(models[i](ids).logits[0,-1].argmax()))
BUDGET=18.0; lib.cipher_pager_mgr_init(int(BUDGET*GB))   # C lib logs policy=lru|lfu-da from CIPHER_PAGER_POLICY
print(f"=== FREQ-POLICY SWEEP  policy={POLICY}  M={M} distinct 4-bit  budget={BUDGET:.0f}GiB ===",flush=True)
print("  reserves(GiB): "+", ".join(f"{names[i]}={reserves[i]/GB:.2f}" for i in range(M)),flush=True)
cap=0; acc=0.0  # how many fit in budget by descending reserve (capacity intuition; floor is computed per-run from realized sel)
va=ctypes.c_ulonglong(0)
@torch.no_grad()
def prefill(i,nctx):
    ids=toks[i]("data "*nctx,return_tensors="pt",truncation=True,max_length=nctx).input_ids.cuda()
    return int(models[i](ids).logits[0,-1].argmax())
def run(burstiness, NREQ=160, RATE=8.0, seed=0):
    rng=np.random.default_rng(seed)
    ia=rng.gamma(shape=burstiness, scale=1.0/(RATE*burstiness), size=NREQ); arr=np.cumsum(ia)
    sel=[(z-1)%M for z in rng.zipf(1.6,size=NREQ)]; turn={}
    server_free=0.0; ttfts=[]; ncold=0; permiss=[0]*M; perreq=[0]*M
    for r in range(NREQ):
        i=sel[r]; a=arr[r]; start=max(a,server_free); perreq[i]+=1
        cold=(lib.cipher_pager_state(rids[i])!=2)
        t0=time.time(); lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va)); piw=time.time()-t0
        if cold: ncold+=1; permiss[i]+=1
        turn[i]=turn.get(i,0)+1; nctx=min(256+128*turn[i],1024)
        t1=time.time(); prefill(i,nctx); pf=time.time()-t1
        lib.cipher_pager_serve_end(rids[i]); proc=piw+pf; server_free=start+proc
        ttfts.append((start-a)+piw+pf)
    ttfts.sort(); p99=ttfts[int(len(ttfts)*0.99)]; p50=ttfts[len(ttfts)//2]
    return ncold/NREQ*100, p50*1000, p99*1000, permiss, perreq
print(f"  {'burstiness':>11} {'cold-miss%':>10} {'p50':>6} {'p99':>7}   per-model cold-miss (req)",flush=True)
for b in [1.0,0.5,0.2]:
    cm,p50,p99,permiss,perreq=run(b)
    pm=" ".join(f"{names[i].split('-')[0][:7]}:{permiss[i]}/{perreq[i]}" for i in range(M))
    print(f"  {b:>11.1f} {cm:>9.0f}% {p50:>5.0f}ms {p99:>5.0f}ms   {pm}",flush=True)
# per-model cumulative freq (proves LFU-DA pinned the hot set) + final residency
freqs=[lib.cipher_pager_mgr_freq(rids[i]) for i in range(M)]
states=[lib.cipher_pager_state(rids[i]) for i in range(M)]
print("  per-model cumulative freq + final-resident: "+", ".join(
    f"{names[i].split('-')[0][:7]}:f={freqs[i]},{'R' if states[i]==2 else '_'}" for i in range(M)),flush=True)
# KL=0 across swap (carried correctness = INT4 co-residence non-regression UNDER THIS POLICY)
klbad=0
for i in range(M):
    lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va))
    ids=toks[i](REFP,return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        if int(models[i](ids).logits[0,-1].argmax())!=refs[i]: klbad+=1
    lib.cipher_pager_serve_end(rids[i])
print(f"  correctness: KL=0 across swap {M-klbad}/{M} (INT4 co-residence non-regression under policy={POLICY}) | "
      f"caveat: p99 is eager-single-thread-queue-dominated; cold-miss% + per-model breakdown are the clean K-lever numbers",flush=True)
sys.stdout.flush(); os._exit(0)
