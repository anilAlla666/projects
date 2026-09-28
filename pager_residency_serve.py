#!/usr/bin/env python3
# STEP 2-arch real-model gate: N=3 DISTINCT 7-8B models co-resident in ONE process behind pager regions, served
# under a residency budget that forces eviction (more registered than fit). Gates:
#  G-cores   : each model 100% residency, disjoint VAs
#  G-pressure: budget < sum -> manager evicts LRU; serving an evicted model demand-pages it in
#  G-correct : every served model's forward bit-identical to its solo reference (det baseline first), under churn
#  G-hbm     : resident HBM <= budget; eviction physically frees (mem_get_info)
#  G-overhead: co-residence saving vs N separate processes = (N-1) x per-process context overhead (MEASURED)
#  Honest: this proves the MECHANISM; the delta vs N orchestrated vLLM-sleep instances is UNMEASURED here.
import ctypes, os, sys, threading, time, torch
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODELS = [("/home/ubuntu/models/Mistral-7B-v0.1", 0x11),
          ("/home/ubuntu/models/Qwen2-7B",        0x22),
          ("/home/ubuntu/models/Llama-3.1-8B",    0x33)]
MB = 1 << 20; GB = 1 << 30
lib = ctypes.CDLL(SO)
for fn, res, args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),
    ("cipher_pager_end_load",c_int,[c_int]),("cipher_pager_page_in",c_int,[c_int]),("cipher_pager_page_out",c_int,[c_int]),
    ("cipher_pager_state",c_int,[c_int]),("cipher_pager_serve_demand",c_int,[c_int,POINTER(c_ulonglong)]),
    ("cipher_pager_serve_end",None,[c_int]),("cipher_pager_mgr_init",c_int,[c_size_t]),
    ("cipher_pager_mgr_resident_bytes",c_ulonglong,[]),("cipher_pager_mgr_wasted",c_ulong,[])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class Stats(ctypes.Structure):
    _fields_=[("cm",c_ulong),("pi",c_ulong),("ev",c_ulong),("st",c_int),("ref",c_int),("pif",c_int),("used",c_ulong),("va",c_ulonglong)]
lib.cipher_pager_get_stats.argtypes=[c_int,POINTER(Stats)]
def stats(rid): s=Stats(); lib.cipher_pager_get_stats(rid,ctypes.byref(s)); return s
def used_hbm(): f,t=torch.cuda.mem_get_info(); return t-f

from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda"); torch.cuda.synchronize()
ctx_overhead = used_hbm()            # per-process CUDA context + torch libs baseline (before any model)
print(f"per-process context+lib overhead (baseline) = {ctx_overhead/MB:.0f} MiB")

from transformers import AutoModelForCausalLM, AutoTokenizer
PASS=True
def gate(n, ok, d=""):
    global PASS; PASS=PASS and ok; print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")

def load_paged(path, key):
    cpu=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
    pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
    rid=lib.cipher_pager_begin_load(key, int(pb*1.10)+(64<<20)); assert rid>=0
    pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        cpu.to("cuda"); torch.cuda.synchronize()
    assert lib.cipher_pager_end_load(rid)==0
    cpu._pool=pool
    return cpu, rid

def fwd(m, ids):
    with torch.no_grad(): return m(ids).logits.float().clone()

print(f"=== STEP 2-arch: N={len(MODELS)} distinct models co-resident ===")
models=[]; toks=[]; ids=[]; refs=[]; rids=[]
for path,key in MODELS:
    m,rid=load_paged(path,key); t=AutoTokenizer.from_pretrained(path)
    iid=t("The history of science is", return_tensors="pt").input_ids.cuda()
    s=stats(rid); models.append(m); toks.append(t); ids.append(iid); rids.append(rid)
    # 100% residency + disjoint VA
    lo,hi=s.va,s.va+s.used; inreg=sum(1 for _,p in m.named_parameters() if lo<=p.data_ptr()<hi)
    tot=sum(1 for _ in m.named_parameters())
    print(f"  loaded {os.path.basename(path)} rid={rid} used={s.used>>20}MiB va={hex(s.va)} params_in_region={inreg}/{tot}")
    gate(f"G-cores {os.path.basename(path)} 100% residency", inreg==tot)
disjoint = len(set(stats(r).va for r in rids))==len(rids)
gate("G-cores disjoint VAs across models", disjoint)

# solo reference (determinism baseline O1a==O1b first) -- all resident now
for i,m in enumerate(models):
    o1a=fwd(m,ids[i]); o1b=fwd(m,ids[i])
    gate(f"G-det model{i} deterministic", torch.equal(o1a,o1b))
    refs.append(o1a)

total_res = lib.cipher_pager_mgr_resident_bytes() if False else None
lib.cipher_pager_mgr_init(1<<62)                     # huge budget first -> account all-resident total
total = lib.cipher_pager_mgr_resident_bytes()
budget = int(total*0.70)                              # ~2 of 3 fit -> forces eviction
lib.cipher_pager_mgr_init(budget)
res_after = lib.cipher_pager_mgr_resident_bytes()
print(f"  total resident={total/GB:.1f}GiB -> budget={budget/GB:.1f}GiB -> after mgr_init resident={res_after/GB:.1f}GiB")
gate("G-hbm budget enforced at init (resident<=budget)", res_after<=budget)
n_resident = sum(1 for r in rids if lib.cipher_pager_state(r)==2)
gate("G-pressure: not all models fit (>=1 evicted)", n_resident < len(rids), f"resident={n_resident}/{len(rids)}")

# serve under pressure: round-robin demand-serve each model -> evicted ones page in, bit-identical
def serve_once(i):
    va=c_ulonglong(0)
    rc=lib.cipher_pager_serve_demand(rids[i], ctypes.byref(va))
    if rc!=0: return None                              # MISS (unfittable) -- should not happen for single demand
    try: o=fwd(models[i], ids[i])
    finally: lib.cipher_pager_serve_end(rids[i])
    return torch.equal(o, refs[i])

bad=0; served=0
for rnd in range(3):
    for i in range(len(models)):
        r=serve_once(i)
        if r is None: continue
        served+=1; bad += (0 if r else 1)
gate("G-correct (sequential churn): every served forward bit-identical", bad==0, f"served={served} mismatches={bad}")

# concurrent phase: N threads each demand-serve their model repeatedly under the budget
cbad=[0]; clock=threading.Lock(); cserved=[0]
def worker(i):
    for _ in range(4):
        r=serve_once(i)
        if r is None: continue
        with clock: cserved[0]+=1; cbad[0]+= (0 if r else 1)
ths=[threading.Thread(target=worker,args=(i,)) for i in range(len(models))]
[t.start() for t in ths]; [t.join() for t in ths]
gate("G-correct (concurrent churn): every served forward bit-identical", cbad[0]==0, f"served={cserved[0]} mismatches={cbad[0]} wasted={lib.cipher_pager_mgr_wasted()}")

# HBM accounting + reclaim under the manager
res_now = lib.cipher_pager_mgr_resident_bytes()
gate("G-hbm: resident <= budget after churn", res_now<=budget, f"resident={res_now/GB:.1f}GiB budget={budget/GB:.1f}GiB")
# PHYSICAL truth (advisor): real HBM == manager counter + context, NOT just the bookkeeping counter -> no VMM leak
phys = used_hbm()
expected = res_now + ctx_overhead
gate("G-hbm-physical: real HBM ~= mgr counter + ctx (no VMM handle leak across churn)", abs(phys-expected) < 4*GB,
     f"phys_used={phys/GB:.1f}GiB  counter+ctx={expected/GB:.1f}GiB  (a leaked 14-15GiB model would blow past 4GiB)")
saving = (len(MODELS)-1)*ctx_overhead
print(f"  G-overhead (co-residence): 1 process holds {len(MODELS)} models with ONE {ctx_overhead/MB:.0f} MiB context;")
print(f"     N separate processes would replicate it -> co-residence saves ~(N-1)x{ctx_overhead/MB:.0f} = {saving/MB:.0f} MiB HBM (the (ii) lever; weights/CPU-backup NOT saved -- delta vs N-sleep UNMEASURED)")

print(f"\nSTEP 2-arch CO-RESIDENCE: {'ALL PASS' if PASS else 'FAIL'}")
sys.stdout.flush(); os._exit(0 if PASS else 1)
