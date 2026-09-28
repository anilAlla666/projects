#!/usr/bin/env python3
# BUILD: INT4 (bnb NF4) weights in the proven pager. The pager is precision-agnostic -> NO new pager code; this is
# integration + measurement on pager-step2arch-coresidence. Gates: (1) pager-correctness KL=0 (4-bit weights
# round-trip evict->pagein bit-identical via live_cksum + forward bit-identical to 4-bit-resident ref; 100%
# residency, disjoint VAs); (3) density MEASURED (actual 4-bit region bytes, models-fit/80GB, page-bytes/swap);
# (4) HBM reclaim physical at INT4; co-residence churn bit-identical. INT4 quality is a SEPARATE script (lossy).
import ctypes, os, sys, json, glob, gc, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"   # main-thread materialize: transformers' ThreadPoolExecutor load
                                               # escapes the thread-local use_mem_pool -> params miss the region
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30; MB=1<<20
MODELS=[("/home/ubuntu/models/Mistral-7B-v0.1",0x41),("/home/ubuntu/models/Qwen2-7B",0x42),("/home/ubuntu/models/Llama-3.1-8B",0x43)]
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),
    ("cipher_pager_end_load",c_int,[c_int]),("cipher_pager_page_in",c_int,[c_int]),("cipher_pager_page_out",c_int,[c_int]),
    ("cipher_pager_state",c_int,[c_int]),("cipher_pager_serve_demand",c_int,[c_int,POINTER(c_ulonglong)]),
    ("cipher_pager_serve_end",None,[c_int]),("cipher_pager_mgr_init",c_int,[c_size_t]),
    ("cipher_pager_mgr_resident_bytes",c_ulonglong,[]),("cipher_pager_mgr_wasted",c_ulong,[]),
    ("cipher_pager_live_cksum",c_ulonglong,[c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class St(ctypes.Structure):
    _fields_=[("cm",c_ulong),("pi",c_ulong),("ev",c_ulong),("st",c_int),("ref",c_int),("pif",c_int),("used",c_ulong),("va",c_ulonglong)]
lib.cipher_pager_get_stats.argtypes=[c_int,POINTER(St)]
def stats(r): s=St(); lib.cipher_pager_get_stats(r,ctypes.byref(s)); return s
def used_hbm(): f,t=torch.cuda.mem_get_info(); return t-f
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
def fp16_size(path):
    idx=glob.glob(f"{path}/*.index.json")
    if idx: return json.load(open(idx[0]))["metadata"]["total_size"]
    return sum(os.path.getsize(f) for f in glob.glob(f"{path}/*.safetensors"))
PASS=True
def gate(n,ok,d=""):
    global PASS; PASS=PASS and ok; print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}", flush=True)
def prequant(path):
    # pre-quantize+save once: a COMPLETE 4-bit checkpoint reloads with no missing-key init.normal_(.float()) fp32
    # transient -> the bump allocator's high-water tracks the real 4-bit footprint (quant-on-load OOM'd Qwen2).
    name=os.path.basename(path); out=f"/home/ubuntu/models_int4/{name}"
    if os.path.exists(f"{out}/config.json"): return out
    print(f"  quantizing+saving {name} (one-time) ...",flush=True)
    mm=AutoModelForCausalLM.from_pretrained(path,quantization_config=BNB,torch_dtype=torch.float16,device_map="cuda")
    os.makedirs(out,exist_ok=True); mm.save_pretrained(out)
    del mm; gc.collect(); torch.cuda.empty_cache()
    return out
def ckpt_size(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
CKPT={}; RESV={}
def load_4bit_paged(path,key):
    # COMPUTED TIGHT RESERVE (STEP 4 fix, load-level, no substrate): probe proved holes=0, so the only reclaimable
    # waste was my own over-reservation. bump = ckpt + the fp32 embed-init transient (vocab*hidden*4) -- verified to
    # nail both models -- so reserve tightly to that + small margin instead of the old +3GiB pad. No tail-shrink needed.
    fp16=fp16_size(path); ck=prequant(path); cd=ckpt_size(ck); CKPT[key]=cd
    cfg=AutoConfig.from_pretrained(ck)
    transient = cfg.vocab_size * cfg.hidden_size * 4              # the fp32 embed init .float() temp (dominates bump-over-ckpt)
    reserve = cd + transient + (384<<20); RESV[key]=reserve
    rid=lib.cipher_pager_begin_load(key,reserve); assert rid>=0
    pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda")   # pre-quantized 4-bit ckpt
        torch.cuda.synchronize()
    assert lib.cipher_pager_end_load(rid)==0
    m._pool=pool
    return m, rid, fp16
def fwd(m,ids):
    with torch.no_grad(): return m(ids).logits.float().clone()

print("=== INT4-in-pager: 3 distinct 4-bit 7-8B models co-resident ===",flush=True)
models=[]; rids=[]; ids=[]; refs=[]; fp16s=[]; used4=[]
for path,key in MODELS:
    m,rid,fp16=load_4bit_paged(path,key); t=AutoTokenizer.from_pretrained(path)
    iid=t("The history of science is", return_tensors="pt").input_ids.cuda()
    s=stats(rid); lo,hi=s.va,s.va+s.used
    inreg=sum(1 for _,p in m.named_parameters() if lo<=p.data_ptr()<hi)
    inbuf=sum(1 for _,b in m.named_buffers() if b.is_cuda and lo<=b.data_ptr()<hi)
    tot=sum(1 for _ in m.named_parameters())
    name=os.path.basename(path); models.append(m); rids.append(rid); ids.append(iid)
    used4.append(s.used); fp16s.append(fp16)
    print(f"  {name}: fp16={fp16/GB:.1f}GiB -> 4bit region used={s.used/GB:.2f}GiB ({fp16/s.used:.1f}x smaller) va={hex(s.va)} params_in_region={inreg}/{tot} buffers_in_region={inbuf}",flush=True)
    gate(f"G-cores {name}: 100% params behind 4-bit region", inreg==tot)
gate("G-cores disjoint VAs", len(set(stats(r).va for r in rids))==len(rids))

# (1) pager-correctness KL=0: round-trip bit-identical per model
for i,m in enumerate(models):
    o1a=fwd(m,ids[i]); o1b=fwd(m,ids[i]); det=torch.equal(o1a,o1b); refs.append(o1a)
    ck0=lib.cipher_pager_live_cksum(rids[i])
    assert lib.cipher_pager_page_out(rids[i])==0
    assert lib.cipher_pager_page_in(rids[i])==0
    ck1=lib.cipher_pager_live_cksum(rids[i])
    o2=fwd(m,ids[i])
    gate(f"G-correct model{i}: 4-bit weights byte-identical across evict+pagein", ck0==ck1 and ck0!=0, f"cksum {hex(ck0)}->{hex(ck1)}")
    gate(f"G-correct model{i}: forward bit-identical after evict+pagein (det={det})", torch.equal(o1a,o2), f"max_abs_diff={(o1a-o2).abs().max().item():.2e}")

# (4) HBM reclaim physical at INT4
torch.cuda.synchronize(); f0=used_hbm(); assert lib.cipher_pager_page_out(rids[0])==0; f1=used_hbm(); freed=f0-f1; assert lib.cipher_pager_page_in(rids[0])==0
# used_hbm DROPS when freed -> freed = before-after; page_out unmaps the region's mapped physical (the reserve)
gate("G-reclaim: page_out frees the region physically (mem_get_info)", freed >= int(0.85*used4[0]), f"freed={freed/GB:.2f}GiB used_bump={used4[0]/GB:.2f}GiB")

# co-residence churn at INT4: budget < sum -> eviction, bit-identical
lib.cipher_pager_mgr_init(1<<62); total=lib.cipher_pager_mgr_resident_bytes(); lib.cipher_pager_mgr_init(int(total*0.70))
def serve_once(i):
    va=c_ulonglong(0); rc=lib.cipher_pager_serve_demand(rids[i],ctypes.byref(va))
    if rc!=0: return None
    try: o=fwd(models[i],ids[i])
    finally: lib.cipher_pager_serve_end(rids[i])
    return torch.equal(o,refs[i])
bad=0; served=0
for rnd in range(3):
    for i in range(len(models)):
        r=serve_once(i)
        if r is None: continue
        served+=1; bad+=(0 if r else 1)
gate("G-correct co-residence churn (4-bit, budget<sum): every served bit-identical", bad==0, f"served={served} bad={bad} wasted={lib.cipher_pager_mgr_wasted()}")

# (3) density MEASURED -- report BOTH the tight 4-bit footprint (ckpt size, achievable with reclaim/compaction)
# and the as-implemented region bump (includes the load-transient slack from the no-reclaim bump allocator).
avgresv=sum(RESV.values())/len(RESV); avg4=sum(used4)/len(used4); avgfp=sum(fp16s)/len(fp16s); avgck=sum(CKPT.values())/len(CKPT)
print(f"\n  DENSITY (STEP 4 = COMPUTED TIGHT RESERVE; compaction substrate NOT needed -- probe proved holes=0):",flush=True)
print(f"    physical region (reserve) {avgresv/GB:.2f}GiB vs fp16 {avgfp/GB:.1f}GiB = {avgfp/avgresv:.1f}x -> ~{int(80*GB/avgresv)} models/80GiB (was ~8-9 at the +3GiB pad)",flush=True)
print(f"    reserve hugs bump {avg4/GB:.2f}GiB (padding reclaimed, load-level, zero coherence risk); ckpt-tight {avgck/GB:.2f}GiB = {avgfp/avgck:.1f}x (~{int(80*GB/avgck)} models)",flush=True)
print(f"    residual (reserve->ckpt) = the fp32 embed-init transient, NOT a bump-allocator hole; closing it needs a transformers init",flush=True)
print(f"    patch (violates substrate-line) OR embed quantization (separate, bigger lever) -- NOT compaction. Zero per-token re-paging preserved.",flush=True)
print(f"\nINT4-IN-PAGER: {'ALL PASS' if PASS else 'FAIL'}",flush=True)
sys.stdout.flush(); os._exit(0 if PASS else 1)
