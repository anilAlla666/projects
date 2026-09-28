#!/usr/bin/env python3
# PROBE (mechanism-decision): measure bump vs peak_live vs cur_live vs ckpt for a vocab-heavy (Qwen2) + a normal
# (Mistral) 4-bit load. Settles which reclaim mechanism is needed:
#   waste = padding (reserve-bump) + holes (bump-cur_live).  peak_live = min reserve (transient coexistence).
#   if cur_live << bump -> big freed-transient hole (tail-shrink-to-bump can't reclaim it; need shrink-to-cur_live
#      via live-aware compaction OR avoid the transient at load).  if peak_live ~ bump -> transient coexists w/ weights.
import ctypes, os, sys, glob, torch
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
TESTS=[("/home/ubuntu/models_int4/Mistral-7B-v0.1",0x61,13.5),("/home/ubuntu/models_int4/Qwen2-7B",0x62,14.2)]
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),("cipher_pager_end_load",c_int,[c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class St(ctypes.Structure):
    _fields_=[("cm",c_ulong),("pi",c_ulong),("ev",c_ulong),("st",c_int),("ref",c_int),("pif",c_int),
              ("used",c_ulong),("va",c_ulonglong),("peak",c_ulong),("live",c_ulong)]
lib.cipher_pager_get_stats.argtypes=[c_int,POINTER(St)]
def stats(r): s=St(); lib.cipher_pager_get_stats(r,ctypes.byref(s)); return s
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM
def cksz(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
for ck,key,fp16 in TESTS:
    sz=cksz(ck); reserve=int(sz*1.25)+(3<<30)
    rid=lib.cipher_pager_begin_load(key,reserve)
    pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda"); torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid)
    s=stats(rid); name=os.path.basename(ck)
    pad=(reserve-s.used); hole=(s.used-s.live)
    print(f"{name}: ckpt={sz/GB:.2f} reserve={reserve/GB:.2f} bump={s.used/GB:.2f} peak_live={s.peak/GB:.2f} cur_live={s.live/GB:.2f}",flush=True)
    print(f"    waste: padding(reserve-bump)={pad/GB:.2f}GiB  holes(bump-live)={hole/GB:.2f}GiB",flush=True)
    print(f"    tail-shrink-to-bump -> {fp16/ (s.used/GB):.1f}x ; shrink-to-cur_live -> {fp16/(s.live/GB):.1f}x ; min-reserve(peak)={s.peak/GB:.2f}",flush=True)
    del m; import gc; gc.collect()
print("DECISION INPUT: if holes>>0 the embed-transient dominates (need shrink-to-live or load-avoid); if padding dominates, tail-shrink suffices.",flush=True)
sys.stdout.flush(); os._exit(0)
