#!/usr/bin/env python3
# Verify the fix: pre-quantize+save 4-bit, then reload the COMPLETE 4-bit checkpoint behind the pager. A complete
# checkpoint has no missing keys -> no init.normal_(.float()) fp32 transient -> the bump allocator's high-water
# tracks the real 4-bit footprint (not the quant-load transient peak that OOM'd Qwen2).
import ctypes, os, sys, glob, gc, torch
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; GB=1<<30
TESTS=[("/home/ubuntu/models/Mistral-7B-v0.1",0x51),("/home/ubuntu/models/Qwen2-7B",0x52)]  # incl the OOM model
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),
    ("cipher_pager_end_load",c_int,[c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class St(ctypes.Structure):
    _fields_=[("cm",c_ulong),("pi",c_ulong),("ev",c_ulong),("st",c_int),("ref",c_int),("pif",c_int),("used",c_ulong),("va",c_ulonglong)]
lib.cipher_pager_get_stats.argtypes=[c_int,POINTER(St)]
def used(r): s=St(); lib.cipher_pager_get_stats(r,ctypes.byref(s)); return s.used
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
def prequant(path):
    name=os.path.basename(path); out=f"/home/ubuntu/models_int4/{name}"
    if os.path.exists(f"{out}/config.json"): return out
    print(f"  quantizing+saving {name} -> {out} ...",flush=True)
    m=AutoModelForCausalLM.from_pretrained(path,quantization_config=BNB,torch_dtype=torch.float16,device_map="cuda")
    os.makedirs(out,exist_ok=True); m.save_pretrained(out)
    del m; gc.collect(); torch.cuda.empty_cache()
    return out
def ckpt_size(d): return sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors"))
key=0x51
for path,key in TESTS:
    ck=prequant(path); sz=ckpt_size(ck); reserve=int(sz*1.20)+(1<<30)
    rid=lib.cipher_pager_begin_load(key,reserve)
    pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.float16,device_map="cuda")
        torch.cuda.synchronize()
    assert lib.cipher_pager_end_load(rid)==0
    u=used(rid)
    print(f"  {os.path.basename(path)}: ckpt_disk={sz/GB:.2f}GiB reserve={reserve/GB:.2f}GiB region_bump={u/GB:.2f}GiB  (quant-on-load bump was 4.79/OOM)",flush=True)
    del m; gc.collect()
print("VERIFY: pre-quantized reload bump ~= ckpt size (no fp32 transient) -> fix works if bumps are tight",flush=True)
sys.stdout.flush(); os._exit(0)
