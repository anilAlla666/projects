#!/usr/bin/env python3
# Advisor catch: prove page_out actually RECLAIMS physical HBM (the density thesis), not just that it's
# byte-correct. mem_get_info() is driver-level (cuMemGetInfo) -> reflects real physical free regardless of torch
# caching. Expect: evict frees ~region bytes; pagein takes them back.
import ctypes, os, torch
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
A = "/home/ubuntu/models/TinyLlama-1.1B"
lib = ctypes.CDLL(SO)
for fn, res, args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),
    ("cipher_pager_end_load",c_int,[c_int]),("cipher_pager_page_in",c_int,[c_int]),("cipher_pager_page_out",c_int,[c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class Stats(ctypes.Structure):
    _fields_=[("cm",c_ulong),("pi",c_ulong),("ev",c_ulong),("st",c_int),("ref",c_int),("pif",c_int),("used",c_ulong),("va",c_ulonglong)]
lib.cipher_pager_get_stats.argtypes=[c_int,POINTER(Stats)]
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM
cpu=AutoModelForCausalLM.from_pretrained(A,torch_dtype=torch.float16)
pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
reserve=int(pb*1.10)+(64<<20)
rid=lib.cipher_pager_begin_load(0xA1,reserve); assert rid>=0
pool=torch.cuda.MemPool(alloc.allocator())
with torch.cuda.use_mem_pool(pool):
    cpu.to("cuda"); torch.cuda.synchronize()
assert lib.cipher_pager_end_load(rid)==0
s=Stats(); lib.cipher_pager_get_stats(rid,ctypes.byref(s)); MB=1<<20
torch.cuda.synchronize()
free0=torch.cuda.mem_get_info()[0]
assert lib.cipher_pager_page_out(rid)==0
free1=torch.cuda.mem_get_info()[0]
assert lib.cipher_pager_page_in(rid)==0
free2=torch.cuda.mem_get_info()[0]
gained=(free1-free0); region=(reserve+(2<<20)-1)//(2<<20)*(2<<20)
print(f"region used={s.used//MB}MiB reserve~={region//MB}MiB")
print(f"free: before_evict={free0//MB}MiB  evicted={free1//MB}MiB  after_pagein={free2//MB}MiB")
print(f"evict reclaimed = {gained//MB} MiB  (expect ~ region {region//MB} MiB)")
reclaim_ok = gained >= int(0.90*s.used)            # at least the weight bytes returned to the driver free pool
restore_ok = abs(free2-free0) < 64*MB              # pagein took them back
print(f"[{'PASS' if reclaim_ok else 'FAIL'}] G-reclaim: page_out returns physical HBM to the driver free pool")
print(f"[{'PASS' if restore_ok else 'FAIL'}] G-reacquire: page_in re-consumes it (free back to baseline)")
import sys; sys.stdout.flush(); os._exit(0 if (reclaim_ok and restore_ok) else 1)
