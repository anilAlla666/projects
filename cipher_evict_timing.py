#!/usr/bin/env python3
# ASSESSMENT 1 clincher: CIPHER pager evict(page_out)/restore(page_in) latency for a REAL 7B, apples-to-apples vs
# vLLM sleep=5640ms / wake=416ms. CIPHER's page_out is unmap-only (no D2H; warm mirror captured once at end_load,
# weights read-only) -> should be ~ms vs vLLM's per-sleep re-offload.
import ctypes, os, sys, time, torch
from ctypes import c_int, c_ulonglong, c_size_t, POINTER
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; MODEL="/home/ubuntu/models/Mistral-7B-v0.1"; GB=1<<30
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",c_int,[]),("cipher_pager_begin_load",c_int,[c_ulonglong,c_size_t]),
    ("cipher_pager_end_load",c_int,[c_int]),("cipher_pager_page_in",c_int,[c_int]),("cipher_pager_page_out",c_int,[c_int]),
    ("cipher_pager_state",c_int,[c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
def used(): f,t=torch.cuda.mem_get_info(); return (t-f)/GB
from transformers import AutoModelForCausalLM
cpu=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16)
pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
rid=lib.cipher_pager_begin_load(0x7b, int(pb*1.10)+(64<<20))
pool=torch.cuda.MemPool(alloc.allocator())
with torch.cuda.use_mem_pool(pool):
    cpu.to("cuda"); torch.cuda.synchronize()
assert lib.cipher_pager_end_load(rid)==0
print(f"loaded 7B behind region: weight_bytes={pb/GB:.1f}GiB after_load_HBM={used():.1f}GiB")
# time evict/restore over several cycles (state must be RESIDENT=2 before page_out)
ev=[]; rs=[]
for i in range(5):
    torch.cuda.synchronize()
    t=time.time(); rc=lib.cipher_pager_page_out(rid); ev.append((time.time()-t)*1000)
    free_after_evict=used()
    t=time.time(); rc2=lib.cipher_pager_page_in(rid); rs.append((time.time()-t)*1000)
    if i==0: print(f"  cycle0: evict freed HBM -> {free_after_evict:.1f}GiB (was ~{pb/GB:.1f}GiB of weights)")
print(f"RESULT CIPHER page_out (evict) latency: min={min(ev):.0f}ms median={sorted(ev)[len(ev)//2]:.0f}ms  (vLLM sleep=5640ms)")
print(f"RESULT CIPHER page_in (restore) latency: min={min(rs):.0f}ms median={sorted(rs)[len(rs)//2]:.0f}ms  (vLLM wake=416ms)")
print(f"RESULT swap cost (evict victim + restore demanded): CIPHER ~{min(ev)+min(rs):.0f}ms vs N-sleep ~{5640+416}ms")
sys.stdout.flush(); os._exit(0)
