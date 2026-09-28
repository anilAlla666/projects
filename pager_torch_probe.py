#!/usr/bin/env python3
# PROBE 2 (make-or-break): can a torch-OWNED weight tensor live behind a CIPHER-pageable cuMemMap VA?
# torch CUDAPluggableAllocator -> weight tensor's storage IS my VA -> torch.matmul reads through it ->
# evict (unmap) + pagein (remap) the VA + recopy from RAM warm -> matmul restored. The real vLLM-integration path.
import os, ctypes, torch
import warnings; warnings.filterwarnings("ignore")
SO="/home/ubuntu/pager_probe_alloc.so"
lib=ctypes.CDLL(SO)
for fn in ("cipher_evict","cipher_pagein","cipher_mapped"):
    getattr(lib,fn).argtypes=[ctypes.c_void_p]; getattr(lib,fn).restype=ctypes.c_int
# install the cuMemMap allocator as torch's CUDA allocator BEFORE any CUDA alloc
alloc=torch.cuda.memory.CUDAPluggableAllocator(SO,"cipher_alloc","cipher_free")
torch.cuda.memory.change_current_allocator(alloc)

N=4096
W=(torch.randn(N,N,device="cuda",dtype=torch.float16)*0.1)   # torch-owned weight, storage = my cuMemMap VA
x=(torch.randn(8,N,device="cuda",dtype=torch.float16)*0.3)
va=W.data_ptr()
print(f"W.data_ptr()=0x{va:x}  mapped(before)={lib.cipher_mapped(ctypes.c_void_p(va))}")
y1=(x@W.T).clone(); torch.cuda.synchronize()             # real torch matmul reading weight through my VA
W_warm=W.detach().to("cpu").clone()                       # RAM warm copy (what the pager would keep)

# EVICT: unmap W's VA (its HBM is released) — event-gated: sync first so no in-flight read races the unmap
torch.cuda.synchronize()
ev=lib.cipher_evict(ctypes.c_void_p(va))
print(f"evict rc={ev}  mapped(after evict)={lib.cipher_mapped(ctypes.c_void_p(va))} (0 = unmapped, VA reserved)")

# PAGE-IN: remap fresh HBM to the SAME VA, recopy warm weights (W's data_ptr unchanged)
pi=lib.cipher_pagein(ctypes.c_void_p(va))
print(f"pagein rc={pi}  mapped(after pagein)={lib.cipher_mapped(ctypes.c_void_p(va))}")
W.copy_(W_warm.to("cuda", non_blocking=False)); torch.cuda.synchronize()   # restore from RAM warm
y2=(x@W.T).clone(); torch.cuda.synchronize()

ok=torch.equal(y2,y1); rel=(y2.float()-y1.float()).abs().max().item()
print(f"torch matmul after evict+pagein+restore == before: {ok} (max_abs_diff={rel:.2e})")
print(f"PROBE2 {'HOLDS' if ok else 'FAILS'}: torch-owned weight tensor lives behind a CIPHER-pageable VA, evictable+restorable, read by real torch.matmul")
