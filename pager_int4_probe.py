#!/usr/bin/env python3
# PROBE (binding, before the INT4 build): do bnb NF4 Linear4bit weights land in a torch MemPool? i.e. does bnb
# allocate the packed 4-bit weight through the torch allocator that use_mem_pool captures? If captured ~= 4-bit
# weight bytes -> the proven pager holds INT4 weights with NO new code. If ~0 -> bnb bypasses (diagnose).
import ctypes, os, sys, torch
SO = "/home/ubuntu/pager_vllm_probe.so"     # pool-counting allocator (vp_malloc counts every in-pool alloc)
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"; GB = 1e9
lib = ctypes.CDLL(SO)
from torch.cuda.memory import CUDAPluggableAllocator
alloc = CUDAPluggableAllocator(SO, "vp_malloc", "vp_free")
pool = torch.cuda.MemPool(alloc.allocator())
torch.cuda.init(); _w = torch.zeros(1, device="cuda")
def stats(): buf=(ctypes.c_ulonglong*4)(); lib.vp_stats(buf); return list(buf)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = stats()[0]
try:
    with torch.cuda.use_mem_pool(pool):
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda")
        torch.cuda.synchronize()
except Exception as e:
    print("LOAD RAISED:", str(e)[:200]);
    b=stats(); print(f"captured-so-far={(b[0]-base)/GB:.2f}GB"); sys.stdout.flush(); os._exit(1)
b = stats()
wb = sum(p.numel()*p.element_size() for p in m.parameters())   # 4-bit packed (uint8) + non-quant params
nq = sum(1 for _,mod in m.named_modules() if mod.__class__.__name__=="Linear4bit")
captured = (b[0]-base)/GB
print(f"4-bit Linear4bit modules = {nq}")
print(f"model param bytes (packed) = {wb/GB:.2f} GB")
print(f"CAPTURED in MemPool = {captured:.2f} GB  count={b[1]}  alloc_pid={b[3]} (main={os.getpid()})")
ok = captured >= 0.5*wb/GB and captured > 0.3
print(f"PROBE: bnb 4-bit weights captured by MemPool = {ok}  -> {'PAGER HOLDS INT4 (no new code)' if ok else 'BNB BYPASSES (diagnose)'}")
# quick correctness: a forward works (weights usable from the pool)
ids = __import__('transformers').AutoTokenizer.from_pretrained(MODEL)("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad(): out = m(ids)
print(f"forward OK from pool-resident 4-bit weights: logits {tuple(out.logits.shape)}")
sys.stdout.flush(); os._exit(0 if ok else 2)
