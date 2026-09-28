#!/usr/bin/env python3
# STEP 0 routing-signal probe (coexistence-correct): route weight allocations via a SCOPED torch.cuda.MemPool
# (custom allocator active only inside `with use_mem_pool`) instead of a global change_current_allocator. The
# load-phase window IS the `with` block. Default allocator (+ its memory_stats, which transformers/vLLM call)
# stays intact for KV/activation. Signal robust iff in-window bytes ~= param bytes AND the forward (outside the
# pool) does NOT route to the pager.
import ctypes, torch
SO = "/home/ubuntu/pager_route_probe.so"
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"

lib = ctypes.CDLL(SO)
from torch.cuda.memory import CUDAPluggableAllocator
alloc = CUDAPluggableAllocator(SO, "route_malloc", "route_free")
pool = torch.cuda.MemPool(alloc.allocator())

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
torch.cuda.init()
_ = torch.zeros(1, device="cuda")        # touch default allocator first (memory_stats path warm)

# ---- weight-load window = the MemPool context ----
lib.route_begin()
with torch.cuda.use_mem_pool(pool):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
    torch.cuda.synchronize()
lib.route_end()

# does the framework's memory bookkeeping still work (coexistence)?
mem_stats_ok = True
try:
    _ = torch.cuda.memory_reserved()
    _ = torch.cuda.memory_allocated()
except Exception as e:
    mem_stats_ok = False
    print("memory_stats ERROR:", e)

# real forward OUTSIDE the pool -> activations/scratch use default allocator, must NOT count
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(3):
        _ = model(ids)
torch.cuda.synchronize()

param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
nparam = sum(p.numel() for p in model.parameters())
buf = (ctypes.c_ulonglong * 6)()
lib.route_stats(buf)
win_b, win_c, aft_b, aft_c, win_max, aft_max = list(buf)
MB = 1024 * 1024
print(f"MODEL params={nparam/1e9:.3f}B  param_bytes={param_bytes/MB:.1f} MiB (fp16)")
print(f"IN-WINDOW  (weight-load): {win_b/MB:8.1f} MiB  count={win_c:5d}  max1alloc={win_max/MB:.1f} MiB")
print(f"POST-WINDOW(fwd/kv/act):  {aft_b/MB:8.1f} MiB  count={aft_c:5d}  max1alloc={aft_max/MB:.1f} MiB")
ratio = win_b / param_bytes if param_bytes else 0
print(f"in-window / param_bytes = {ratio:.3f}   memory_stats_works={mem_stats_ok}")
captures = 0.95 <= ratio <= 1.20
post_small = aft_b < 0.5 * param_bytes
print(f"SIGNAL: window-captures-weights={captures}  post-window-not-weights={post_small}  coexist={mem_stats_ok}"
      f" -> {'MEMPOOL PHASE-MARKER ROBUST' if (captures and post_small and mem_stats_ok) else 'NEEDS REFINEMENT'}")
