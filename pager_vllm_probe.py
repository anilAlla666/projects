#!/usr/bin/env python3
# Does an outer `with use_mem_pool(my_pool)` around LLM(...) CAPTURE vLLM's weight allocations?
# Modes (argv[1]): m1 = default (multiprocessing as-is), m2 = force in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0),
#                  m3 = sleep mode on (vLLM installs its OWN CuMemAllocator pool -> should shadow my outer pool).
# Confound controls (advisor): PID logging (subprocess vs bypass); in default mode assert CuMemAllocator.instance
# is None (a miss is then locality/process, NOT vLLM's own pool); in m3 read CuMemAllocator.get_current_usage().
import ctypes, os, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "m2"
if mode in ("m2", "m3"):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # force in-process worker
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
SO = "/home/ubuntu/pager_vllm_probe.so"
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
MB = 1 << 20
print(f"=== MODE {mode}  MAIN PID {os.getpid()}  mp={'off' if mode in ('m2','m3') else 'default'} ===")

import torch
lib = ctypes.CDLL(SO)
from torch.cuda.memory import CUDAPluggableAllocator
alloc = CUDAPluggableAllocator(SO, "vp_malloc", "vp_free")
pool = torch.cuda.MemPool(alloc.allocator())
torch.cuda.init(); _w = torch.zeros(1, device="cuda")
def stats():
    buf = (ctypes.c_ulonglong * 4)(); lib.vp_stats(buf); return list(buf)

# reference weight bytes
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL)
import vllm.device_allocator.cumem as cm
print(f"cumem_available (C-ext built) = {cm.cumem_available}")

from vllm import LLM
kw = dict(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048, dtype="float16")
if mode == "m3":
    kw["enable_sleep_mode"] = True

os.environ["VLLM_USE_DEEP_GEMM"] = "0"   # avoid an unrelated deep_gemm init crash; not relevant to weight alloc
base = stats()[0]
crashed = None
try:
    with torch.cuda.use_mem_pool(pool):
        llm = LLM(**kw)
    torch.cuda.synchronize()
except Exception as e:
    crashed = str(e)[:160]
b = stats()
inpool_mb = (b[0] - base) / MB
if crashed:
    print(f"(LLM init raised AFTER weight-load phase: {crashed})")

# confound control: was vLLM's OWN CuMemAllocator instantiated?
inst = cm.CuMemAllocator.instance
own_usage = None
if inst is not None:
    try: own_usage = inst.get_current_usage()
    except Exception as e: own_usage = f"err {e}"

# rough weight-bytes estimate (TinyLlama ~2.1 GiB fp16) for the captures? comparison
print(f"IN-POOL captured = {inpool_mb:8.1f} MiB  count={b[1]}  alloc_pid_seen={b[3]} (main={os.getpid()})")
print(f"vLLM CuMemAllocator.instance = {'None' if inst is None else 'PRESENT'}  own_pool_usage = {own_usage}")
verdict = ("CAPTURES (weights landed in my outer pool)" if inpool_mb > 1000 else
           "BYPASS (my outer pool captured ~0 of the weights)")
print(f"VERDICT[{mode}]: {verdict}")
sys.stdout.flush(); os._exit(0)
