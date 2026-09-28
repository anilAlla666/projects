#!/usr/bin/env python3
# INCREMENT-1 PROBE-FIRST: the unproven mechanism = CUDA-graph decode over the PAGER's cuMemMap weight region.
# graph-gate-step1 proved Marlin (a GEMM) captures KL=0; a FULL decode step (static-KV) over pager-resident weights
# is unproven. Test: load a model with weights in the pager region (proven MemPool routing) + StaticCache +
# torch.compile(reduce-overhead) [torch's CUDA-graph capture] -> generate; compare to EAGER (per-agent KL=0) +
# measure burst_t (must drop from HF-generate ~73ms/tok toward graph-decode). If it fails, isolate pager-vs-compile.
import ctypes, os, sys, time, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"; P="/home/ubuntu/models/TinyLlama-1.1B"; GB=1<<30
lib=ctypes.CDLL(SO)
for fn,res,args in [("cipher_pager_init",ctypes.c_int,[]),("cipher_pager_begin_load",ctypes.c_int,[ctypes.c_ulonglong,ctypes.c_size_t]),
    ("cipher_pager_end_load",ctypes.c_int,[ctypes.c_int])]:
    f=getattr(lib,fn); f.restype=res; f.argtypes=args
class St(ctypes.Structure):
    _fields_=[("cm",ctypes.c_ulong),("pi",ctypes.c_ulong),("ev",ctypes.c_ulong),("st",ctypes.c_int),("ref",ctypes.c_int),
              ("pif",ctypes.c_int),("used",ctypes.c_ulong),("va",ctypes.c_ulonglong),("peak",ctypes.c_ulong),("live",ctypes.c_ulong)]
lib.cipher_pager_get_stats.argtypes=[ctypes.c_int,ctypes.POINTER(St)]
from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(P)
# --- load weights INTO the pager region (proven routing) ---
cpu=AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.float16)
pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
rid=lib.cipher_pager_begin_load(0xC1, int(pb*1.10)+(256<<20))
pool=torch.cuda.MemPool(alloc.allocator())
with torch.cuda.use_mem_pool(pool):
    cpu.to("cuda"); torch.cuda.synchronize()
lib.cipher_pager_end_load(rid)
m=cpu
s=St(); lib.cipher_pager_get_stats(rid, ctypes.byref(s))
inreg=sum(1 for _,p in m.named_parameters() if s.va<=p.data_ptr()<s.va+s.used); tot=sum(1 for _ in m.named_parameters())
print(f"weights in pager region: {inreg}/{tot} (va={hex(s.va)} used={s.used>>20}MiB)",flush=True)
ids=tok("The history of artificial intelligence began in the", return_tensors="pt").input_ids.cuda()
N=32
def egen():
    with torch.no_grad(): return m.generate(ids, max_new_tokens=N, do_sample=False, use_cache=True)
# eager reference + burst_t
o_eager=egen(); torch.cuda.synchronize()
t=time.time(); o_eager=egen(); torch.cuda.synchronize(); eager_t=time.time()-t
print(f"EAGER over pager weights: {eager_t*1000:.0f}ms/{N}tok = {N/eager_t:.0f} tok/s  out={tok.decode(o_eager[0][ids.shape[1]:])[:40]!r}",flush=True)
# --- graph-decode: StaticCache + torch.compile (CUDA-graph) ---
try:
    m.generation_config.cache_implementation="static"
    m.forward=torch.compile(m.forward, mode="reduce-overhead", fullgraph=True)
    # warmup/compile (first calls compile + capture graphs)
    for _ in range(2):
        with torch.no_grad(): _=m.generate(ids, max_new_tokens=N, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    t=time.time()
    with torch.no_grad(): o_c=m.generate(ids, max_new_tokens=N, do_sample=False, use_cache=True)
    torch.cuda.synchronize(); comp_t=time.time()-t
    kl0 = torch.equal(o_eager[0][:ids.shape[1]+N], o_c[0][:ids.shape[1]+N])
    print(f"GRAPH-DECODE (compile+static, over pager): {comp_t*1000:.0f}ms/{N}tok = {N/comp_t:.0f} tok/s ({eager_t/comp_t:.1f}x faster)",flush=True)
    print(f"PROBE VERDICT: KL=0 (graph-decode == eager) = {kl0}; mechanism = {'HOLDS (graph-decode over pager region, KL=0, faster)' if kl0 else 'DIVERGES (diagnose)'}",flush=True)
except Exception as e:
    import traceback
    print(f"GRAPH-DECODE over pager FAILED: {str(e)[:200]}",flush=True)
    traceback.print_exc()
    print("  -> isolate: is it torch.compile or the pager cuMemMap region? (diagnose for the engine build)",flush=True)
sys.stdout.flush(); os._exit(0)
