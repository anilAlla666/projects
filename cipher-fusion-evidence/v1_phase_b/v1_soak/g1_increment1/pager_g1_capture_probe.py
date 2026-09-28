#!/usr/bin/env python3
# INCREMENT-1 PROBE: CIPHER-owned manual static-KV graph-decode over the PAGER's cuMemMap weight region.
# ONE condition per subprocess (mode argv), ONE capture, ONE monotonic burst (the empirically-deterministic
# vehicle -- single-step re-replay aliases the graph's private pool and is noisy; the burst is not). Compare the
# greedy token sequence to eager-static (KL=0 = greedy match, the gate Anil wrote). vLLM NOT in path.
#   resident  : load->prefill->es->capture->burst vs es              (graph-decode over pager, resident)
#   pagecycle : load->prefill->es->capture->page_out->page_in->burst vs es   <- MAKE-OR-BREAK (graph survives remap?)
#   negctrl   : load->prefill->es->capture->zero 2 down_proj->burst vs es     (must DIVERGE; proves discriminating power)
# Run with CIPHER_RT_DISABLE_AUTO_INIT=1 to isolate the pager from CIPHER's legacy compute actuators (Koopman/
# DET-SVD via cusolver are NOT capture-safe -- a real, separate sub-finding; the engine serving path is pager+graph).
import ctypes, os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
P=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
MODE=sys.argv[3] if len(sys.argv)>3 else "resident"
assert MODE in ("resident","pagecycle","negctrl")

lib=ctypes.CDLL(SO)
lib.cipher_pager_init.restype=ctypes.c_int
lib.cipher_pager_begin_load.restype=ctypes.c_int; lib.cipher_pager_begin_load.argtypes=[ctypes.c_ulonglong,ctypes.c_size_t]
lib.cipher_pager_end_load.restype=ctypes.c_int; lib.cipher_pager_end_load.argtypes=[ctypes.c_int]
lib.cipher_pager_page_out.restype=ctypes.c_int; lib.cipher_pager_page_out.argtypes=[ctypes.c_int]
lib.cipher_pager_page_in.restype=ctypes.c_int; lib.cipher_pager_page_in.argtypes=[ctypes.c_int]
lib.cipher_pager_live_cksum.restype=ctypes.c_ulonglong; lib.cipher_pager_live_cksum.argtypes=[ctypes.c_int]
class St(ctypes.Structure):
    _fields_=[("cold_miss",ctypes.c_ulong),("pagein_cnt",ctypes.c_ulong),("evict_cnt",ctypes.c_ulong),
              ("state",ctypes.c_int),("ref",ctypes.c_int),("pages_in_flight",ctypes.c_int),
              ("used_bytes",ctypes.c_ulong),("base_va",ctypes.c_ulonglong),
              ("peak_live",ctypes.c_ulong),("cur_live",ctypes.c_ulong)]
lib.cipher_pager_get_stats.argtypes=[ctypes.c_int,ctypes.POINTER(St)]

from torch.cuda.memory import CUDAPluggableAllocator
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _w=torch.zeros(1,device="cuda")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
tok=AutoTokenizer.from_pretrained(P); V=tok.vocab_size

# --- load weights INTO the pager cuMemMap region (proven routing) ---
cpu=AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.float16)
pb=sum(p.numel()*p.element_size() for p in cpu.parameters())+sum(b.numel()*b.element_size() for b in cpu.buffers())
rid=lib.cipher_pager_begin_load(0xC1, int(pb*1.10)+(256<<20))
pool=torch.cuda.MemPool(alloc.allocator())
with torch.cuda.use_mem_pool(pool):
    cpu.to("cuda"); torch.cuda.synchronize()
lib.cipher_pager_end_load(rid)
m=cpu.eval()
s=St(); lib.cipher_pager_get_stats(rid, ctypes.byref(s))
inreg=sum(1 for _,p in m.named_parameters() if s.base_va<=p.data_ptr()<s.base_va+s.used_bytes)
tot=sum(1 for _ in m.named_parameters())
assert inreg==tot, f"routing precondition failed ({inreg}/{tot})"

prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); Plen=pids.shape[1]; dev=pids.device; Lc=Plen+N+64
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(Plen,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()

# eager-static reference burst (computed BEFORE capture; eager-before-capture is safe)
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=Lc); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt))
        nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([Plen+i],device=dev),
             past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
es=eager_static()

# capture the single decode step over the SAME retained cache (cache RETAINED = use-after-free fix)
cache=StaticCache(config=m.config, max_cache_len=Lc); t1=prefill(cache)
sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([Plen],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits

note=""
if MODE=="pagecycle":
    ckb=lib.cipher_pager_live_cksum(rid); sb=St(); lib.cipher_pager_get_stats(rid,ctypes.byref(sb))
    ro=lib.cipher_pager_page_out(rid); ri=lib.cipher_pager_page_in(rid)
    cka=lib.cipher_pager_live_cksum(rid); sa=St(); lib.cipher_pager_get_stats(rid,ctypes.byref(sa))
    note=f"page_out={ro} page_in={ri} VA {hex(sb.base_va)}->{hex(sa.base_va)} stable={sb.base_va==sa.base_va} bytes_restored={ckb==cka} state={sa.state}"
elif MODE=="negctrl":
    nz=0
    with torch.no_grad():
        for nm_,p in m.named_parameters():
            if "mlp.down_proj" in nm_ and s.base_va<=p.data_ptr()<s.base_va+s.used_bytes:
                p.data.zero_(); nz+=1
                if nz>=2: break
    torch.cuda.synchronize(); note=f"zeroed {nz} down_proj at pager VA"

# ONE monotonic burst (the deterministic vehicle): page-cycle/perturb already happened BEFORE any replay
@torch.no_grad()
def burst_once():
    toks=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(Plen)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); toks.append(nt); sin.fill_(nt); spos.add_(1)
    return toks
gs=burst_once()
match=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
expect_kl0 = MODE in ("resident","pagecycle")
ok = (match==N) if expect_kl0 else (match<N)
print(f"[RESULT mode={MODE} model={os.path.basename(P)} N={N}] graph-burst vs eager-static: {match}/{N} (1stdiff@{fd})  "
      f"{'KL=0' if match==N else 'DIVERGES'}  {note}", flush=True)
print(f"[GATE mode={MODE}] expect={'KL=0' if expect_kl0 else 'DIVERGE'} -> {'PASS' if ok else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if ok else 1)
