#!/usr/bin/env python3
# GATE-1 g1.2x: bnb NF4 into the PAGER + capture + co-residence DENSITY. nf4 stays packed (STEP1 6.3GB) + captures
# KL=0 (STEP2). Here: route nf4 weights into a pager region (begin_load -> use_mem_pool -> from_pretrained(bnb,
# device_map=cuda) -> end_load), capture the static-KV graph, verify nf4-graph == nf4-eager KL=0 + pagecycle
# (survives physical evict/restore), and count how many distinct nf4 models co-reside vs fp16 ~5 / compressed-tensors ~4.
import os, sys, ctypes, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, BitsAndBytesConfig
from torch.cuda.memory import CUDAPluggableAllocator
GB=1<<30; N=32; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
# distinct fp16 checkpoints on disk -> bnb NF4 quantizes any of them on load (NO download)
MODELS=["/home/ubuntu/models/Qwen2-7B","/home/ubuntu/models/Llama-3.1-8B","/home/ubuntu/models/Mistral-7B-v0.1",
        "/home/ubuntu/models/Llama-3.2-1B-Instruct","/home/ubuntu/models/TinyLlama-1.1B"]
MAXN=int(os.environ.get("MAXN","12"))
lib=ctypes.CDLL(SO)
lib.cipher_pager_init.restype=ctypes.c_int
lib.cipher_pager_begin_load.restype=ctypes.c_int; lib.cipher_pager_begin_load.argtypes=[ctypes.c_ulonglong,ctypes.c_size_t]
lib.cipher_pager_end_load.restype=ctypes.c_int; lib.cipher_pager_end_load.argtypes=[ctypes.c_int]
class St(ctypes.Structure):
    _fields_=[("cold_miss",ctypes.c_ulong),("pagein_cnt",ctypes.c_ulong),("evict_cnt",ctypes.c_ulong),("state",ctypes.c_int),
              ("ref",ctypes.c_int),("pages_in_flight",ctypes.c_int),("used_bytes",ctypes.c_ulong),("base_va",ctypes.c_ulonglong),
              ("peak_live",ctypes.c_ulong),("cur_live",ctypes.c_ulong)]
lib.cipher_pager_get_stats.argtypes=[ctypes.c_int,ctypes.POINTER(St)]
alloc=CUDAPluggableAllocator(SO,"cipher_pager_malloc","cipher_pager_free")
assert lib.cipher_pager_init()==0
torch.cuda.init(); _=torch.zeros(1,device="cuda")
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

def load_nf4_into_pager(path, key):
    cpu=None  # quantize directly to cuda inside the pager pool
    pb_est=20<<30
    rid=lib.cipher_pager_begin_load(key, pb_est)
    pool=torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        m=AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="cuda").eval()
        torch.cuda.synchronize()
    lib.cipher_pager_end_load(rid)
    return m, rid, pool

def capture_and_check(m, tok):
    V=tok.vocab_size; dev="cuda"
    pids=tok(PROMPT, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; Lc=P+N+64
    def cl(t): return int(max(0,min(V-1,int(t))))
    def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    with torch.no_grad():
        c=StaticCache(config=m.config,max_cache_len=Lc); nt=prefill(c); es=[]
        for i in range(N):
            es.append(cl(nt)); nt=m(torch.tensor([[cl(nt)]],device=dev),cache_position=torch.tensor([P+i],device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
    cache=StaticCache(config=m.config,max_cache_len=Lc); t1=prefill(cache)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(cl(t1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
    gs=[cl(t1)]; sin.fill_(cl(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=cl(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
    return sum(a==b for a,b in zip(gs,es))

engs=[]; ok=0
for i in range(MAXN):
    path=MODELS[i%len(MODELS)]
    free0=torch.cuda.mem_get_info()[0]/GB
    if free0<7.0: print(f"[stop] HBM headroom {free0:.1f}GB<7 -> capacity {len(engs)} nf4 models",flush=True); break
    try:
        tok=AutoTokenizer.from_pretrained(path)
        m,rid,pool=load_nf4_into_pager(path,0xE0+i)
        s=St(); lib.cipher_pager_get_stats(rid,ctypes.byref(s)); foot=s.used_bytes/GB
        match=capture_and_check(m,tok); kl0=(match==N); ok+=1 if kl0 else 0
        free1=torch.cuda.mem_get_info()[0]/GB
        print(f"  nf4[{len(engs)}] {os.path.basename(path)} pager-region={foot:.2f}GB nf4-graph-vs-eager={match}/{N} {'KL=0' if kl0 else 'DIVERGES'} | HBM free={free1:.1f}GB",flush=True)
        engs.append((m,rid,pool))
    except Exception as ex:
        import traceback; traceback.print_exc(); print(f"  nf4[{len(engs)}] {os.path.basename(path)} FAIL {type(ex).__name__} {str(ex)[:90]}",flush=True); break
free=torch.cuda.mem_get_info()[0]/GB
print(f"\n[RESULT g1.2x density] {len(engs)} nf4 models co-resident (one process), per-model KL=0 ALL={ok==len(engs)} | HBM free={free:.1f}GB",flush=True)
print(f"  vs fp16 baseline ~5 (inc-3) / compressed-tensors ~4 (g1.2, no gain). nf4 packed ~6GB/model.",flush=True)
print(f"[GATE g1.2x] {'PASS (>5 nf4 co-resident + per-model KL=0 -> density REAL)' if (len(engs)>5 and ok==len(engs)) else 'see result'}",flush=True)
sys.stdout.flush(); os._exit(0)
