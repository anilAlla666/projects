#!/usr/bin/env python3
# GATE-1 g1.2x reconciling co-residence measurement (advisor): keep model+captured-graph+cache ALIVE per model
# (engine-representative -- the engine keeps each model's graph), and print torch.cuda.memory_allocated (LIVE) vs nvml
# used per model. Settles density: live ~6-7GB/model -> ~10-12 fit (density real); live ~14-20GB -> graph/cache eats it
# (~4-5, no gain). device_map=cuda (NO pager, to avoid the routing-fallback confound; count is HBM-total). nf4.
import os, sys, torch, pynvml
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
pynvml.nvmlInit(); d=pynvml.nvmlDeviceGetHandleByIndex(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, BitsAndBytesConfig
GB=1<<30
def nvml(): return pynvml.nvmlDeviceGetMemoryInfo(d).used/GB
def live(): return torch.cuda.memory_allocated()/GB
# non-SWA distinct checkpoints (Mistral EXCLUDED -- SWA capture, separate binding term) + replicas
MODELS=["/home/ubuntu/models/Qwen2-7B","/home/ubuntu/models/Llama-3.1-8B","/home/ubuntu/models/Llama-3.2-1B-Instruct","/home/ubuntu/models/TinyLlama-1.1B"]
MAXN=int(os.environ.get("MAXN","12")); PROMPT="The history of artificial intelligence began in the 1950s when researchers first"
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
held=[]   # keep (model, graph, cache, slog, sin, spos) ALIVE -- engine-representative
torch.cuda.init(); _=torch.zeros(1,device="cuda")
base_live=live(); base_nvml=nvml()
for i in range(MAXN):
    path=MODELS[i%len(MODELS)]; nm=os.path.basename(path)
    if nvml()>72: print(f"[stop] nvml {nvml():.1f}GB -> capacity {len(held)} nf4 models",flush=True); break
    l0=live()
    try:
        tok=AutoTokenizer.from_pretrained(path)
        m=AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="cuda").eval()
        V=tok.vocab_size; dev="cuda"; pids=tok(PROMPT,return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; Lc=P+40
        def cl(t): return int(max(0,min(V-1,int(t))))
        with torch.no_grad():
            cache=StaticCache(config=m.config,max_cache_len=Lc)
            t1=m(pids,cache_position=torch.arange(P,device=dev),past_key_values=cache,use_cache=True).logits[0,-1].argmax()
            sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(cl(t1)); spos=torch.tensor([P],device=dev)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
            g.replay(); torch.cuda.synchronize()
        torch.cuda.empty_cache()   # reclaim only UNUSED cached blocks (can't free live)
        held.append((m,g,cache,slog,sin,spos))
        print(f"  nf4[{len(held)-1}] {nm}: LIVE={live()-base_live:.1f}GB (+{live()-l0:.1f}/model) | nvml={nvml()-base_nvml:.1f}GB | free={pynvml.nvmlDeviceGetMemoryInfo(d).free/GB:.1f}GB",flush=True)
    except Exception as ex:
        print(f"  nf4[{len(held)}] {nm} FAIL {type(ex).__name__} {str(ex)[:80]}",flush=True); break
nL=len(held); pm_live=(live()-base_live)/nL if nL else 0
print(f"\n[RESULT g1.2x co-residence] {nL} nf4 models held (model+captured-graph+cache ALIVE) | LIVE total={live()-base_live:.1f}GB = ~{pm_live:.1f}GB/model | nvml={nvml()-base_nvml:.1f}GB",flush=True)
cap=int(75/pm_live) if pm_live>0 else 0
print(f"  per-model LIVE footprint ~{pm_live:.1f}GB -> ~{cap} fit/80GB vs fp16 ~5 (inc-3) / compressed-tensors ~4 (g1.2)",flush=True)
print(f"[VERDICT] {'DENSITY REAL (live ~{:.0f}GB/model << fp16 ~15)'.format(pm_live) if pm_live<10 else 'NO density in-engine (graph/cache eats it, ~{:.0f}GB/model)'.format(pm_live)}",flush=True)
sys.stdout.flush(); os._exit(0)
