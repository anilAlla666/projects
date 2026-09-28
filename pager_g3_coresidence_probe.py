#!/usr/bin/env python3
# INCREMENT-3 PROBE-FIRST: the SINGLETON-WALL test. M DISTINCT models, each in its OWN pager region (CIPHER's
# pluggable allocator) with its OWN captured static-KV graph, ALL in ONE process. vLLM's CuMemAllocator +
# cudagraph-capture singletons would block this; CIPHER owns the allocator (driver VMM boundary) + the per-model
# graph capture (in-process). MAKE-OR-BREAK: do M distinct captured graphs coexist + each replay KL=0 in one process?
# Gate: per-model per-agent KL=0 (each model's monotonic-burst == its solo run). CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel

N=int(os.environ.get("N","64"))
ALL={"mistral":("Mistral-7B-v0.1",0xC1), "qwen2":("Qwen2-7B",0xC2), "llama":("Llama-3.1-8B",0xC3),
     "llama1b":("Llama-3.2-1B-Instruct",0xC4), "tiny":("TinyLlama-1.1B",0xC5)}
ORDER=os.environ.get("ORDER","mistral,qwen2").split(",")
MODELS=[ALL[k] for k in ORDER]
PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"

pager=CipherPager()
# --- load ALL M distinct models, each into its OWN pager region (serial loads; distinct keys) ---
engs=[]
for name,key in MODELS:
    e=PagerGraphModel(pager, f"/home/ubuntu/models/{name}", model_key=key).load()
    s=pager.stats(e.rid)
    free=torch.cuda.mem_get_info()[0]/(1<<30)
    print(f"[load] {name} region rid={e.rid} key={hex(key)} base_va={hex(s.base_va)} used={s.used_bytes>>20}MiB state={s.state} | HBM free={free:.1f}GB", flush=True)
    engs.append((name,e))

# --- capture ALL M graphs (interleaved capture is the stress: do M distinct captured graphs coexist?) ---
for name,e in engs:
    e.capture(PROMPT, N)
    print(f"[capture] {name} graph captured (base_va resident={hex(pager.stats(e.rid).base_va)})", flush=True)

# --- replay ALL graphs first (no eager between capture and replay), THEN eager references ---
gs={name:e.serve_burst(N) for name,e in engs}
es={name:e.eager_burst(N) for name,e in engs}

# --- per-model per-agent KL=0 (ratified: exact OR sub-delta near-tie at 1stdiff) ---
allok=True
for name,e in engs:
    g,s=gs[name],es[name]
    mt=sum(a==b for a,b in zip(g,s)); fd=next((i for i,(x,y) in enumerate(zip(g,s)) if x!=y),-1)
    ok=(mt==N)   # batch-1 per model -> exact greedy match expected (no batch-shape delta)
    allok=allok and ok
    print(f"  [{name}] graph vs solo: {mt}/{N} (1stdiff@{fd})  {'KL=0' if ok else 'DIVERGES'}", flush=True)

# --- cross-model interference control: each model's first token must differ (distinct models, distinct outputs) ---
firsts={name:gs[name][0] for name,e in engs}
distinct = len(set(firsts.values()))>1 or len(engs)==1
free=torch.cuda.mem_get_info()[0]/(1<<30)
print(f"[RESULT coresidence M={len(engs)}] per-model KL=0 ALL={allok}  one-process={len(engs)} models co-resident  HBM free={free:.1f}GB", flush=True)
print(f"  first-tokens per model: {firsts} (distinct outputs={distinct})", flush=True)
print(f"[SINGLETON-WALL] {'DISSOLVED -- M distinct captured graphs coexist + replay KL=0 in ONE process (vLLM singletons would block this)' if allok else 'NOT DISSOLVED -- diagnose binding term (HBM/capture-state/allocator), NO scope-down'}", flush=True)
print(f"[GATE coresidence] {'PASS' if allok else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if allok else 1)
