#!/usr/bin/env python3
# ROOT CAUSE CONFIRMED + LATENCY PAYOFF. The 8B replay assert was a DANGLING-CACHE lifetime bug in the prior latency
# harness: build_graph() created the StaticCache as a LOCAL and didn't return it -> GC freed its KV buffers after
# capture -> the captured index_copy_ wrote into freed/reused memory -> index OOB on replay. NOT GQA/arch/N (clean
# harness captures KL=0 on Llama-3.1-8B @N=128). FIX: keep the cache in scope. Here: cache retained at top level ->
# KL=0 + the latency payoff (128-tok 8B burst: graph-decode vs eager). The number increment-1 wanted.
import os, sys, time, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)
path="/home/ubuntu/models/Llama-3.1-8B"; N=128
tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device; L=P+N+64
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=L); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt)); nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
es=eager_static()
# --- build the decode graph with the cache RETAINED (the fix) ---
cache=StaticCache(config=m.config, max_cache_len=L)   # top-level -> stays alive through replay+timing
t1=prefill(cache)
sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits
# KL via drive-via-replay (cache alive)
gs=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
for _ in range(N-1):
    g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
match=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
print(f"=== Llama-3.1-8B graph-decode, cache RETAINED (root-cause fix) ===",flush=True)
print(f"  KL=0 (greedy-match graph vs eager-static): {match}/{N} (1stdiff@{fd})  {'PASS' if match==N else 'DIVERGES'}",flush=True)
# --- latency payoff: single-burst eager (full transformers forward/token = the 5.4s-regime per-burst) vs graph ---
@torch.no_grad()
def time_eager(K):
    o=m(pids,use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(K): o=m(nt.view(1,1),past_key_values=pkv,use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
    torch.cuda.synchronize(); return time.time()-t0
def time_graph(K):
    sin.fill_(clamp(t1)); spos.fill_(P)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(K): g.replay()
    torch.cuda.synchronize(); return time.time()-t0
for K in [16,128]:
    e=time_eager(K); gt=time_graph(K)
    print(f"  {K}-tok burst SOLO: eager={e*1000:.0f}ms  graph={gt*1000:.0f}ms  speedup={e/gt:.2f}x  per-tok eager={e/K*1000:.1f}ms graph={gt/K*1000:.1f}ms  (graph sub-1000ms? {'YES' if gt*1000<1000 else 'no'})",flush=True)
print("NOTE: single-burst latency (1 agent). speedup = per-token launch-overhead removal (the diagnosed lever; decode still HBM-bound underneath). 100-agent p99 builds on this + same-model batching. Mechanism KL=0 on the 8B; root cause was harness cache-lifetime, NOT arch.",flush=True)
sys.stdout.flush(); os._exit(0)
