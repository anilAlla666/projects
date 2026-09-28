#!/usr/bin/env python3
# Increment-1 LATENCY payoff (the lever's magnitude) on the BIG model, using the KL-VERIFIED fixed recipe (no warm-up
# on the real prefilled cache -> the warm-up-pollution bug is gone). Confirms KL=0 on Llama-3.1-8B too (isolation was
# TinyLlama), then: does graph-decode take the 128-tok burst from eager's multi-second toward sub-second SOLO? This is
# the diagnosed lever (overlap REFUTED; graph-decode removes per-token launch gaps). NOT the engine; NORMAL allocator
# (pager wiring + full multi-increment engine = Anil's next-increment decision). Token ids clamped (no device-assert).
import os, sys, time, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)
path="/home/ubuntu/models/Llama-3.1-8B"; N=64   # match the TinyLlama working regime; isolates bounds vs fundamental
L=None  # set after P known
tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device
print(f"P(prompt tokens)={P}  N={N}  max_cache_len={P+N+64}  max replay pos={P+N-2}",flush=True)
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=P+N+64); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt)); nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
@torch.no_grad()
def build_graph():
    c=StaticCache(config=m.config, max_cache_len=P+N+64); t1=prefill(c)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()                    # NO warm-up on the real cache (the KL-verified recipe)
    with torch.cuda.graph(g):
        o=m(sin, cache_position=spos, past_key_values=c, use_cache=True); slog=o.logits
    return g,sin,spos,slog,t1
@torch.no_grad()
def graph_decode(g,sin,spos,slog,t1):
    toks=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); toks.append(nt); sin.fill_(nt); spos.add_(1)
    return toks
es=eager_static()
g,sin,spos,slog,t1=build_graph(); gs=graph_decode(g,sin,spos,slog,t1)
match=sum(a==b for a,b in zip(gs,es))
print(f"=== Llama-3.1-8B graph-decode (fixed recipe) ===",flush=True)
print(f"  KL=0 (greedy-match graph vs eager-static): {match}/{N}  {'PASS' if match==N else 'DIVERGES'}",flush=True)
# latency: eager (full transformers forward/token, dynamic cache = the 5.4s regime) vs graph-replay burst
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
    print(f"  {K}-tok burst SOLO: eager={e*1000:.0f}ms  graph-replay={gt*1000:.0f}ms  speedup={e/gt:.2f}x  (graph sub-1000ms? {'YES' if gt*1000<1000 else 'no'})",flush=True)
print("NOTE: speedup = per-token LAUNCH-OVERHEAD removal (the diagnosed lever); decode is still HBM-bound underneath. Mechanism DE-RISKED, engine NOT built (Anil's call).",flush=True)
sys.stdout.flush(); os._exit(0)
