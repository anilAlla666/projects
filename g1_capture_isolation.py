#!/usr/bin/env python3
# ISOLATION (one run, then report — advisor): is the token-1 capture divergence WARM-UP POLLUTION (the 3 warm-ups ran
# at cache_position=P on the REAL prefilled cache, likely advancing StaticCache's seen-length so the captured mask is
# wrong) or a FUNDAMENTAL non-capturable HF forward (mask/RoPE/control-flow baked at capture = the vLLM-core-rewrite
# wall)? Test variants vs eager-static (the clean capture-correctness signal): (NW) no warm-up; (TW) warm-up on a
# THROWAWAY cache, capture on the freshly-prefilled real cache. TinyLlama only (model-independent verdict). Token ids
# CLAMPED so a garbage argmax can't trigger an out-of-vocab embedding device-assert (which would poison the context).
# No timing (only the KL verdict matters here).
import os, sys, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)
path="/home/ubuntu/models/TinyLlama-1.1B"; N=48
tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device
def clamp(t): return int(max(0, min(V-1, int(t))))
@torch.no_grad()
def prefill(cache):
    o=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True)
    return o.logits[0,-1].argmax()
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=P+N+8); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt))
        o=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True)
        nt=o.logits[0,-1].argmax()
    return out
@torch.no_grad()
def graph_decode(warmup):   # warmup in {'none','throwaway'}
    cache=StaticCache(config=m.config, max_cache_len=P+N+8); t1=prefill(cache)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1))
    spos=torch.tensor([P],device=dev)
    if warmup=='throwaway':
        cw=StaticCache(config=m.config, max_cache_len=P+N+8); prefill(cw)
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): m(sin, cache_position=spos, past_key_values=cw, use_cache=True)   # pollutes cw, NOT cache
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits
    except Exception as e:
        return None, f"CAPTURE-ERROR: {repr(e)[:100]}"
    toks=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize()
        nt=clamp(slog[0,-1].argmax()); toks.append(nt); sin.fill_(nt); spos.add_(1)
    return toks, None
def firstdiff(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return -1
es=eager_static()
print(f"=== capture isolation (TinyLlama, N={N}); reference=eager-static (clean capture signal) ===",flush=True)
for wm in ['none','throwaway']:
    gs,err=graph_decode(wm)
    if err: print(f"  warmup={wm:<9}: {err}",flush=True); continue
    match=sum(a==b for a,b in zip(gs,es))
    print(f"  warmup={wm:<9}: graph vs eager-static = {match}/{N} (1stdiff@{firstdiff(gs,es)})  {'PASS capturable' if match==N else 'DIVERGES'}",flush=True)
print("VERDICT: throwaway/none PASS -> warm-up pollution (mechanism FEASIBLE, de-risked, NOT built); both DIVERGE@~1 -> HF forward not capture-safe (vLLM-core-rewrite wall, months/fragile).",flush=True)
sys.stdout.flush(); os._exit(0)
