#!/usr/bin/env python3
# PROBE-FIRST (the riskiest mechanism, before any months-grind): can CIPHER do MANUAL static-KV torch.cuda.graph
# capture of a decode step and replay it KL=0? The transformers torch.compile(reduce-overhead)/CUDAGraph-trees path
# FAILED (aliasing bug, V0_GO1_INCREMENT1_ENGINE_COST.md) -> CIPHER must hand-roll: StaticCache (fixed KV buffers,
# in-place write at cache_position) + raw torch.cuda.graph (CIPHER owns the buffers; raw capture is NOT vLLM's
# monitor-blocked path). GATE: greedy-match KL=0 vs eager + decode tok/s graph-vs-eager (does graph remove the
# per-token launch gaps -> lower per-burst latency, the diagnosed lever; overlap was REFUTED).
# Reference chain for clean attribution: eager-dynamic (ground truth) -> eager-static (isolates StaticCache numerics)
# -> graph-static (isolates the GRAPH). NORMAL allocator here (isolate the capture mechanism; pager wiring = increment 1).
import os, sys, time, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)
def probe(path, N=64):
    nm=os.path.basename(path)
    tok=AutoTokenizer.from_pretrained(path)
    m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
    prompt="The history of artificial intelligence began in the 1950s, when researchers first"
    pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]
    dev=pids.device
    @torch.no_grad()
    def eager_dynamic():           # ground-truth greedy (default dynamic cache)
        ids=pids.clone(); out=[]
        pkv=None; cp=torch.arange(P,device=dev)
        o=m(ids, use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
        for _ in range(N):
            out.append(int(nt)); o=m(nt.view(1,1), past_key_values=pkv, use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
        return out
    @torch.no_grad()
    def eager_static():            # StaticCache, NO graph (isolates cache numerics)
        L=P+N+8; cache=StaticCache(config=m.config, max_cache_len=L)
        o=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True)
        nt=o.logits[0,-1].argmax(); out=[]
        for i in range(N):
            out.append(int(nt))
            o=m(nt.view(1,1), cache_position=torch.tensor([P+i],device=dev), past_key_values=cache, use_cache=True)
            nt=o.logits[0,-1].argmax()
        return out
    @torch.no_grad()
    def graph_static():            # StaticCache + MANUAL torch.cuda.graph capture of the decode step
        L=P+N+8; cache=StaticCache(config=m.config, max_cache_len=L)
        o=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True)
        t1=o.logits[0,-1].argmax()
        sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.copy_(t1.view(1,1))
        spos=torch.tensor([P],device=dev)
        s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())   # capture-pool warmup
        with torch.cuda.stream(s):
            for _ in range(3): m(sin, cache_position=spos, past_key_values=cache, use_cache=True)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=out.logits
        # FIX: do NOT read slog straight from capture (output not valid until a replay). Drive EVERY decode token
        # through replay(), reading slog only AFTER replay. First replay re-runs forward(t1 @ P) -> valid t2.
        toks=[int(t1)]; sin.copy_(t1.view(1,1)); spos.fill_(P)
        for _ in range(N-1):
            g.replay(); torch.cuda.synchronize()
            nt=slog[0,-1].argmax(); toks.append(int(nt))
            sin.copy_(nt.view(1,1)); spos.add_(1)
        return toks, g, sin, spos, slog, cache, P
    ed=eager_dynamic(); es=eager_static()
    gs, g, sin, spos, slog, cache, P = graph_static()
    def firstdiff(a,b):
        for i,(x,y) in enumerate(zip(a,b)):
            if x!=y: return i
        return -1
    kl_gs_ed = sum(a==b for a,b in zip(gs,ed))   # graph-static vs ground-truth eager (THE gate)
    kl_gs_es = sum(a==b for a,b in zip(gs,es))   # graph vs eager-static (isolates the graph mechanism = capture-correctness)
    kl_es_ed = sum(a==b for a,b in zip(es,ed))
    print(f"[{nm}] N={N}  greedy-match: graph vs eager-DYNAMIC(gt)={kl_gs_ed}/{N} (1stdiff@{firstdiff(gs,ed)})  "
          f"graph vs eager-static={kl_gs_es}/{N} (1stdiff@{firstdiff(gs,es)}; the clean capture signal)  "
          f"eager-static vs dynamic={kl_es_ed}/{N} (1stdiff@{firstdiff(es,ed)})",flush=True)
    # latency: eager-dynamic decode tok/s vs graph-replay tok/s
    @torch.no_grad()
    def time_eager():
        ids=pids.clone(); o=m(ids,use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(N): o=m(nt.view(1,1),past_key_values=pkv,use_cache=True); pkv=o.past_key_values; nt=o.logits[0,-1].argmax()
        torch.cuda.synchronize(); return N/(time.time()-t0)
    def time_graph():
        spos.fill_(P)                     # reset to a safe in-bounds position (constant -> measures replay cost; no overflow)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(N): g.replay()
        torch.cuda.synchronize(); return N/(time.time()-t0)
    te=time_eager(); tg=time_graph()
    print(f"[{nm}] decode tok/s: eager={te:.0f}  graph-replay={tg:.0f}  speedup={tg/te:.2f}x (LAUNCH-OVERHEAD removal, not bandwidth-floor)",flush=True)
    print(f"[{nm}] per-token: eager={1000/te:.1f}ms graph={1000/tg:.1f}ms | 128-tok burst SOLO: eager={128/te*1000:.0f}ms graph={128/tg*1000:.0f}ms (sub-1000ms? {'YES' if 128/tg*1000<1000 else 'no'})",flush=True)
    del m; torch.cuda.empty_cache()
    return kl_gs_ed==N
print("=== manual static-KV CUDA-graph capture probe (KL=0 + decode speedup) ===",flush=True)
ok1=probe("/home/ubuntu/models/TinyLlama-1.1B")     # fast mechanism probe
ok2=probe("/home/ubuntu/models/Llama-3.1-8B")       # big model (the 128-tok->5.4s regime)
print(f"PROBE VERDICT: manual capture KL=0 vs eager: TinyLlama={'PASS' if ok1 else 'FAIL'}  Llama-3.1={'PASS' if ok2 else 'FAIL'}",flush=True)
sys.stdout.flush(); os._exit(0)
