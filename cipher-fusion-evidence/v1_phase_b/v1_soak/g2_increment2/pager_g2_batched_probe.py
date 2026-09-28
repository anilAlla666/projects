#!/usr/bin/env python3
# INCREMENT-2 PROBE-FIRST: batched static-KV graph capture+replay over the PAGER region.
# Generalize the proven batch-1 capture (cipher_engine.PagerGraphModel) to batch dimension B: B DISTINCT agent
# sequences in ONE forward over the same pager-resident model. Risk = per-row cache_position/mask handling.
# GATE: per-agent KL=0 -- agent i's batched-burst output == its SOLO (batch-1 eager-static) run. Monotonic burst
# vehicle (single-step re-replay aliases the graph pool -- do NOT use it). Run CIPHER_RT_DISABLE_AUTO_INIT=1.
# First-probe simplification: B distinct prompts truncated to a COMMON length P -> no padding, clean attention.
import os, sys, time, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel

MODEL=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
B=int(sys.argv[3]) if len(sys.argv)>3 else 4

# B distinct agent prompts (genuinely different tokens = B distinct sequences sharing model weights)
PROMPTS=[
 "The history of artificial intelligence began in the 1950s, when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes, a civilization had",
 "The recipe calls for two cups of flour, a pinch of salt, and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty",
][:B]

eng=PagerGraphModel(CipherPager(), MODEL).load()   # weights -> pager cuMemMap region (proven brick)
m=eng.m; tok=eng.tok; V=eng.V; dev="cuda"
def clamp(t): return int(max(0,min(V-1,int(t))))

# tokenize + truncate each prompt to a COMMON length P (no padding)
toks_list=[tok(p, return_tensors="pt").input_ids[0] for p in PROMPTS]
P=min(t.shape[0] for t in toks_list)
pids=torch.stack([t[:P] for t in toks_list]).cuda()   # [B, P]
Lc=P+N+64
print(f"[batched probe] model={os.path.basename(MODEL)} B={B} P={P} N={N}", flush=True)

# ---- SOLO reference: each agent run alone (batch-1 eager-static), greedy + per-step top-2 margin ----
@torch.no_grad()
def solo(row_ids):
    rid=row_ids.unsqueeze(0)  # [1,P]
    c=StaticCache(config=m.config, max_cache_len=Lc)
    lg=m(rid, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1]
    out=[]; margins=[]
    for i in range(N):
        top2=lg.topk(2).values; margins.append(float(top2[0]-top2[1]))  # near-tie if small
        nt=int(lg.argmax()); out.append(clamp(nt))
        lg=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev),
             past_key_values=c, use_cache=True).logits[0,-1]
    return out, margins
solo_out=[solo(pids[b]) for b in range(B)]; solos=[s[0] for s in solo_out]; solo_margins=[s[1] for s in solo_out]

# ---- BATCHED: capture single decode step at batch B, monotonic burst ----
cache=StaticCache(config=m.config, max_cache_len=Lc)
logits=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True).logits  # [B,P,V]
t1=logits[:,-1].argmax(dim=-1)  # [B] first tokens
sin=torch.zeros(B,1,dtype=torch.long,device=dev); sin.copy_(t1.view(B,1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits  # [B,1,V]

@torch.no_grad()
def batched_burst():
    cur=t1.clone()                      # [B]
    seqs=[[clamp(cur[b])] for b in range(B)]
    sin.copy_(cur.view(B,1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize()
        nxt=slog[:,-1].argmax(dim=-1)   # [B]
        for b in range(B): seqs[b].append(clamp(nxt[b]))
        sin.copy_(nxt.view(B,1)); spos.add_(1)
    return seqs
t0=time.time(); bseqs=batched_burst(); torch.cuda.synchronize(); bt=time.time()-t0

# ---- per-agent KL=0: batched row b == solo b ----
allok=True
for b in range(B):
    mt=sum(x==y for x,y in zip(bseqs[b],solos[b])); fd=next((i for i,(x,y) in enumerate(zip(bseqs[b],solos[b])) if x!=y),-1)
    ok=(mt==N); allok=allok and ok
    margin=f" solo-margin@1stdiff={solo_margins[b][fd]:.4f} (near-tie if <~0.05)" if fd>=0 else ""
    print(f"  agent[{b}] batched vs solo: {mt}/{N} (1stdiff@{fd})  {'KL=0' if ok else 'DIVERGES'}{margin}", flush=True)
print(f"[RESULT batched B={B}] per-agent KL=0 ALL={allok}  batched_burst={bt*1000:.0f}ms ({B*N/bt:.0f} tok/s aggregate, {N/bt:.0f} tok/s/step)", flush=True)
print(f"[GATE batched B={B}] {'PASS' if allok else 'FAIL -- diagnose per-row cache_position/mask, NO scope-down'}", flush=True)
sys.stdout.flush(); os._exit(0 if allok else 1)
