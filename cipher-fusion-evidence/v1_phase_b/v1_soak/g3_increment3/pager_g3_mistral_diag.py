#!/usr/bin/env python3
# Isolate Mistral's batch-1 graph-vs-eager divergence: benign near-tie FP flip, or a real capture bug (SWA/arch)?
# Compare graph-replay logits vs eager-static logits per step; report solo top1-top2 margin AND max|graph-eager Δ|
# at the first token divergence. FP-small delta + sub-delta margin => benign near-tie (ratified). Large delta => bug.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel
NAME=sys.argv[1] if len(sys.argv)>1 else "Mistral-7B-v0.1"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
e=PagerGraphModel(CipherPager(), f"/home/ubuntu/models/{NAME}").load()
m=e.m; tok=e.tok; V=e.V; dev="cuda"
print(f"[{NAME}] sliding_window={getattr(m.config,'sliding_window',None)} arch={m.config.architectures}", flush=True)
def clamp(t): return int(max(0,min(V-1,int(t))))
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; Lc=P+N+64
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1]
# eager-static: tokens + per-step logits
@torch.no_grad()
def eager():
    c=StaticCache(config=m.config, max_cache_len=Lc); lg=prefill(c); toks=[]; logs=[]
    for i in range(N):
        logs.append(lg.float().clone()); nt=clamp(lg.argmax()); toks.append(nt)
        lg=m(torch.tensor([[nt]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1]
    return toks, logs
es,elog=eager()
# graph replay (inc-1 recipe): tokens + per-step logits
cache=StaticCache(config=m.config, max_cache_len=Lc); t1=prefill(cache).argmax()
sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    slog=m(sin, cache_position=spos, past_key_values=cache, use_cache=True).logits
gs=[clamp(t1)]; glog=[]; sin.fill_(clamp(t1)); spos.fill_(P)
for _ in range(N-1):
    g.replay(); torch.cuda.synchronize(); glog.append(slog[0,-1].float().clone()); nt=clamp(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
# compare
mt=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
# at each step, graph-vs-eager logit delta (only valid while token sequences still agree; after divergence the KV diverges)
deltas=[float((glog[i]-elog[i+1]).abs().max()) for i in range(min(len(glog),len(elog)-1))]   # glog[i] is step i+1's logit (gs[0]=t1 from prefill)
# solo margins
mar=[float((l.topk(2).values[0]-l.topk(2).values[1])) for l in elog]
print(f"[{NAME}] graph vs eager: {mt}/{N} (1stdiff@{fd})", flush=True)
if fd>=0:
    md_at = deltas[fd-1] if 0<fd<=len(deltas) else float('nan')
    print(f"  at 1stdiff token {fd}: solo top1-top2 margin={mar[fd]:.4f}  graph-vs-eager logit max|Δ|@prev-steps~{md_at:.4f}", flush=True)
    pre_delta=max(deltas[:fd]) if fd>0 else 0.0
    print(f"  max graph-vs-eager logit |Δ| over agreeing steps (0..{fd-1}) = {pre_delta:.4f}", flush=True)
    verdict = "BENIGN near-tie FP flip (margin < graph-eager FP delta)" if (fd>0 and mar[fd] < max(pre_delta,1e-3)*2) else "REAL capture bug (margin >> FP delta)"
    print(f"  VERDICT: {verdict}", flush=True)
else:
    print(f"  EXACT KL=0", flush=True)
sys.stdout.flush(); os._exit(0)
