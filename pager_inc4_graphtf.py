#!/usr/bin/env python3
# DECISIVE (advisor): teacher-force the GRAPH path (serve_wave's B=8 captured graph), not eager. Feed row p2 the
# SOLO tokens each step (no free-run -> kills the cascade), capture the GRAPH's per-step logits for p2, compare to
# solo logits. Flat ~0.03 incl. step ~22 -> graph computes solo's logits correctly, divergence was a downstream
# near-tie cascade, gate over-flagged. A spike where solo margin > delta -> a REAL B=8 graph divergence (not benign).
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager
from cipher_engine_batched import WaveServer
q=WaveServer(CipherPager(),"/home/ubuntu/models/Qwen2-7B",0xC2); m=q.m; dev="cuda"; V=q.V
POOL=["The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty"]
rows=[q.tok(p,return_tensors="pt").input_ids[0][:16] for p in POOL]; Pc=min(r.shape[0] for r in rows); rows=[r[:Pc] for r in rows]
G=48; TGT=2
def cl(t): return int(max(0,min(V-1,int(t))))
# solo p2: per-step tokens + logits (batch-1 eager)
@torch.no_grad()
def solo(row):
    P=row.shape[0]; Lc=P+G+8; c=StaticCache(config=m.config,max_cache_len=Lc)
    lg=m(row.unsqueeze(0).to(dev),cache_position=torch.arange(P,device=dev),past_key_values=c,use_cache=True).logits[0,-1]
    tk=[]; lo=[]
    for i in range(G):
        lo.append(lg.float().clone()); nt=cl(lg.argmax()); tk.append(nt)
        lg=m(torch.tensor([[nt]],device=dev),cache_position=torch.tensor([P+i],device=dev),past_key_values=c,use_cache=True).logits[0,-1]
    return tk,lo
st,sl=solo(rows[TGT])
# GRAPH B=8 (mirror serve_wave capture), teacher-force EVERY row to its own solo tokens (no free-run)
allsolo=[solo(rows[b])[0] for b in range(8)]
P=rows[0].shape[0]; Lc=P+G+8
pids=torch.stack([r.to(dev) for r in rows])
cache=StaticCache(config=m.config,max_cache_len=Lc)
first=m(pids,cache_position=torch.arange(P,device=dev),past_key_values=cache,use_cache=True).logits[:,-1].argmax(dim=-1)
sin=torch.zeros(8,1,dtype=torch.long,device=dev); sin.copy_(first.view(8,1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
# teacher-force ONLY row TGT to solo; OTHER rows FREE-RUN (their own argmax) -> replicates the failing free-run
# context for the co-tenants, while keeping TGT on the solo path so its logits are directly comparable.
sin.copy_(first.view(8,1)); spos.fill_(P); deltas=[]
with torch.no_grad():
    for i in range(G-1):
        g.replay(); torch.cuda.synchronize()
        d=float((slog[TGT,-1].float()-sl[i+1]).abs().max()); deltas.append(d)
        nxt=slog[:,-1].argmax(dim=-1).clone()          # other rows free-run
        nxt[TGT]=cl(allsolo[TGT][i+1])                 # ONLY TGT teacher-forced to solo
        sin.copy_(nxt.view(8,1)); spos.add_(1)
mx=max(deltas); amax=deltas.index(mx)
print(f"[graph-TF p2] per-step graph-vs-solo logit max|Δ| over {len(deltas)} steps: max={mx:.3f} at step {amax}", flush=True)
print(f"  first 30: "+" ".join(f"{d:.3f}" for d in deltas[:30]), flush=True)
print(f"  solo margin at step {amax}: {float(sl[amax+1].topk(2).values[0]-sl[amax+1].topk(2).values[1]):.3f}", flush=True)
verdict = "CLEAN -- graph computes solo's logits within batch-shape delta; inc-4 divergence was a near-tie CASCADE, gate over-flagged" if mx<0.2 else f"REAL graph divergence at step {amax} (|Δ|={mx:.2f})"
print(f"[VERDICT graph-TF] {verdict}", flush=True)

# FREE-RUNNING graph dump: where does p2 free-run first diverge from solo, and the TRUE solo margin + graph-vs-solo
# logit delta AT that step (with matched preceding tokens -> meaningful)?
sin.copy_(first.view(8,1)); spos.fill_(P); ftoks=[cl(first[TGT])]; fdelta=[]
with torch.no_grad():
    for i in range(G-1):
        g.replay(); torch.cuda.synchronize()
        fdelta.append(float((slog[TGT,-1].float()-sl[i+1]).abs().max()))
        nxt=slog[:,-1].argmax(dim=-1).clone(); ftoks.append(cl(nxt[TGT])); sin.copy_(nxt.view(8,1)); spos.add_(1)
fr=next((i for i,(x,y) in enumerate(zip(ftoks,st)) if x!=y),-1)
print(f"[free-run graph p2] 1stdiff vs solo @{fr}", flush=True)
if fr>=0:
    sm=float(sl[fr].topk(2).values[0]-sl[fr].topk(2).values[1])
    dl=fdelta[fr-1] if 0<fr<=len(fdelta) else float('nan')
    print(f"  at fd={fr}: solo top1-2 margin={sm:.4f}  graph-vs-solo logit |Δ| (matched-context, step {fr})={dl:.4f}  -> {'SUB-DELTA NEAR-TIE (benign, gate over-flags)' if sm<0.05 else 'solo CONFIDENT -- but |Δ|={:.3f} << margin so cannot be a real flip unless earlier cascade'.format(dl)}", flush=True)
sys.stdout.flush(); os._exit(0)
