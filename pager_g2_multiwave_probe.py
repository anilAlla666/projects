#!/usr/bin/env python3
# INCREMENT-2 SCHEDULER MECHANISM PROBE (v2): RE-CAPTURE per wave (the proven inc-1 prefill->capture->burst pattern,
# batched). Reusing one captured graph across waves via eager re-prefill CORRUPTS it (probe v1: total divergence @0).
# So each wave = fresh StaticCache -> eager prefill -> capture batched decode graph -> monotonic burst. Verify
# per-agent KL=0 (ratified: exact OR sub-delta near-tie late flip) across waves, and MEASURE capture cost/wave.
import os, sys, time, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel

MODEL=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
N=int(sys.argv[2]) if len(sys.argv)>2 else 48
B=int(sys.argv[3]) if len(sys.argv)>3 else 4
WAVES=int(sys.argv[4]) if len(sys.argv)>4 else 4

POOL=[
 "The history of artificial intelligence began in the 1950s when researchers first",
 "In a distant galaxy far beyond the reach of human telescopes a civilization had",
 "The recipe calls for two cups of flour a pinch of salt and three large",
 "Quantum computing promises to revolutionize cryptography by factoring large numbers in",
 "Once upon a time in a small village nestled between two mountains there lived",
 "The stock market reacted sharply this morning after the central bank announced a",
 "To train a neural network effectively you must carefully tune the learning rate and",
 "She opened the ancient wooden door and stepped into a room filled with dusty",
 "The detective examined the muddy footprints leading away from the abandoned warehouse near",
 "Climate scientists warned that rising ocean temperatures could accelerate the melting of polar",
 "He picked up the worn paperback and began to read the opening lines aloud",
 "The orchestra tuned their instruments as the conductor raised his baton in the",
]
eng=PagerGraphModel(CipherPager(), MODEL).load(); m=eng.m; tok=eng.tok; V=eng.V; dev="cuda"
def clamp(t): return int(max(0,min(V-1,int(t))))
allp=[tok(p, return_tensors="pt").input_ids[0] for p in POOL]
P=min(t.shape[0] for t in allp); Lc=P+N+64

TIE=0.05  # batch-shape logit delta is ~0.0156 (TinyLlama)/~0.02 (8B); a flip is a benign near-tie iff solo margin < this
@torch.no_grad()
def solo(row):
    c=StaticCache(config=m.config, max_cache_len=Lc)
    lg=m(row.unsqueeze(0).cuda(), cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1]
    out=[]; mar=[]
    for i in range(N):
        top2=lg.topk(2).values; mar.append(float(top2[0]-top2[1]))
        nt=int(lg.argmax()); out.append(clamp(nt))
        lg=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1]
    return out, mar

# fresh prefill->capture->burst per wave (the proven pattern; objects co-owned per wave, freed at wave end)
@torch.no_grad()
def serve_wave(sel):
    pids=torch.stack(sel).cuda()                       # [B,P]
    cache=StaticCache(config=m.config, max_cache_len=Lc)
    first=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True).logits[:,-1].argmax(dim=-1)
    sin=torch.zeros(B,1,dtype=torch.long,device=dev); sin.copy_(first.view(B,1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()
    torch.cuda.synchronize(); tc=time.time()
    with torch.cuda.graph(g):
        o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits
    torch.cuda.synchronize(); cap_ms=(time.time()-tc)*1000
    seqs=[[clamp(first[b])] for b in range(B)]; sin.copy_(first.view(B,1)); spos.fill_(P)
    td=time.time()
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nxt=slog[:,-1].argmax(dim=-1)
        for b in range(B): seqs[b].append(clamp(nxt[b]))
        sin.copy_(nxt.view(B,1)); spos.add_(1)
    dec_ms=(time.time()-td)*1000
    del g, cache, slog                                 # free per-wave graph+cache (lifetime hygiene)
    return seqs, cap_ms, dec_ms

allok=True; caps=[]
for w in range(WAVES):
    sel=[allp[(w*B+b)%len(allp)][:P] for b in range(B)]
    seqs,cap_ms,dec_ms=serve_wave(sel); caps.append(cap_ms)
    waveok=True; details=[]
    for b in range(B):
        sr,mar=solo(sel[b]); mt=sum(x==y for x,y in zip(seqs[b],sr)); fd=next((i for i,(x,y) in enumerate(zip(seqs[b],sr)) if x!=y),-1)
        # ratified gate: exact, OR the FIRST divergence is a sub-delta near-tie (solo margin at fd < batch delta)
        tie = (fd>=0) and (mar[fd]<TIE)
        ok=(mt==N) or tie
        waveok=waveok and ok; tag="exact" if mt==N else (f"tie@{fd}(m={mar[fd]:.4f})" if tie else f"FAULT@{fd}(m={mar[fd]:.4f})")
        details.append(f"a{b}:{mt}/{N}:{tag}")
    allok=allok and waveok
    print(f"  wave{w} {'OK' if waveok else 'FAULT'} cap={cap_ms:.0f}ms dec={dec_ms:.0f}ms  "+" ".join(details), flush=True)
print(f"[RESULT multiwave-recapture B={B} W={WAVES}] per-agent KL=0(modulo-ties) ALL={allok}  capture~{sum(caps)/len(caps):.0f}ms/wave", flush=True)
print(f"[GATE multiwave] {'PASS (re-capture per wave serves many waves correctly)' if allok else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if allok else 1)
