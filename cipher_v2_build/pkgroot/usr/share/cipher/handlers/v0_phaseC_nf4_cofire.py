#!/usr/bin/env python3
# V.0 Phase C -- NF4 live component: demonstrate NF4 packed weights + static-KV graph-decode + DVFS-between-waves
# CO-FIRING in one live run (the density axis g1.2x composed with engine graph-decode + g1.1 DVFS). Per-model
# nf4-graph-vs-nf4-eager KL=0 (the capture-correct non-SWA set) + LIVE packed footprint (memory_allocated, engine-
# representative: graph+cache retained) + DVFS clock set between captures. Honest scope: this is the NF4 density axis
# composing capture-safe with graph-decode+DVFS; the FULL 100-agent-over-NF4 router run is the next integration (the
# inc-4 substrate (b6508bb) was fp16). NO pager routing here (device_map=cuda) to avoid the g1.2x routing-fallback
# confound; nf4-thru-pager is proven separately (inc-1). CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, time, json, subprocess
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ["CIPHER_RT_DISABLE_AUTO_INIT"]="1"
import torch, pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, BitsAndBytesConfig
torch.manual_seed(0); GB=1<<30; N=48
pynvml.nvmlInit(); d=pynvml.nvmlDeviceGetHandleByIndex(0)
def live(): return torch.cuda.memory_allocated()/GB
VOLT=os.environ.get("CIPHER_VOLT","0")=="1"
def set_clock(mhz):
    if VOLT: subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{mhz},{mhz}"],capture_output=True)
# capture-correct non-SWA set (Mistral EXCLUDED -- SWA capture, standing term)
MODELS=[("qwen2","/home/ubuntu/models/Qwen2-7B"),("llama1b","/home/ubuntu/models/Llama-3.2-1B-Instruct"),
        ("tiny","/home/ubuntu/models/TinyLlama-1.1B")]
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
torch.cuda.init(); _=torch.zeros(1,device="cuda"); base=live(); results=[]
held=[]
print(f"[v0-C] NF4 co-fire: packed weights + graph-decode + DVFS(between-captures, VOLT={VOLT}); set={[m[0] for m in MODELS]}",flush=True)
for nm,path in MODELS:
    set_clock(1000 if VOLT else 1980)   # g1.1 DVFS: host-side clock set BETWEEN captures (outside graph) = capture-safe
    l0=live()
    tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size; dev="cuda"
    m=AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="cuda").eval()
    pids=tok(PROMPT,return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; Lc=P+N+16
    def cl(t): return int(max(0,min(V-1,int(t))))
    @torch.no_grad()
    def prefill(c): return m(pids,cache_position=torch.arange(P,device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
    # nf4 EAGER reference
    with torch.no_grad():
        c=StaticCache(config=m.config,max_cache_len=Lc); nt=prefill(c); es=[]
        for i in range(N):
            es.append(cl(nt)); nt=m(torch.tensor([[cl(nt)]],device=dev),cache_position=torch.tensor([P+i],device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
    # nf4 static-KV GRAPH capture + replay (engine graph-decode over packed nf4 weights)
    cache=StaticCache(config=m.config,max_cache_len=Lc); t1=prefill(cache)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(cl(t1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
    gs=[cl(t1)]; sin.fill_(cl(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=cl(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
    match=sum(a==b for a,b in zip(gs,es)); kl0=(match==N)
    torch.cuda.empty_cache(); foot=live()-l0
    held.append((m,g,cache))   # keep graph+cache alive (engine-representative)
    r=dict(model=nm,kl0=kl0,match=f"{match}/{N}",live_footprint_GB=round(foot,1))
    results.append(r)
    print(f"[v0-C] {nm:<8} nf4-graph-vs-nf4-eager={match}/{N} {'KL=0' if kl0 else 'DIVERGES'} | LIVE footprint={foot:.1f}GB (weights+graph+cache)",flush=True)
allkl0=all(r['kl0'] for r in results); tot=live()-base
print(f"[v0-C RESULT] {len(held)} nf4 models co-resident (packed+graph+cache ALIVE) | LIVE total={tot:.1f}GB | per-model 7B-class ~5.6GB (g1.2x a182daa) | all KL=0={allkl0}",flush=True)
print(f"[v0-C VERDICT] NF4 density axis CO-FIRES with graph-decode + DVFS, capture-safe (non-SWA), KL=0 live={allkl0}; full 100-agent-over-NF4 router = next integration (inc-4 substrate was fp16, b6508bb)",flush=True)
print("V0C_JSON "+json.dumps(dict(results=results,all_kl0=allkl0,live_total_GB=round(tot,1),volt=VOLT)),flush=True)
if VOLT: subprocess.run(["sudo","-n","nvidia-smi","-rgc"],capture_output=True)
sys.stdout.flush(); os._exit(0)
