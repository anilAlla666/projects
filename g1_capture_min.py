#!/usr/bin/env python3
# Minimal parameterized capture+replay (the KL-verified no-warmup recipe) to ISOLATE the 8B replay assert axis.
# argv: <model_basename> <N> [instrument]. Reports KL vs eager-static, or dies on the device-assert (caught by the
# driver via exit code). GQA is NOT the axis (TinyLlama is GQA too) -> isolate model/size/N/structure cleanly here.
# instrument=1 -> monkeypatch index_copy_ to log (shape,dim,idx_min,idx_max) during one eager decode at an ADVANCED
# position, to name the position-dependent-small-dim index_copy that OOBs when the captured graph (frozen dim) replays.
import os, sys, torch
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
torch.manual_seed(0)
name=sys.argv[1]; N=int(sys.argv[2]); INSTR=len(sys.argv)>3 and sys.argv[3]=="instrument"
path=f"/home/ubuntu/models/{name}"
tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda").eval()
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device
L=P+N+64
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
if INSTR:
    orig=torch.Tensor.index_copy_; rec=[]
    def patched(self,dim,index,src,*a,**k):
        try: rec.append((tuple(self.shape),dim,int(index.min()),int(index.max())))
        except Exception: rec.append((tuple(self.shape),dim,"?","?"))
        return orig(self,dim,index,src,*a,**k)
    torch.Tensor.index_copy_=patched
    c=StaticCache(config=m.config, max_cache_len=L); prefill(c)
    # one decode at an ADVANCED position (within bounds) to see which index_copy's index tracks position vs its dim
    advp=P+40
    with torch.no_grad():
        m(torch.tensor([[clamp(0)+5]],device=dev), cache_position=torch.tensor([advp],device=dev), past_key_values=c, use_cache=True)
    torch.cuda.synchronize(); torch.Tensor.index_copy_=orig
    print(f"[{name}] index_copy_ calls during ONE eager decode @pos={advp} (cache_len={L}):",flush=True)
    seen=set()
    for (sh,dim,lo,hi) in rec:
        key=(sh,dim)
        if key in seen: continue
        seen.add(key)
        flag=" <-- OOB-RISK (dim<=adv pos)" if isinstance(hi,int) and dim<len(sh) and sh[dim]<=advp else ""
        print(f"    shape={sh} dim={dim} idx=[{lo}..{hi}] dim_size={sh[dim] if dim<len(sh) else '?'}{flag}",flush=True)
    sys.exit(0)
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=L); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt)); nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
@torch.no_grad()
def graph_replay():
    c=StaticCache(config=m.config, max_cache_len=L); t1=prefill(c)
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o=m(sin, cache_position=spos, past_key_values=c, use_cache=True); slog=o.logits
    toks=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); toks.append(nt); sin.fill_(nt); spos.add_(1)
    return toks
es=eager_static()
gs=graph_replay()   # if this device-asserts, the process dies here -> driver sees nonzero exit
match=sum(a==b for a,b in zip(gs,es))
fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y), -1)
print(f"[{name}] N={N} P={P}: graph vs eager-static = {match}/{N} (1stdiff@{fd})  {'PASS' if match==N else 'DIVERGES'}",flush=True)
sys.stdout.flush(); os._exit(0)
