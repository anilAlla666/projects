#!/usr/bin/env python3
# GATE-1 g1.2x PROBE STEP 2: does a bnb-NF4 model CAPTURE into the engine's static-KV torch.cuda.graph and replay
# KL=0 vs eager (SAME-PRECISION nf4-graph == nf4-eager)? STEP 1 proved nf4 stays packed (6.3GB, forward +0.1GB).
# Risk: bnb's fused dequant kernel may fire capture-illegal ops (like Koopman cusolver / act-order g_idx). This probe
# (no pager) isolates bnb capture-safety; if CAPTURE-FAIL -> engine-incompatible. CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, BitsAndBytesConfig
PATH=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/Qwen2-7B"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
tok=AutoTokenizer.from_pretrained(PATH); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(PATH, quantization_config=bnb, device_map="cuda").eval()
pb=(sum(p.numel()*p.element_size() for p in m.parameters())+sum(x.numel()*x.element_size() for x in m.buffers()))/(1<<30)
print(f"[load] {os.path.basename(PATH)} bnb-NF4 packed footprint={pb:.2f}GB", flush=True)
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device; Lc=P+N+64
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=Lc); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt))
        nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
es=eager_static()
cache=StaticCache(config=m.config, max_cache_len=Lc); t1=prefill(cache)
sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
try:
    with torch.no_grad(), torch.cuda.graph(g):
        o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\n[CAPTURE-FAIL bnb-NF4] {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("  -> bnb fused-dequant fires capture-illegal ops in torch.cuda.graph -> ENGINE-INCOMPATIBLE. Diagnose, do NOT force.", flush=True)
    sys.stdout.flush(); os._exit(2)
gs=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
for _ in range(N-1):
    g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
match=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
print(f"\n=== g1.2x STEP2 bnb-NF4 static-KV capture ({os.path.basename(PATH)}, N={N}) ===", flush=True)
print(f"  FIDELITY nf4-graph vs nf4-eager (SAME-PRECISION): {match}/{N} (1stdiff@{fd})  {'KL=0 (bnb NF4 CAPTURE-SAFE in engine torch.cuda.graph)' if match==N else 'DIVERGES'}", flush=True)
print(f"  VERDICT: {'bnb NF4 CAPTURE-SAFE + PACKED (6.3GB) -> density REAL in engine, proceed to pager + co-residence' if match==N else 'diagnose divergence (NO scope-down)'}", flush=True)
sys.stdout.flush(); os._exit(0 if match==N else 1)
