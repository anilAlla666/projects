#!/usr/bin/env python3
# GATE-1 g1.2 PROBE: does an int4 (compressed-tensors W4A16 -> Marlin/Machete) model capture into the engine's
# static-KV torch.cuda.graph and replay KL=0 vs eager (SAME-PRECISION: int4-graph == int4-eager, NOT fp16-bit-exact)?
# STEP 1 (this probe): isolate Marlin capture-safety in torch.cuda.graph (NO pager yet). If Marlin fires
# capture-illegal ops (like Koopman's cusolver) -> CAPTURE-FAIL -> engine-incompatible (diagnose, do not force).
# STEP 2 (next): add the pager region routing. CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
PATH=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/Qwen2.5-7B-w4a16"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
tok=AutoTokenizer.from_pretrained(PATH); V=tok.vocab_size
m=AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float16, device_map="cuda").eval()
qc=getattr(m.config,"quantization_config",None)
print(f"[load] {os.path.basename(PATH)} quant={qc if qc is None else getattr(qc,'quant_method',type(qc).__name__)} dtype={next(m.parameters()).dtype}", flush=True)
prompt="The history of artificial intelligence began in the 1950s, when researchers first"
pids=tok(prompt, return_tensors="pt").input_ids.cuda(); P=pids.shape[1]; dev=pids.device; Lc=P+N+64
def clamp(t): return int(max(0,min(V-1,int(t))))
@torch.no_grad()
def prefill(c): return m(pids, cache_position=torch.arange(P,device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
# eager-int4 reference (same-precision baseline)
@torch.no_grad()
def eager_static():
    c=StaticCache(config=m.config, max_cache_len=Lc); nt=prefill(c); out=[]
    for i in range(N):
        out.append(clamp(nt))
        nt=m(torch.tensor([[clamp(nt)]],device=dev), cache_position=torch.tensor([P+i],device=dev), past_key_values=c, use_cache=True).logits[0,-1].argmax()
    return out
es=eager_static()
# capture static-KV decode graph over the int4 model (the Marlin-in-capture test)
cache=StaticCache(config=m.config, max_cache_len=Lc); t1=prefill(cache)
sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(clamp(t1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
try:
    with torch.no_grad(), torch.cuda.graph(g):
        o=m(sin, cache_position=spos, past_key_values=cache, use_cache=True); slog=o.logits
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\n[CAPTURE-FAIL int4 Marlin] {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("  -> int4 Marlin kernel fires capture-illegal ops in torch.cuda.graph -> ENGINE-INCOMPATIBLE (like Koopman). Diagnose, do NOT force.", flush=True)
    sys.stdout.flush(); os._exit(2)
gs=[clamp(t1)]; sin.fill_(clamp(t1)); spos.fill_(P)
for _ in range(N-1):
    g.replay(); torch.cuda.synchronize(); nt=clamp(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
match=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
print(f"\n=== g1.2 PROBE int4 static-KV capture ({os.path.basename(PATH)}, N={N}) ===", flush=True)
print(f"  FIDELITY graph-int4 vs eager-int4 (SAME-PRECISION): {match}/{N} (1stdiff@{fd})  {'KL=0 (Marlin capture-safe in the engine torch.cuda.graph)' if match==N else 'DIVERGES'}", flush=True)
print(f"  VERDICT: {'MARLIN CAPTURE-SAFE in engine path -- proceed to pager + density' if match==N else 'diagnose divergence (NO scope-down)'}", flush=True)
sys.stdout.flush(); os._exit(0 if match==N else 1)
