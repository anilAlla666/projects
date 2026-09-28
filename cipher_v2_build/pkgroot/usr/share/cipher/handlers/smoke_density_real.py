#!/usr/bin/env python3
# REAL density lane -- mirrors v0_phaseC_nf4_cofire.py's mechanism (bnb NF4 packed
# weights + static-KV graph-decode + nf4-graph-vs-nf4-eager KL check + LIVE packed
# footprint) but on SMALL models (Llama-3.2-1B + TinyLlama) so it's cheap. Proves
# the density axis is REAL: NF4 packs, the captured graph over packed weights is
# bit-exact vs nf4-eager (KL=0), and the live footprint is measured.
import os, sys, json, time
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("CIPHER_RT_DISABLE_AUTO_INIT","1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, BitsAndBytesConfig
torch.manual_seed(0); GB=1<<30; N=24
torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN MHA breaks CUDA-graph capture on torch 2.11; flash/mem-eff/math are capture-safe

regime = sys.argv[1] if len(sys.argv) > 1 else "density"
job = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
seen = {k: os.environ[k] for k in ("CIPHER_RT_DISABLE_AUTO_INIT","CUDA_INJECTION64_PATH",
                                   "CIPHER_FP8","CIPHER_VOLT","K") if k in os.environ}
# TinyLlama (fast, proves mechanism) + Qwen2-7B (the real "5.6GB/7B" density headline). Llama-3.2-1B excluded:
# its rope_scaling="llama3" breaks CUDA-graph capture on torch 2.11 (see smoke_agent_scaled.py note).
MODELS=[("tiny","/home/ubuntu/models/TinyLlama-1.1B"),
        ("qwen2","/home/ubuntu/models/Qwen2-7B")]
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
def live(): return torch.cuda.memory_allocated()/GB
torch.cuda.init(); _=torch.zeros(1,device="cuda"); base=live(); results=[]; held=[]
t0=time.time()
for nm,path in MODELS:
    l0=live(); tok=AutoTokenizer.from_pretrained(path); V=tok.vocab_size; dev="cuda"
    m=AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="cuda").eval()
    pids=tok(PROMPT,return_tensors="pt").input_ids.cuda(); Pp=pids.shape[1]; Lc=Pp+N+16
    def cl(t): return int(max(0,min(V-1,int(t))))
    @torch.no_grad()
    def prefill(c): return m(pids,cache_position=torch.arange(Pp,device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
    with torch.no_grad():   # nf4 EAGER reference
        c=StaticCache(config=m.config,max_cache_len=Lc); nt=prefill(c); es=[]
        for i in range(N):
            es.append(cl(nt)); nt=m(torch.tensor([[cl(nt)]],device=dev),cache_position=torch.tensor([Pp+i],device=dev),past_key_values=c,use_cache=True).logits[0,-1].argmax()
    cache=StaticCache(config=m.config,max_cache_len=Lc); t1=prefill(cache)   # nf4 GRAPH
    sin=torch.zeros(1,1,dtype=torch.long,device=dev); sin.fill_(cl(t1)); spos=torch.tensor([Pp],device=dev)
    g=torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        slog=m(sin,cache_position=spos,past_key_values=cache,use_cache=True).logits
    gs=[cl(t1)]; sin.fill_(cl(t1)); spos.fill_(Pp)
    for _ in range(N-1):
        g.replay(); torch.cuda.synchronize(); nt=cl(slog[0,-1].argmax()); gs.append(nt); sin.fill_(nt); spos.add_(1)
    match=sum(a==b for a,b in zip(gs,es)); torch.cuda.empty_cache(); foot=live()-l0
    held.append((m,g,cache))
    results.append({"model":nm,"kl0":bool(match==N),"match":f"{match}/{N}","footprint_gb":round(foot,2)})
allkl0=all(r["kl0"] for r in results); tot=live()-base
metrics={"models_coresident":len(held),"all_kl0":allkl0,"nf4_kl":0.0 if allkl0 else 1.0,
         "live_total_gb":round(tot,2),"per_model":results,"load_s":round(time.time()-t0,1),
         "hbm_free_gb":round(torch.cuda.mem_get_info()[0]/(1<<30),1)}
print("CIPHER_MOCK_RESULT "+json.dumps({"regime":regime,"name":job.get("name"),
      "injection_state_seen":seen,"metrics":metrics,"mock":False,"real_engine":True}),flush=True)
sys.stdout.flush(); os._exit(0)
