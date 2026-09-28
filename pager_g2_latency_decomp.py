#!/usr/bin/env python3
# INCREMENT-2 LATENCY DECOMPOSITION: is CIPHER's ~5ms/step per-step cost bandwidth (HBM) or host overhead?
# (a) per-step loop  = g.replay(); sync(); argmax([B,V]); copy_   (what the scheduler does, autoregressive)
# (b) GPU-only       = G back-to-back g.replay() then ONE sync     (pure captured-graph GPU time, no host work)
# If (b) ~ bandwidth floor and (a) >> (b), the excess is HOST overhead (sync + CPU sampling), NOT HBM physics.
import os, sys, time, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from transformers import StaticCache
from cipher_engine import CipherPager, PagerGraphModel
MODEL=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
B=int(sys.argv[2]) if len(sys.argv)>2 else 8
G=int(sys.argv[3]) if len(sys.argv)>3 else 64
eng=PagerGraphModel(CipherPager(), MODEL).load(); m=eng.m; tok=eng.tok; dev="cuda"
import glob
wbytes=sum(os.path.getsize(f) for f in glob.glob(os.path.join(MODEL,"*.safetensors")))
floor=wbytes/2.5e12*1000
ids=tok("The history of artificial intelligence began in the 1950s when researchers first", return_tensors="pt").input_ids[0]
P=ids.shape[0]; Lc=P+G+8
pids=ids.unsqueeze(0).repeat(B,1).cuda()
cache=StaticCache(config=m.config, max_cache_len=Lc)
first=m(pids, cache_position=torch.arange(P,device=dev), past_key_values=cache, use_cache=True).logits[:,-1].argmax(dim=-1)
sin=torch.zeros(B,1,dtype=torch.long,device=dev); sin.copy_(first.view(B,1)); spos=torch.tensor([P],device=dev)
g=torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    slog=m(sin, cache_position=spos, past_key_values=cache, use_cache=True).logits
# (b) GPU-only FIRST (clean state): advancing burst (proven-safe) with NO host work between replays, CUDA events.
# Differs from (a) ONLY by the per-step sync + CPU argmax/copy -> (a)-(b) isolates host overhead.
sin.copy_(first.view(B,1)); spos.fill_(P); torch.cuda.synchronize()
e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(G-1): g.replay(); spos.add_(1)    # advance pos (proven), but no sync/argmax between
e1.record(); torch.cuda.synchronize(); b=e0.elapsed_time(e1)/(G-1)
# (a) per-step autoregressive loop (what the scheduler pays): replay + sync + CPU argmax + copy, advancing
sin.copy_(first.view(B,1)); spos.fill_(P); torch.cuda.synchronize(); t0=time.time()
for _ in range(G-1):
    g.replay(); torch.cuda.synchronize(); nxt=slog[:,-1].argmax(dim=-1).clamp_(0,eng.V-1); sin.copy_(nxt.view(B,1)); spos.add_(1)
a=(time.time()-t0)/(G-1)*1000
print(f"[latency-decomp {os.path.basename(MODEL)} B={B}] weight-bandwidth floor~{floor:.2f}ms/step", flush=True)
print(f"  (a) per-step loop (replay+sync+argmax+copy) = {a:.2f}ms/step   <- what the scheduler pays", flush=True)
print(f"  (b) GPU-only (back-to-back replay, 1 sync)   = {b:.2f}ms/step   <- pure captured-graph GPU time", flush=True)
print(f"  -> HOST overhead per step = (a)-(b) = {a-b:.2f}ms  ({'HOST-DOMINATED (sync+CPU sampling), NOT bandwidth' if (a-b)>b else 'GPU-bound'})", flush=True)
print(f"  -> GPU-only vs floor: {b:.2f} vs {floor:.2f}ms ({'~bandwidth-floored' if b<floor*2 else 'above floor'})", flush=True)
sys.stdout.flush(); os._exit(0)
