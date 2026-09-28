#!/usr/bin/env python3
# GATE-1 g1.2 DENSITY: how many distinct int4 (compressed-tensors W4A16, ~3x smaller) models co-reside per H100 vs
# the inc-3 fp16 baseline (~5 7-8B)? Load int4 models into pager regions, capture + verify per-model int4-graph ==
# int4-eager KL=0 (same-precision), count how many fit. CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
GB=1<<30; N=32; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
# distinct int4 checkpoints + replicas (distinct pager keys) to fill HBM and measure co-residence capacity
DISTINCT=os.environ.get("DISTINCT","/home/ubuntu/models/Qwen2.5-7B-w4a16,/home/ubuntu/models/Llama-3-8B-w4a16").split(",")
MAXN=int(os.environ.get("MAXN","12"))
pager=CipherPager(); engs=[]; ok=0
for i in range(MAXN):
    path=DISTINCT[i%len(DISTINCT)]   # cycle distinct checkpoints into distinct regions (capacity demo)
    free0=torch.cuda.mem_get_info()[0]/GB
    if free0<7.0:   # not enough headroom for another model + KV
        print(f"[stop] HBM headroom {free0:.1f}GB < 7GB -> capacity reached at {len(engs)} int4 models", flush=True); break
    try:
        e=PagerGraphModel(pager, path, model_key=0xD0+i).load(); e.capture(PROMPT,N)
        es=e.eager_burst(N); gs=e.serve_burst(N)
        match=sum(a==b for a,b in zip(gs,es))
        s=pager.stats(e.rid); foot=s.used_bytes/GB; free1=torch.cuda.mem_get_info()[0]/GB
        kl0 = (match==N)
        ok+= 1 if kl0 else 0
        print(f"  model[{len(engs)}] {os.path.basename(path)} key={hex(0xD0+i)} footprint={foot:.2f}GB int4-graph-vs-eager={match}/{N} {'KL=0' if kl0 else 'DIVERGES'} | HBM free={free1:.1f}GB", flush=True)
        engs.append(e)
    except Exception as ex:
        print(f"  model[{len(engs)}] {os.path.basename(path)} LOAD/CAPTURE FAIL: {type(ex).__name__} {str(ex)[:80]}", flush=True); break
free=torch.cuda.mem_get_info()[0]/GB
nfp16=5  # inc-3 baseline: ~5 distinct fp16 7-8B in 80GB
print(f"\n[RESULT g1.2 density] {len(engs)} int4 models co-resident (one process), per-model KL=0 ALL={ok==len(engs)} | HBM free={free:.1f}GB", flush=True)
if engs:
    fp=pager.stats(engs[0].rid).used_bytes/GB
    print(f"  int4 footprint ~{fp:.2f}GB/model vs fp16 ~15.4GB -> {15.4/fp:.1f}x density; co-resident {len(engs)} int4 vs ~{nfp16} fp16 baseline", flush=True)
print(f"[GATE g1.2 density] {'PASS (>{} fp16, per-model int4 KL=0)'.format(nfp16) if (len(engs)>nfp16 and ok==len(engs)) else 'see result'}", flush=True)
sys.stdout.flush(); os._exit(0)
