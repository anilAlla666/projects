#!/usr/bin/env python3
# Baseline serving probe: real Mistral-7B prefill + decode loop with KV cache, large batch to
# saturate. Measures prefill/decode/aggregate tok/s. Saturation is sampled separately via dmon.
import time, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
B   = int(sys.argv[1]) if len(sys.argv)>1 else 64
P   = int(sys.argv[2]) if len(sys.argv)>2 else 512
D   = int(sys.argv[3]) if len(sys.argv)>3 else 64
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation="sdpa").cuda().eval()
print(f"loaded; B={B} P={P} D={D}")
ids=torch.randint(0,32000,(B,P),device='cuda')

@torch.no_grad()
def run(reps=3):
    pref=[]; dec=[]
    for _ in range(reps):
        torch.cuda.synchronize(); t0=time.perf_counter()
        out=m(ids,use_cache=True); pkv=out.past_key_values
        torch.cuda.synchronize(); t1=time.perf_counter()
        nxt=out.logits[:,-1:].argmax(-1)
        for _ in range(D):
            out=m(nxt,past_key_values=pkv,use_cache=True); pkv=out.past_key_values
            nxt=out.logits[:,-1:].argmax(-1)
        torch.cuda.synchronize(); t2=time.perf_counter()
        pref.append(t1-t0); dec.append(t2-t1)
    import statistics
    pt=statistics.median(pref); dt=statistics.median(dec)
    pref_toks=B*P; dec_toks=B*D
    print(f"  prefill: {pt*1000:8.1f}ms  {pref_toks/pt:10.0f} tok/s  ({pt/ (B*P/1e6)*1e3:.2f}us/Mtok)")
    print(f"  decode : {dt*1000:8.1f}ms  {dec_toks/dt:10.0f} tok/s  ({dt/D*1000:.2f}ms/step, {B/(dt/D):.0f} tok/s)")
    print(f"  aggregate (pref+dec): {(pref_toks+dec_toks)/(pt+dt):10.0f} tok/s")
    print(f"  prefill share of wall: {pt/(pt+dt)*100:.1f}%  decode share: {dt/(pt+dt)*100:.1f}%")
    return dict(B=B,P=P,D=D,prefill_ms=pt*1000,decode_ms=dt*1000,
                prefill_toks_s=pref_toks/pt,decode_toks_s=dec_toks/dt,
                decode_ms_per_step=dt/D*1000,
                aggregate_toks_s=(pref_toks+dec_toks)/(pt+dt),
                prefill_wall_share=pt/(pt+dt))
# warm
with torch.no_grad():
    o=m(ids,use_cache=True); _=o.logits
    print("mem after load+prefill (GB):", torch.cuda.max_memory_allocated()/1e9)
r=run()
import json; json.dump(r,open(f'baseline_B{B}_P{P}_D{D}.json','w'),indent=1)
