#!/usr/bin/env python3
# Overhead bench: median forward time with CIPHER_GEOM_ATTN on vs off (the ~1ns-per-launch
# claim — verify by measurement). Same binary; the env toggles the gated counter.
import os, time, json, statistics, torch
from transformers import AutoModelForCausalLM
M = "/home/ubuntu/models/Mistral-7B-v0.1"
B, S = 64, 512
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = torch.randint(1, 31000, (B, S), device='cuda')
with torch.no_grad():
    for _ in range(5): m(ids); torch.cuda.synchronize()
ts = []
with torch.no_grad():
    for _ in range(40):
        t0 = time.time(); m(ids); torch.cuda.synchronize(); ts.append((time.time()-t0)*1000)
res = {"geom_attn_on": os.environ.get("CIPHER_GEOM_ATTN","0") not in ("0","",None),
       "median_ms": round(statistics.median(ts),2), "mean_ms": round(statistics.mean(ts),2),
       "min_ms": round(min(ts),2), "n": len(ts)}
json.dump(res, open(os.environ["OVH_OUT"], "w"), indent=2)
print("OVH geom_attn=%s median=%.2fms" % (res["geom_attn_on"], res["median_ms"]), flush=True)
