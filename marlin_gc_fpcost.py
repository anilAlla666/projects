#!/usr/bin/env python3
# Marlin GC-on-free — honest fingerprint hot-path cost (fp ON vs OFF, single tenant).
# Times steady-state Marlin-engaged decode generations; the env CIPHER_MARLIN_GC_FP
# selects the build's per-call content-fingerprint on/off. Report median gen time.
import os, time, json, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/home/ubuntu/models/Mistral-7B-v0.1"
t = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = t("The history of computing spans several distinct", return_tensors="pt").input_ids.cuda()
def gen(n=32):
    with torch.no_grad():
        m.generate(ids, max_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
for _ in range(5): gen()   # warm up Marlin engagement (steady state)
ts = []
for _ in range(30):
    t0 = time.time(); gen(); ts.append((time.time() - t0) * 1000)
res = {"fp_on": os.environ.get("CIPHER_MARLIN_GC_FP", "1") not in ("0", "n", "N"),
       "median_ms": round(statistics.median(ts), 2), "mean_ms": round(statistics.mean(ts), 2),
       "min_ms": round(min(ts), 2), "n": len(ts)}
json.dump(res, open(os.environ["FPCOST_OUT"], "w"), indent=2)
print("FPCOST fp_on=%s median=%.2fms mean=%.2fms" % (res["fp_on"], res["median_ms"], res["mean_ms"]), flush=True)
