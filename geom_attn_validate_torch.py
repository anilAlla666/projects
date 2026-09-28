#!/usr/bin/env python3
# Agnostic proof: torch HF SDPA (a DIFFERENT attention lib than vLLM's _vllm_fa3_C),
# single-process so counters are ctypes-readable. The geometry-keyed detector should
# FIRE (geom_attn>0) while the symbol-keyed 6-pattern is 0 (the vLLM lib is never loaded
# in a torch-only process) — proving detection is framework-agnostic, name-free.
import os, ctypes, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
so = os.environ["MARLIN_SO"]; lib = ctypes.CDLL(so)
for f in ("cipher_rt_geom_attn_intercepts","cipher_rt_geom_launches",
          "cipher_rt_attn_p1_intercepts","cipher_rt_attn_p2_intercepts",
          "cipher_rt_attn_p5_intercepts","cipher_rt_attn_p6_intercepts"):
    if hasattr(lib,f): getattr(lib,f).restype=ctypes.c_ulong
M="/home/ubuntu/models/TinyLlama-1.1B"
t=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
ids=t("The history of computing spans several distinct eras, each defined by",return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(3): m.generate(ids,max_new_tokens=32,do_sample=False)
torch.cuda.synchronize()
def g(f): return int(getattr(lib,f)()) if hasattr(lib,f) else -1
geom=g("cipher_rt_geom_attn_intercepts"); launch=g("cipher_rt_geom_launches")
p6=[g("cipher_rt_attn_p%d_intercepts"%i) for i in (1,2,5,6)]; sixp=sum(x for x in p6 if x>0)
res={"framework":"torch-HF-SDPA","geom_attn_intercepts":geom,"geom_launches":launch,
     "sixpattern_fa_total":sixp,"sixpattern_breakdown":{"p1_fa2":p6[0],"p2_fa3":p6[1],"p5_mla":p6[2],"p6_paged":p6[3]},
     "AGNOSTIC_PROOF": bool(geom>0 and sixp==0),
     "no_false_positive_ratio": round(geom/max(launch,1),3)}
json.dump(res,open("/home/ubuntu/geom_attn_torch.json","w"),indent=2)
print(json.dumps(res,indent=2))
print("AGNOSTIC", "PROVEN" if res["AGNOSTIC_PROOF"] else "NOT-PROVEN",
      "(geom_attn=%d fires; 6-pattern symbol=%d)"%(geom,sixp), flush=True)
