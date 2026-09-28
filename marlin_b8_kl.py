#!/usr/bin/env python3
# Teacher-forced correctness gate: off (bf16 ref) vs on (Marlin INT4), matched inputs.
import numpy as np, json, sys
OUT = sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/marlin_b8_torch"
off = np.load(f"{OUT}/off_logits.npy").astype(np.float64)   # [B,T,V] reference
on  = np.load(f"{OUT}/on_logits.npy").astype(np.float64)    # [B,T,V] Marlin
assert off.shape == on.shape, (off.shape, on.shape)
B,T,V = off.shape
def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e/e.sum(-1, keepdims=True)
P = softmax(off); Q = softmax(on)
amo = off.argmax(-1); amn = on.argmax(-1)
argmax_agree = float((amo == amn).mean())
kl = (P * (np.log(P+1e-12) - np.log(Q+1e-12))).sum(-1)   # KL(ref||marlin) per (b,t)
res = {
  "shape_BTV": [B,T,V],
  "argmax_agreement": round(argmax_agree,6),
  "argmax_flips": int((amo != amn).sum()), "positions": B*T,
  "kl_mean": round(float(kl.mean()),6), "kl_max": round(float(kl.max()),6),
  "kl_p50": round(float(np.median(kl)),6), "kl_p99": round(float(np.percentile(kl,99)),6),
  "PASS_kl0_hardstop": bool(argmax_agree == 1.0),
}
json.dump(res, open(f"{OUT}/kl_gate.json","w"), indent=2)
print(json.dumps(res))
