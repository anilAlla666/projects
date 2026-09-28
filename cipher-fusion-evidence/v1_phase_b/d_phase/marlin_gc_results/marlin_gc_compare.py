#!/usr/bin/env python3
# Compare two saved-logit tag sets (from d9_fp8_close_offreg.py) for byte-identical.
import sys, json, numpy as np
MODELS = ["tinyllama", "llama32_1b", "mistral7b", "llama31_8b"]
ta, tb, label = sys.argv[1], sys.argv[2], sys.argv[3]
out = {"label": label, "tag_a": ta, "tag_b": tb, "models": {}}
allok = True
for name in MODELS:
    a = np.load(f"/home/ubuntu/d9clz_{name}_{ta}.npy")
    b = np.load(f"/home/ubuntu/d9clz_{name}_{tb}.npy")
    mad = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    # KL over the last-step softmax (a as ref)
    def kl(p_log, q_log):
        p = np.exp(p_log - p_log.max()); p /= p.sum()
        q = np.exp(q_log - q_log.max()); q /= q.sum()
        m = p > 0
        return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], 1e-12, None))))
    klv = kl(a[-1], b[-1])
    bi = (mad == 0.0)
    out["models"][name] = {"max_abs_diff": mad, "kl_nats": klv, "byte_identical": bi}
    allok = allok and bi
out["byte_identical_all"] = allok
out["n_of_n"] = f"{sum(1 for m in out['models'].values() if m['byte_identical'])}/{len(MODELS)}"
print(json.dumps(out, indent=2))
json.dump(out, open(f"/home/ubuntu/marlin_gc_{label}.json", "w"), indent=2)
sys.exit(0 if allok else 1)
