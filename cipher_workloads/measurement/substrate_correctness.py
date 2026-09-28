"""T4.5.1 S1.D — substrate correctness probe.

Loads TinyLlama, runs a fixed-prompt deterministic forward, prints
logits checksum + argmax-next-token. Compare with substrate-loaded
vs baseline run; both must produce identical output (no actuators
registered means pure passthrough).

Usage:
  python3 substrate_correctness.py
"""
import os, sys, hashlib, json
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PROMPT = "The future of GPU computing is to make every joule"

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
mdl.train(False)

p = tok(PROMPT, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = mdl(**p)
torch.cuda.synchronize()

logits = out.logits[0, -1].float().cpu()
# Hash with finite precision so float fuzz at <1e-6 doesn't cause false miscompares.
buf = (logits * 1000.0).round().to(torch.int32).numpy().tobytes()
sha = hashlib.sha256(buf).hexdigest()[:16]
top5 = torch.topk(logits, 5)
print(json.dumps({
    "model": MODEL,
    "prompt": PROMPT,
    "logits_sha256_16": sha,
    "argmax_token_id": int(logits.argmax()),
    "top5_token_ids": top5.indices.tolist(),
    "top5_logits": [round(x, 3) for x in top5.values.tolist()],
    "logits_shape": list(out.logits.shape),
}, indent=2))
