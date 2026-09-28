"""CP 2.4 spec-decode Mistral-arm GPU smoke + token-agreement check.

Same process, v2 lib active (Marlin on) throughout:
  (1) stock greedy generate          -> reference
  (2) cipher_spec_decode.install()
  (3) spec greedy generate           -> must be byte-identical to (1)

Both generates hit the identical Marlin-on target model, so spec decode —
which by construction reproduces the target's greedy output — must match
stock byte-for-byte at temp 0. Divergence => a spec-decode correctness bug.
"""
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
PROMPTS = ["The capital of France is",
           "In a shocking turn of events,",
           "Here is a list: one, two, three, one, two, three, one, two,"]
MAX_NEW = 64

# trust_remote_code=False (explicit): load weights+config only, never execute
# any python supplied by the model repo.
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda().eval()


def gen(prompt):
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0].tolist()


print("=== (0) Marlin warm-up — quantize all weights before the comparison ===",
      flush=True)
# Marlin quantizes each weight lazily, after 4 observations. Without this
# warm-up the stock reference (run first) sees weights mid-transition
# FP16->INT4 while the spec run (later) sees them fully INT4 — an unfair
# comparison, not a spec-decode bug. One full generation (~64 forwards) takes
# every weight past Marlin's 4-observation gate, so steps (1) and (3) below
# both run against the identical fully-quantized target.
_ = gen(PROMPTS[0])

print("=== (1) stock greedy — reference (Marlin fully warmed) ===", flush=True)
ref = [gen(p) for p in PROMPTS]

print("=== (2) install CIPHER speculative decode ===", flush=True)
import cipher_spec_decode
cipher_spec_decode.install()

print("=== (3) spec greedy ===", flush=True)
t0 = time.time()
spec = [gen(p) for p in PROMPTS]
print("spec wall: %.1fs for %d prompts" % (time.time() - t0, len(PROMPTS)))

all_match = True
for i in range(len(PROMPTS)):
    same = ref[i] == spec[i]
    all_match &= same
    print("prompt %d: %s  (ref %d tok / spec %d tok)"
          % (i, "MATCH" if same else "DIVERGE", len(ref[i]), len(spec[i])))
    if not same:
        n = min(len(ref[i]), len(spec[i]))
        d = next((j for j in range(n) if ref[i][j] != spec[i][j]), n)
        print("   first divergence @ index %d" % d)

print("=== SMOKE %s ===" % ("PASS — spec byte-identical to stock greedy"
                            if all_match else "FAIL — divergence"))
sys.exit(0 if all_match else 1)
