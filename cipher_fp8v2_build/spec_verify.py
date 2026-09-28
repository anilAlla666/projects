"""Decide: the prompt-0 spec divergence — a spec_generate bug, or GPU
floating-point non-determinism at a logit near-tie?

Greedy decode on GPU is not guaranteed bit-deterministic run-to-run (matmul
reduction order). If a position's top-2 logits are within fp noise, two
separate forward invocations can flip the argmax. This script establishes the
stock-vs-stock determinism baseline (each prompt generated 3x with plain
greedy): if stock itself diverges at the same index, byte-identical is not an
achievable bar and spec is correct iff it matches to that same level.
"""
import sys
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
PROMPTS = ["The capital of France is",
           "In a shocking turn of events,",
           "Here is a list: one, two, three, one, two, three, one, two,"]
MAX_NEW = 64

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda().eval()


def gen(p):
    enc = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return o[0].tolist()


def firstdiff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


_ = gen(PROMPTS[0])                                # warm Marlin fully

print("=== stock-vs-stock determinism baseline (3 runs / prompt) ===", flush=True)
stock_nondet = False
for i, p in enumerate(PROMPTS):
    r = [gen(p) for _ in range(3)]
    d1, d2 = firstdiff(r[0], r[1]), firstdiff(r[0], r[2])
    det = (d1 < 0 and d2 < 0)
    stock_nondet |= not det
    print("prompt %d: run0vs1 diff@%d  run0vs2 diff@%d  -> %s"
          % (i, d1, d2, "deterministic" if det else "NON-DETERMINISTIC"))

ref = [gen(p) for p in PROMPTS]                    # fresh stock reference

import cipher_spec_decode
cipher_spec_decode.install()
print("=== stock-vs-spec ===", flush=True)
spec_ok = True
for i, p in enumerate(PROMPTS):
    d = firstdiff(ref[i], gen(p))
    spec_ok &= (d < 0)
    print("prompt %d: stock-vs-spec diff@%d  -> %s"
          % (i, d, "MATCH" if d < 0 else "diverge"))

print("=== VERDICT:",
      "spec byte-identical to stock (PASS)" if spec_ok
      else ("stock greedy is itself non-deterministic — byte-identical is not "
            "an achievable bar; judge spec against this baseline"
            if stock_nondet else "spec-specific divergence — investigate"),
      "===")
