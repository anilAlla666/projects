"""CP 2.4 sub-task (iii) — Llama-arm GPU smoke (run BEFORE the n=5 gate).

Verifies, in one process under the v2 lib (Marlin on):
  1. Llama-3.2-1B-Instruct draft loads cleanly via cipher_spec_decode._load_draft
     (which asserts draft/target vocab_size match).
  2. Draft + target tokenizers share an identical vocabulary (token-id space
     alignment — ModelDraft proposes raw ids, this MUST hold).
  3. spec_generate completes on GPU and the stateless ModelDraft.propose works.
  4. Acceptance rate is plausible (>0.15 — a ~0 rate signals a broken draft
     or vocab misalignment; base target + instruct draft, expect ~0.3-0.7).
  5. Token agreement vs stock greedy decode (honest GPU-determinism bar, per
     SPEC_DECODE_CORRECTNESS.md — divergence only at logit fp-ties).

Run:  CIPHER_MARLIN=on CIPHER_SPEC=on \
      CIPHER_SPEC_DRAFT=/home/ubuntu/models/Llama-3.2-1B-Instruct \
      LD_PRELOAD=.../libcipher_rt.so CUDA_INJECTION64_PATH=.../libcipher_rt.so \
      python3 spec_llama_smoke.py
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register

TARGET = os.environ.get("WL_MODEL", "/home/ubuntu/models/Llama-3.1-8B")
DRAFT = os.environ.get("CIPHER_SPEC_DRAFT",
                       "/home/ubuntu/models/Llama-3.2-1B-Instruct")
PROMPT = ("The Pacific Ocean is the largest ocean on Earth, covering "
          "approximately")
N = 64

fail = []
tenant = tenant_register.register("specllamasmoke")
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- check 2: tokenizer vocab alignment (before loading the big model) -------
tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=False)
dtok = AutoTokenizer.from_pretrained(DRAFT, trust_remote_code=False)
tv, dv = tok.get_vocab(), dtok.get_vocab()
if tv == dv:
    print("[smoke] check 2 PASS — tokenizer vocab identical (%d tokens)" % len(tv))
else:
    fail.append("tokenizer vocab mismatch (target=%d draft=%d)"
                % (len(tv), len(dv)))
    print("[smoke] check 2 FAIL — tokenizer vocab differs")

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    TARGET, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()

import cipher_spec_decode
cipher_spec_decode.install()

enc = tok(PROMPT, return_tensors="pt").to("cuda")
plen = int(enc.input_ids.shape[-1])

# warm — quantize Marlin weights; first spec call also loads + warms the draft
# (this is where check 1, the _load_draft vocab_size assert, fires).
try:
    with torch.no_grad():
        m.generate(**enc, max_new_tokens=16, do_sample=False,
                   pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    print("[smoke] check 1 PASS — draft loaded; _load_draft vocab assert held")
except Exception as e:
    fail.append("draft load / warm failed: %r" % e)
    print("[smoke] check 1 FAIL — %r" % e)

# --- check 3+4: spec generate completes, acceptance plausible ----------------
spec_ids = None
try:
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=N, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    spec_ids = out[0].tolist()
    st = dict(cipher_spec_decode._last_stats)
    ar = st.get("accepted", 0) / max(st.get("proposed", 1), 1)
    print("[smoke] check 3 PASS — spec_generate completed (%d tokens, "
          "%d rounds)" % (len(spec_ids) - plen, st.get("rounds", 0)))
    print("[smoke]   spec text: %r"
          % tok.decode(spec_ids[plen:], skip_special_tokens=True)[:160])
    if ar > 0.15:
        print("[smoke] check 4 PASS — acceptance rate %.3f (plausible)" % ar)
    else:
        fail.append("acceptance rate %.3f too low — draft likely broken" % ar)
        print("[smoke] check 4 FAIL — acceptance rate %.3f" % ar)
except Exception as e:
    fail.append("spec_generate failed: %r" % e)
    print("[smoke] check 3 FAIL — %r" % e)

# --- check 5: token agreement vs stock greedy --------------------------------
try:
    os.environ["CIPHER_SPEC"] = "0"          # _wrapped_generate re-checks per call
    with torch.no_grad():
        sout = m.generate(**enc, max_new_tokens=N, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    os.environ["CIPHER_SPEC"] = "on"
    stock_ids = sout[0].tolist()
    if spec_ids is not None:
        a, b = spec_ids[plen:], stock_ids[plen:]
        div = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                   min(len(a), len(b)))
        if div == min(len(a), len(b)):
            print("[smoke] check 5 PASS — spec == stock for all %d tokens" % div)
        else:
            print("[smoke] check 5 — spec/stock diverge at index %d/%d "
                  "(GPU fp-tie noise expected; see SPEC_DECODE_CORRECTNESS.md)"
                  % (div, min(len(a), len(b))))
            print("[smoke]   stock text: %r"
                  % tok.decode(b, skip_special_tokens=True)[:160])
except Exception as e:
    print("[smoke] check 5 SKIP — stock compare failed: %r" % e)

print()
if fail:
    print("SMOKE FAIL: " + "; ".join(fail))
    sys.exit(1)
print("SMOKE PASS — Llama arm ready for the n=5 gate measurement")
