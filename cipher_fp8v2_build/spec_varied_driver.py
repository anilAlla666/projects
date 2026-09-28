"""CP 2.4 sub-task (iii) — varied-prompt spec-decode lift measurement driver.

Replaces the single-prompt spec_measure_driver.py. Generates a FIXED token
budget (WL_MAX_NEW, default 128) for each of 5 representative prompts —
factual / narrative / code / reasoning / conversational — and records per-
prompt tok/s, generated-token count and (spec-on) acceptance rate.

The arm is selected by env: CIPHER_SPEC=0 -> Marlin-only baseline (install()
no-ops, stock generate); CIPHER_SPEC=on -> Marlin + speculative decode.
CIPHER_SPEC_DRAFT picks the draft policy ("ngram" for the Mistral arm, a
1B model path for the Llama arm). run_spec_varied.sh toggles the arm across
n=5 matched pairs; analyze_spec_varied.py computes the 25-point lift.

Prompt-set rationale: see cp_2_4/SPEC_DECODE_METHODOLOGY.md — workload
representativeness governs gate validity for spec decode (n-gram and draft-
model acceptance are both acutely workload-dependent).
"""
import os
import sys
import json
import time

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register

MODEL = os.environ["WL_MODEL"]
TENANT = os.environ.get("WL_TENANT_ID", "specvaried")
OUT_JSON = os.environ["OUT_JSON"]
MAX_NEW = int(os.environ.get("WL_MAX_NEW", "128"))

# Varied prompt set (CP 2.4 SPEC_DECODE_METHODOLOGY.md) — 5 genres, open-ended
# so a fixed 128-token budget will not hit EOS early.
PROMPTS = [
    # 0 factual
    "The Pacific Ocean is the largest ocean on Earth, covering approximately",
    # 1 narrative
    "Sarah opened the old letter with trembling hands. The handwriting was "
    "her grandmother's, and the date read",
    # 2 code
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot =",
    # 3 reasoning
    "If a train leaves Chicago at 3 PM traveling east at 60 mph and another "
    "leaves New York at 4 PM traveling west at 80 mph,",
    # 4 conversational
    "User: What are the main differences between supervised and unsupervised "
    "learning?\nAssistant: The main differences are",
]

tenant = tenant_register.register(TENANT)
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.eval()

# Operator-policy injection — install() no-ops when CIPHER_SPEC=0, so the same
# driver measures both the spec-off baseline and the spec-on arm.
import cipher_spec_decode
cipher_spec_decode.install()


def generate(prompt):
    """One timed 128-token generation; tok/s = generated / wall."""
    enc = tok(prompt, return_tensors="pt").to("cuda")
    plen = int(enc.input_ids.shape[-1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen_tokens = int(out.shape[-1]) - plen
    st = dict(cipher_spec_decode._last_stats)      # {} on the Marlin-only arm
    acc = st.get("accepted")
    prop = st.get("proposed")
    ar = (acc / prop) if (prop not in (None, 0)) else None
    return dt, gen_tokens, ar, st


# warm — run the full prompt set once at the timed token budget, untimed.
# Pays every one-time cost up front: Marlin lazy quantization (>>4
# observations), cuDNN/Marlin kernel JIT + autotune across the full decode
# kv_len range (a short warm left the first 128-token generate paying a
# ~3.7 s JIT — which corrupted prompt 0's lift), and, on the spec arm, the
# draft load + warm across every prompt shape.
for _p in PROMPTS:
    _we = tok(_p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        m.generate(**_we, max_new_tokens=MAX_NEW, do_sample=False,
                   pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

# Power-sentinel window — the composed-gate harness (run_composed.sh) keys
# its nvidia-smi power sampler off DECODE_START/DECODE_END so watts.csv covers
# exactly the timed decode loop (model load + warm + Marlin compile excluded).
# Harmless for the plain spec-arm runs — just writes an unread /tmp file.
SENTINEL = "/tmp/cipher_specvaried_%s.decode_window" % TENANT
with open(SENTINEL, "w") as _s:
    _s.write("DECODE_START %.3f\n" % time.time())

results = []
for i, p in enumerate(PROMPTS):
    dt, gt, ar, st = generate(p)
    toks = gt / max(dt, 1e-6)
    results.append({"prompt": i, "wall_s": round(dt, 4), "gen_tokens": gt,
                    "tok_s": round(toks, 4), "accept_rate": ar,
                    "accepted": st.get("accepted"),
                    "proposed": st.get("proposed"),
                    "rounds": st.get("rounds")})
    print("RESULT prompt=%d tok_s=%.3f gen_tokens=%d wall=%.3fs accept_rate=%s"
          % (i, toks, gt, dt, ("%.3f" % ar) if ar is not None else "n/a"),
          file=sys.stderr)

with open(SENTINEL, "a") as _s:
    _s.write("DECODE_END %.3f\n" % time.time())

out = {"model": MODEL, "spec": os.environ.get("CIPHER_SPEC", "1"),
       "draft": os.environ.get("CIPHER_SPEC_DRAFT", "ngram"),
       "tenant": TENANT, "max_new": MAX_NEW, "n_prompts": len(PROMPTS),
       "prompts": results}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("wrote " + OUT_JSON, file=sys.stderr)
