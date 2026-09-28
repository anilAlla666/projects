"""CP 2.4 sub-task (iii) — Mistral-arm spec-decode lift measurement driver.

Loads Mistral-7B, installs CIPHER speculative decode (operator-policy; no-op
if CIPHER_SPEC=0), runs a B=1 decode loop for WL_DURATION seconds. tok/s is
read from the run_for_duration progress 'end' line. run_spec_measure.sh
toggles CIPHER_SPEC across n=5 matched pairs to measure spec-on vs spec-off
(both CIPHER_MARLIN=on) — the lift over the Marlin-alone baseline.
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register
import run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "60"))
TENANT = os.environ.get("WL_TENANT_ID", "specmeasure")
MODEL = "/home/ubuntu/models/Mistral-7B-v0.1"
PROMPT = "Hello, world."
MAX_NEW = 32

tenant = tenant_register.register(TENANT)
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda().eval()

# Operator-policy injection — install() no-ops when CIPHER_SPEC=0, so the same
# driver measures both the spec-off baseline and the spec-on arm.
import cipher_spec_decode
cipher_spec_decode.install()

enc = tok(PROMPT, return_tensors="pt").to("cuda")
PROMPT_LEN = int(enc.input_ids.shape[-1])


def step():
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 1, int(out.shape[-1] - PROMPT_LEN)


# warm one call — JITs cuDNN, allocates KV, and (Marlin) quantizes the weights
with torch.no_grad():
    m.generate(**enc, max_new_tokens=8, do_sample=False,
               pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

run_for_duration.run(step, DURATION, "SPEC", tenant, "decodes")
