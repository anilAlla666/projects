"""T4.3.x VOLT envelope measurement.

One workload (selected via env WL_MODEL + WL_BATCH), one arm (VOLT off
or on via env CIPHER_VOLT), one 60s decode window, power averaged
over the window. Designed to be invoked as a child of run_envelope.sh
which orchestrates n=3 matched pairs per condition.

Mirrors T4.3.2 wl01_decode_b1 structure: tight generate loop inside a
duration window via the shared run_for_duration helper.

Env knobs:
  WL_MODEL=mistral7b|tinyllama11b    (default mistral7b)
  WL_BATCH=N                         (default 1)
  WL_DURATION=60                     (seconds of decode loop)
  WL_TENANT_ID=str                   (for cipher tenant register)
  CIPHER_VOLT=on|off                 (VOLT actuator gate)
  CIPHER_VOLT_BATCH=N                (VOLT actuator's target-batch)
"""
import os, sys, time, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

MODEL_TAG = os.environ.get("WL_MODEL", "mistral7b")
WL_BATCH  = int(os.environ.get("WL_BATCH", "1"))
DURATION  = int(os.environ.get("WL_DURATION", "60"))
TENANT_ID = os.environ.get("WL_TENANT_ID", "envelope")

if MODEL_TAG == "tinyllama11b":
    MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PROMPT = "Hello, world."
    MAX_NEW = 32
elif MODEL_TAG == "mistral7b":
    MODEL_PATH = "/home/ubuntu/models/Mistral-7B-v0.1"
    PROMPT = "Hello, world."
    MAX_NEW = 32
else:
    raise SystemExit(f"unknown WL_MODEL={MODEL_TAG}")

tenant = tenant_register.register(TENANT_ID)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                          attn_implementation="sdpa").cuda()
m.train(False)

# Build batched prompt: replicate the same prompt WL_BATCH times so each
# decode iteration generates WL_BATCH × MAX_NEW tokens.
texts = [PROMPT] * WL_BATCH
enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
PROMPT_LEN = int(enc.input_ids.shape[-1])

def step():
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    new_tok = (out.shape[-1] - PROMPT_LEN) * WL_BATCH
    return 1, int(new_tok)

# Warm one call to JIT cuDNN / allocate KV. The Marlin NVRTC cubin compile
# (~12 s, first process only) also lands here, before the timed window.
with torch.no_grad():
    m.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()

# Decode-window sentinel (CP 2.4 DVFS harness fix, 2026-05-15). The power
# sampler in run_dvfs_sweep.sh keys off this file so watts.csv covers EXACTLY
# the decode loop — model load + Marlin compile + warmup are all done by now
# and must be excluded from the tok/W power average. DECODE_START is written
# the instant before the decode loop begins; DECODE_END the instant it ends.
# This is a dedicated file, not the run_for_duration progress log, because
# run_for_duration opens that log with mode "w" (truncate) — a sentinel there
# would be wiped. Fires at the identical logical moment in both off/on arms.
SENTINEL = f"/tmp/cipher_dvfs_{TENANT_ID}.decode_window"
with open(SENTINEL, "w") as _s:
    _s.write(f"DECODE_START {time.time():.3f}\n")
try:
    run_for_duration.run(step, DURATION, "ENV", tenant, "decodes")
finally:
    with open(SENTINEL, "a") as _s:
        _s.write(f"DECODE_END {time.time():.3f}\n")
