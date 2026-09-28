#!/usr/bin/env python3
# D.8 correctness gate (Mem #11): enforcement is timing-only, so per-tenant
# output MUST be bit-identical with FAIRNESS+SHIELD armed (throttle firing) vs
# disarmed. Runs the SAME fixed-greedy decode twice in one process — disarmed
# first (env unset path), then armed via a forced-throttle config — and asserts
# logits byte-identical (max_logit_diff == 0, KL == 0). Any divergence = HARD STOP.
#
# Armed path uses CIPHER_FAIRNESS=on + BURST_MIN=0 + a real throttle so the
# sleep actually fires during the armed decode; identical logits then prove the
# CPU-sleep-before-submit changes only timing, never the computation.
import os, sys, json, ctypes, numpy as np

LIB = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/models/TinyLlama-1.1B"

# Two passes via re-exec: parent = disarmed reference, child (D8_KL_ARMED set) =
# armed with the throttle forced to fire. ONLY the disarmed parent strips the
# fairness env — the armed child must KEEP its CIPHER_FAIRNESS/FORCE_YIELD env
# (popping it here was the bug that made the armed pass silently disarm).
ARMED_PASS = "D8_KL_ARMED" in os.environ
if not ARMED_PASS:
    for k in ("CIPHER_FAIRNESS", "CIPHER_SHIELD", "CIPHER_SHIELD_BAND",
              "CIPHER_FAIRNESS_BURST_MIN", "CIPHER_FAIRNESS_THROTTLE_US",
              "CIPHER_FAIRNESS_FORCE_YIELD"):
        os.environ.pop(k, None)

lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
lib.cipher_rt_fairness_self_yields.restype = ctypes.c_ulonglong
lib.cipher_rt_fairness_armed.restype = ctypes.c_int

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
ids = tok("The history of computing began when", return_tensors="pt").input_ids.cuda()

def decode_logits(nsteps=24):
    logits = []
    with torch.no_grad():
        out = model(ids, use_cache=True); past = out.past_key_values
        cur = ids[:, -1:]
        for _ in range(nsteps):
            o = model(cur, past_key_values=past, use_cache=True); past = o.past_key_values
            lg = o.logits[0, -1].float().cpu().numpy()
            logits.append(lg)
            cur = torch.tensor([[int(lg.argmax())]], device="cuda")
    return np.stack(logits)

# 1) disarmed reference (g_armed should latch -1: env unset)
ref = decode_logits()
armed_before = lib.cipher_rt_fairness_armed()

# 2) arm in-process: the lazy init already latched disarmed, so force a fresh
#    armed run via a second process-like path is cleanest. Here we instead rely
#    on the orchestrator running this twice; if armed_before==0 we can still arm.
# Simplest robust approach: re-exec ourselves with armed env for the armed pass.
if not ARMED_PASS:
    np.save("/home/ubuntu/d8_kl_ref.npy", ref)
    env = dict(os.environ)
    env.update({"D8_KL_ARMED": "1", "CIPHER_FAIRNESS": "on",
                "CIPHER_FAIRNESS_FORCE_YIELD": "1",  # force the throttle to fire single-process
                "CIPHER_FAIRNESS_BURST_MIN": "0", "CIPHER_FAIRNESS_THROTTLE_US": "150"})
    import subprocess
    r = subprocess.run([sys.executable, __file__, LIB, MODEL], env=env)
    sys.exit(r.returncode)

# armed pass (this is the re-exec'd child)
arm = decode_logits()
ref = np.load("/home/ubuntu/d8_kl_ref.npy")
yields = int(lib.cipher_rt_fairness_self_yields())
armed = int(lib.cipher_rt_fairness_armed())
max_logit_diff = float(np.max(np.abs(arm - ref)))
lp = torch.log_softmax(torch.tensor(ref), -1)
lq = torch.log_softmax(torch.tensor(arm), -1)
kl = float((lp.exp() * (lp - lq)).sum(-1).max())
# The throttle MUST have fired (yields>0) or the test is vacuous (advisor catch):
# identical logits only prove timing-only-ness if the CPU sleep actually happened.
ok = (max_logit_diff == 0.0) and (armed == 1) and (yields > 0)
reason = "bit-identical with throttle firing (timing-only proven)" if ok else (
    "VACUOUS — throttle never fired (yields=0)" if yields == 0 else
    "DIVERGENCE — output changed under throttle")
print(json.dumps({"armed": armed, "self_yields": yields,
                  "max_logit_diff": max_logit_diff, "kl_max": kl,
                  "PASS": ok, "reason": reason}, indent=2))
print(f"D8 KL GATE: {'PASS — ' + reason if ok else 'FAIL (HARD STOP) — ' + reason} "
      f"(armed={armed}, throttle fired {yields}x)")
sys.exit(0 if ok else 1)
