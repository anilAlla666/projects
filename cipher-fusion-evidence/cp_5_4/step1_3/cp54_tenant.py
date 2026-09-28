"""CP 5.4 Step 1.3 Phase D — single-tenant TinyLlama harness (THROWAWAY).

Loads TinyLlama-1.1B fp16, runs a deterministic teacher-forced forward pass
plus a 24-token greedy decode on a fixed prompt, and saves logits + generated
token ids to $OUT. Used to compare the new libcipher_rt (SHARED / PARTITION)
against the a7ac8e97 baseline (Tests A/B), and as a quick CP 5.4 tenant for
the reaper/disjointness check (FAST=1 — load + one forward + exit).

The qos_class is declared to libcipher_rt purely via env (CP 5.4 Step 1.3 Q2):
  CIPHER_QOS_CLASS = partition | shared | pool   (default shared)
  CIPHER_SM_COUNT  = <int>                       (PARTITION only)
This script does not set them — the caller does — so it is qos-agnostic.
"""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL  = '/home/ubuntu/models/TinyLlama-1.1B'
OUT    = os.environ['OUT']
FAST   = os.environ.get('FAST', '0') == '1'
PROMPT = "The history of computing spans several distinct eras, each defined by"

torch.manual_seed(0)
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
model.train(False)                                    # inference mode
ids   = tok(PROMPT, return_tensors='pt').input_ids.cuda()

if FAST:
    with torch.no_grad():
        _ = model(ids)
    torch.cuda.synchronize()
    print("FAST_TENANT_DONE", flush=True)
    sys.exit(0)

with torch.no_grad():
    logits = model(ids).logits.float().cpu()          # [1, T, V]

gen = ids.clone()
with torch.no_grad():
    for _ in range(24):
        nxt = model(gen).logits[:, -1, :].argmax(-1, keepdim=True)
        gen = torch.cat([gen, nxt], dim=1)

torch.save({'logits': logits, 'gen_ids': gen[0].tolist()}, OUT)
print("TENANT_DONE gen_text:", repr(tok.decode(gen[0])), flush=True)
