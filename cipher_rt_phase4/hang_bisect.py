"""CP 2.4 — Llama-arm hang bisection (minimal: 1 prompt, 64 tokens).

Scopes the cuDNN-attention x Marlin hang. The arm is set entirely by env:
  CIPHER_SPEC=0                      -> stock generate (no spec at all)
  CIPHER_SPEC=on CIPHER_SPEC_DRAFT=ngram   -> n-gram spec (no 2nd stream)
  CIPHER_SPEC=on CIPHER_SPEC_DRAFT=<path>  -> model-draft spec (2nd stream)
Marlin / substrate set by CIPHER_MARLIN + LD_PRELOAD/CUDA_INJECTION64_PATH.
Prints '[bisect] DONE' on success; a hang is observed as a timeout.
"""
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
sys.path.insert(0, "/home/ubuntu/cipher_rt_phase4")
import torch
import tenant_register

MODEL = os.environ["WL_MODEL"]
tenant = tenant_register.register(os.environ.get("WL_TENANT_ID", "hangbisect"))
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa",
    trust_remote_code=False).cuda()
m = m.train(False)

import cipher_spec_decode
cipher_spec_decode.install()

enc = tok("The Pacific Ocean is the largest ocean on Earth, covering "
          "approximately", return_tensors="pt").to("cuda")
print("[bisect] model loaded; generating 64 tokens "
      "(CIPHER_SPEC=%s CIPHER_SPEC_DRAFT=%s CIPHER_MARLIN=%s)..."
      % (os.environ.get("CIPHER_SPEC", "1"),
         os.environ.get("CIPHER_SPEC_DRAFT", "ngram"),
         os.environ.get("CIPHER_MARLIN", "off")), flush=True)
t0 = time.time()
with torch.no_grad():
    out = m.generate(**enc, max_new_tokens=64, do_sample=False,
                     pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
print("[bisect] DONE — %d tokens in %.2fs"
      % (int(out.shape[-1]) - int(enc.input_ids.shape[-1]), time.time() - t0),
      flush=True)
