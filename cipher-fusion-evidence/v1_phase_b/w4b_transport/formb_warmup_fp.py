"""W.4b.2 Form B' — one-time off-hot-path executor fingerprint warmup (§2 #2a).

Launched INJECTED (CUDA_INJECTION64_PATH=libcipher_rt.so) exactly once before the
vanilla executor's decode loop. Loads the deployed model, runs >=64 GEMMs so the
substrate observes the LM-head + FFN shapes, then reads the real W.6 sub-B
fingerprint and writes it to FP_OUT. The executor reads that value as its
deployment-match reference and then runs its hot path VANILLA (no shim tax).

This keeps the substrate fingerprint definition unified (memo §2 #2(a)): the
executor's deployment-match is a direct compare against a genuine
cipher_workload_model_fingerprint(), not a separate hf_config hash (option b).
"""
import os
import sys
import ctypes

MODEL = os.environ["WL_MODEL"]
FP_OUT = os.environ["FP_OUT"]

import torch  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
ids = tok("warm up the model fingerprint accumulator with enough tokens",
          return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(8):                 # enough forwards for the fp to stabilize
        m(ids)                         # (max_m settles once the LM head is seen)
torch.cuda.synchronize()

# the injected lib is loaded RTLD_LOCAL (symbols not global) — load by path to
# read the SAME in-process g_signals state the injection populated.
RT = os.environ.get("CIPHER_RT_LIB", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
libc = ctypes.CDLL(RT, mode=os.RTLD_LOCAL | 0x0001)   # RTLD_LAZY
libc.cipher_workload_model_fingerprint.restype = ctypes.c_uint64
fp = int(libc.cipher_workload_model_fingerprint())
with open(FP_OUT, "w") as f:
    f.write("%d\n" % fp)
print("WARMUP fp=0x%016x (%d) model=%s" % (fp, fp, MODEL), file=sys.stderr, flush=True)
