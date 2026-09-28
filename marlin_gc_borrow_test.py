#!/usr/bin/env python3
# Marlin GC-on-free — dispatch-borrow balance test (the advisor's leak self-check).
# Generates on Mistral (many bf16 Marlin dispatches), then asserts the dispatch
# borrow counter returned to 0 — i.e. every lookup() borrow was released by the
# actuator. Run with CIPHER_MARLIN_GC_FAULT_BF16=1 to force ~1/3 of dispatch_bf16
# calls onto the early-error return: proves the ERROR-branch release (no leak on
# the malloc-fail / cast-missing paths the borrow leak would otherwise hit).
import ctypes, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SO = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
lib = ctypes.CDLL(SO)
lib.cipher_rt_marlin_engine_dispatch_borrow.restype = ctypes.c_int
lib.cipher_rt_marlin_calls_bf16_substituted.restype = ctypes.c_ulong
M = "/home/ubuntu/models/Mistral-7B-v0.1"
fault = os.environ.get("CIPHER_MARLIN_GC_FAULT_BF16", "0") not in ("0", "", "n", "N")
t = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = t("The history of computing spans several distinct", return_tensors="pt").input_ids.cuda()
def gen():
    with torch.no_grad():
        return m.generate(ids, max_new_tokens=24, do_sample=False)[0, ids.shape[1]:].tolist()
toks = None
for _ in range(8):           # many forwards ⇒ thousands of bf16 dispatches (+ faults)
    toks = gen()
torch.cuda.synchronize()
borrow = int(lib.cipher_rt_marlin_engine_dispatch_borrow())
sub = int(lib.cipher_rt_marlin_calls_bf16_substituted())
coherent = (toks is not None and len(set(toks)) > 1)   # not all-identical garbage
res = {"so": SO, "fault_injection": fault, "dispatch_borrow_after": borrow,
       "bf16_substituted": sub, "coherent_output": coherent, "sample_toks": toks[:8] if toks else None,
       "PASS": bool(borrow == 0 and coherent)}
json.dump(res, open(os.environ.get("BORROW_OUT", "/home/ubuntu/marlin_gc_borrow.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
print("BORROW_TEST", "PASS" if res["PASS"] else "FAIL", "fault=%s borrow_after=%d" % (fault, borrow), flush=True)
