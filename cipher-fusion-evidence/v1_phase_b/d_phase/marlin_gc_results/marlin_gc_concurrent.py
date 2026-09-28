#!/usr/bin/env python3
# Marlin GC-on-free — CONCURRENT safe-reclaim probe (the retired-list UAF test).
# Preload ONE model in the main thread (avoids the HF concurrent-from_pretrained
# meta-tensor race), then race:
#   thread A: continuously generate on M  → live Marlin GEMM launches on M's kits
#   thread B: continuously evict_weight(M's ptrs) + reclaim_retired()
# This races kit frees against in-flight launches on the SAME kits — exactly the
# use-after-free the retire-only + device-synced reclaim design must prevent.
# PASS = no crash / no illegal-memory-access AND A never emits garbage
# (all-identical / corrupt) output. (Strict KL=0 is NOT asserted: mid-flight
# eviction legitimately mixes Marlin/vanilla layers — both correct-but-different.)
import ctypes, torch, hashlib, json, os, sys, threading, time
from transformers import AutoModelForCausalLM, AutoTokenizer

SO = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
DUR = float(os.environ.get("CONC_SEC", "25"))
lib = ctypes.CDLL(SO)
lib.cipher_rt_marlin_engine_bind_model.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
lib.cipher_rt_marlin_engine_evict_weight.argtypes = [ctypes.c_void_p]
lib.cipher_rt_marlin_engine_reclaim_retired.restype = ctypes.c_int
lib.cipher_rt_marlin_engine_gc_fp_failures.restype = ctypes.c_ulong
M = "/home/ubuntu/models/Mistral-7B-v0.1"      # big kits ⇒ meaningful free/launch race
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8], 'little') | 1
PROMPT = "The history of computing spans several distinct"; GEN = 16

t = AutoTokenizer.from_pretrained(M)
if t.pad_token is None: t.pad_token = t.eos_token
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
wptrs = [p.data_ptr() for _, p in m.named_parameters() if p.dim() == 2]
for wp in wptrs: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(wp), mid("mistral"))
ids = t(PROMPT, return_tensors="pt").input_ids.cuda()
def gen():
    with torch.no_grad():
        return m.generate(ids, max_new_tokens=GEN, do_sample=False)[0, ids.shape[1]:].tolist()
# warm up Marlin engagement
for _ in range(3): gen()

state = {"A_iters": 0, "B_iters": 0, "garbage": 0, "errors": [], "stop": False}
lk = threading.Lock()
def worker_gen():
    try:
        while not state["stop"]:
            toks = gen()
            if len(set(toks)) <= 1:        # all-identical ⇒ garbage/corruption
                with lk: state["garbage"] += 1
            with lk: state["A_iters"] += 1
    except Exception as e:
        with lk: state["errors"].append("A: %r" % e); state["stop"] = True
def worker_evict():
    try:
        while not state["stop"]:
            for wp in wptrs:
                lib.cipher_rt_marlin_engine_evict_weight(ctypes.c_void_p(wp))
            torch.cuda.synchronize()
            lib.cipher_rt_marlin_engine_reclaim_retired()
            with lk: state["B_iters"] += 1
    except Exception as e:
        with lk: state["errors"].append("B: %r" % e); state["stop"] = True

ta = threading.Thread(target=worker_gen); tb = threading.Thread(target=worker_evict)
ta.start(); tb.start()
time.sleep(DUR)
state["stop"] = True
ta.join(timeout=60); tb.join(timeout=60)
fpf = int(lib.cipher_rt_marlin_engine_gc_fp_failures())
res = {"so": SO, "dur_s": DUR, "A_gen_iters": state["A_iters"], "B_evict_reclaim_iters": state["B_iters"],
       "garbage_outputs": state["garbage"], "errors": state["errors"], "fp_failures": fpf,
       "PASS": bool(not state["errors"] and state["garbage"] == 0 and state["A_iters"] > 0 and state["B_iters"] > 0)}
json.dump(res, open(os.environ.get("CONC_OUT", "/home/ubuntu/marlin_gc_concurrent.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
print("CONCURRENT", "PASS" if res["PASS"] else "FAIL", flush=True)
sys.exit(0 if res["PASS"] else 1)
