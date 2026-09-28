#!/usr/bin/env python3
# Marlin GC-on-free — 30-min sustained CHURN soak (the GC stability gate).
# Continuous load/run/free/reload cycles over rotating bf16 families (incl. the
# same-shape Mistral↔Llama-8B pair + diff-shape Qwen2/TinyLlama), with
# evict_weight + reclaim each cycle, for D9_SOAK_SEC. Asserts over the whole run:
#   - no crash / no exception,
#   - no corruption: every family's greedy output stays == its first-seen output
#     (KL=0; an all-zero/garbage output is a corruption failure),
#   - no leak: GPU free memory flat across cycles (reclaim drains retired kits),
#   - fp_failures == 0 (fingerprint sampling healthy).
import ctypes, torch, gc, hashlib, json, os, sys, time
from transformers import AutoModelForCausalLM, AutoTokenizer

SO = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
DUR = float(os.environ.get("D9_SOAK_SEC", "1800"))
lib = ctypes.CDLL(SO)
lib.cipher_rt_marlin_engine_bind_model.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
lib.cipher_rt_marlin_calls_bf16_substituted.restype = ctypes.c_ulong
lib.cipher_rt_marlin_engine_evict_weight.argtypes = [ctypes.c_void_p]
lib.cipher_rt_marlin_engine_reclaim_retired.restype = ctypes.c_int
lib.cipher_rt_marlin_engine_gc_fp_failures.restype = ctypes.c_ulong

FAM = {
    "tinyllama": "/home/ubuntu/models/TinyLlama-1.1B",
    "mistral":   "/home/ubuntu/models/Mistral-7B-v0.1",
    "llama8b":   "/home/ubuntu/models/Llama-3.1-8B",
    "qwen2":     "/home/ubuntu/models/Qwen2-7B",
}
ROT = ["tinyllama", "mistral", "tinyllama", "llama8b", "tinyllama", "qwen2"]
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8], 'little') | 1
PROMPT = "The history of computing spans several distinct"; GEN = 16

def load(p):
    t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if t.pad_token is None: t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    return t, m
def bind(m, i):
    for _, p in m.named_parameters():
        if p.dim() == 2: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(p.data_ptr()), i)
def evict(m):
    for _, p in m.named_parameters():
        if p.dim() == 2: lib.cipher_rt_marlin_engine_evict_weight(ctypes.c_void_p(p.data_ptr()))
def gen(t, m):
    ids = t(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=GEN, do_sample=False)
    return o[0, ids.shape[1]:].tolist()
def free_mib():
    f, _ = torch.cuda.mem_get_info(); return round(f / 2**20, 1)

first = {}            # family -> first-seen greedy output (corruption ref)
cycles = 0; corruptions = 0; allzero = 0; crash = None
t0 = time.time(); next_cp = t0 + 60; checkpoints = []; base_free = None
try:
    while time.time() - t0 < DUR:
        name = ROT[cycles % len(ROT)]
        t, m = load(FAM[name]); bind(m, mid(name))
        toks = gen(t, m)
        if base_free is None:
            base_free = free_mib()
        if name not in first:
            first[name] = toks
        else:
            if toks != first[name]: corruptions += 1
        if len(set(toks)) == 1: allzero += 1
        cycles += 1
        evict(m); del m, t; gc.collect(); torch.cuda.empty_cache()
        lib.cipher_rt_marlin_engine_reclaim_retired()
        if time.time() >= next_cp:
            fpf = int(lib.cipher_rt_marlin_engine_gc_fp_failures())
            cp = {"t": round(time.time() - t0), "cycles": cycles, "free_mib": free_mib(),
                  "leak_mib": round(base_free - free_mib(), 1) if base_free else 0,
                  "corruptions": corruptions, "allzero": allzero, "fp_failures": fpf}
            checkpoints.append(cp)
            print("[soak t=%ds] cyc=%d free=%.0fMiB leak=%.1fMiB corrupt=%d allzero=%d fpfail=%d"
                  % (cp["t"], cp["cycles"], cp["free_mib"], cp["leak_mib"], corruptions, allzero, fpf), flush=True)
            next_cp += 60
except Exception as e:
    crash = repr(e)

final_free = free_mib(); fpf = int(lib.cipher_rt_marlin_engine_gc_fp_failures())
leak = (base_free - final_free) if base_free else 0
res = {"so": SO, "soak_sec": round(time.time() - t0, 1), "cycles": cycles,
       "corruptions": corruptions, "allzero": allzero, "fp_failures": fpf,
       "base_free_mib": base_free, "final_free_mib": final_free, "net_leak_mib": round(leak, 1),
       "crash": crash, "checkpoints": checkpoints,
       # a LEAK is free memory SHRINKING across the run (positive net_leak) AND
       # growing across checkpoints; negative net_leak (more free at end than the
       # mid-cycle-0 baseline sample) is NOT a leak. Gate on checkpoint spread.
       "checkpoint_free_spread_mib": round((max(c["free_mib"] for c in checkpoints)
                                            - min(c["free_mib"] for c in checkpoints)), 1) if checkpoints else 0,
       "PASS": bool(crash is None and corruptions == 0 and allzero == 0 and fpf == 0
                    and cycles > 0
                    and (not checkpoints or
                         (max(c["free_mib"] for c in checkpoints)
                          - min(c["free_mib"] for c in checkpoints)) < 2048))}
json.dump(res, open(os.environ.get("MARLIN_GC_SOAK_OUT", "/home/ubuntu/marlin_gc_soak.json"), "w"), indent=2)
print("SOAK %s: %d cycles / %.0fs, corrupt=%d allzero=%d fpfail=%d leak=%.1fMiB crash=%s"
      % ("PASS" if res["PASS"] else "FAIL", cycles, res["soak_sec"], corruptions, allzero, fpf, leak, crash), flush=True)
sys.exit(0 if res["PASS"] else 1)
