#!/usr/bin/env python3
# Marlin GC-on-free — REAL-MODEL churn end-to-end gate (G1-gating).
# Load/free/reload real bf16 families under churn; compare each reloaded model's
# greedy token stream to a clean solo reference (KL=0 ⇔ exact greedy match, since
# Marlin INT4 is deterministic on weight values). Tracks GPU mem across cycles.
#
# Modes:
#   BOUND   (primary): bind on load, evict_weight on free, reclaim at barrier.
#   UNBOUND (defense): no bind, no evict — relies on the fingerprint (default on).
#   CONCURRENT: 2 threads churn concurrently (evict+reclaim racing live launches)
#               → exercises the retired-list safe-reclaim under concurrency.
# Coverage: same-shape pair (Mistral↔Llama-3.1-8B) + diff-shape (Qwen2, TinyLlama).
# Symbol-guarded: runs on the pre-fix .so too (no evict/reclaim → buggy witness).
import ctypes, torch, gc, hashlib, json, os, sys, threading, time
from transformers import AutoModelForCausalLM, AutoTokenizer

SO = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
CYCLES = int(os.environ.get("CHURN_CYCLES", "12"))
lib = ctypes.CDLL(SO)
lib.cipher_rt_marlin_engine_bind_model.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
lib.cipher_rt_marlin_calls_bf16_substituted.restype = ctypes.c_ulong
HAS_EVICT = hasattr(lib, "cipher_rt_marlin_engine_evict_weight")
HAS_RECLAIM = hasattr(lib, "cipher_rt_marlin_engine_reclaim_retired")
if HAS_EVICT:
    lib.cipher_rt_marlin_engine_evict_weight.argtypes = [ctypes.c_void_p]
if HAS_RECLAIM:
    lib.cipher_rt_marlin_engine_reclaim_retired.restype = ctypes.c_int

FAM = {
    "mistral":   "/home/ubuntu/models/Mistral-7B-v0.1",      # 4096/14336  (same-shape A)
    "llama8b":   "/home/ubuntu/models/Llama-3.1-8B",         # 4096/14336  (same-shape B)
    "qwen2":     "/home/ubuntu/models/Qwen2-7B",             # 3584/18944  (diff-shape; orig symptom)
    "tinyllama": "/home/ubuntu/models/TinyLlama-1.1B",       # 2048/5632   (diff-shape small)
    "llama32_1b":"/home/ubuntu/models/Llama-3.2-1B-Instruct",# 2048/8192   (same hidden 2048 as TL → q/o same-shape; diff ffn)
}
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8], 'little') | 1
PROMPT = "The history of computing spans several distinct"
GEN = 16

def load(p):
    t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if t.pad_token is None: t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    return t, m
def bind(m, i):
    for _, p in m.named_parameters():
        if p.dim() == 2:
            lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(p.data_ptr()), i)
def evict(m):
    if not HAS_EVICT: return
    for _, p in m.named_parameters():
        if p.dim() == 2:
            lib.cipher_rt_marlin_engine_evict_weight(ctypes.c_void_p(p.data_ptr()))
def reclaim():
    if HAS_RECLAIM:
        torch.cuda.synchronize()
        return lib.cipher_rt_marlin_engine_reclaim_retired()
    return None
def gen(t, m):
    ids = t(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=GEN, do_sample=False)
    return o[0, ids.shape[1]:].tolist()
def free_mem_mib():
    f, _ = torch.cuda.mem_get_info()
    return round(f / 2**20, 1)

res = {"so": SO, "has_evict": HAS_EVICT, "has_reclaim": HAS_RECLAIM, "cycles": CYCLES, "parts": {}}

# ---- PART A: solo references (same .so, Marlin engaged, bound) ----
solo = {}
for name, path in FAM.items():
    t, m = load(path); bind(m, mid(name))
    s0 = lib.cipher_rt_marlin_calls_bf16_substituted()
    toks = gen(t, m)
    sub = lib.cipher_rt_marlin_calls_bf16_substituted() - s0
    solo[name] = {"toks": toks, "sub": sub}
    print("SOLO %-9s sub=%d toks=%s" % (name, sub, toks[:6]), flush=True)
    evict(m); del m, t; gc.collect(); torch.cuda.empty_cache(); reclaim()
res["parts"]["A_solo_sub"] = {k: v["sub"] for k, v in solo.items()}

def churn_mode(mode, rotation, ncyc):
    """mode in {BOUND, UNBOUND}. OOM-guarded per cycle (records, does not crash)."""
    recs = []; oom = None
    base_free = free_mem_mib()
    for c in range(ncyc):
        name = rotation[c % len(rotation)]
        try:
            t, m = load(FAM[name])
            if mode == "BOUND":
                bind(m, mid(name))
            s0 = lib.cipher_rt_marlin_calls_bf16_substituted()
            toks = gen(t, m)
            sub = lib.cipher_rt_marlin_calls_bf16_substituted() - s0
            ref = solo[name]["toks"]
            match = sum(a == b for a, b in zip(ref, toks))
            allzero = all(x == toks[0] for x in toks) and len(set(toks)) == 1
            recs.append({"cycle": c, "fam": name, "match": match, "tot": len(ref),
                         "sub": sub, "kl0": match == len(ref), "allzero_corrupt": allzero,
                         "free_mib": free_mem_mib()})
            print("  [%s c%02d %-10s] match=%d/%d sub=%d kl0=%s%s" %
                  (mode, c, name, match, len(ref), sub, match == len(ref),
                   " ALLZERO!" if allzero else ""), flush=True)
            if mode == "BOUND":
                evict(m)
            del m, t; gc.collect(); torch.cuda.empty_cache(); reclaim()
        except Exception as e:
            oom = "cycle %d %s: %r" % (c, name, e)
            print("  [%s c%02d %-10s] EXCEPTION %s" % (mode, c, name, oom), flush=True)
            try: torch.cuda.empty_cache(); reclaim()
            except Exception: pass
            break
    final_free = free_mem_mib()
    return recs, base_free, final_free, oom

# BOUND uses the big same-shape pair (mistral↔llama8b) + diff-shape (qwen2/tinyllama):
# evict bounds the large fp16 surrogate, so 12 big-model cycles fit in 80GB.
ROT_BIG = ["mistral", "llama8b", "qwen2", "mistral", "tinyllama", "llama8b"]
# UNBOUND is correctness-only (fingerprint, NO evict) → the large fp16 surrogate
# leaks by design (documented v-next pure-eviction gap), so use SMALL models:
# tinyllama↔llama32_1b share hidden=2048 ⇒ q/o are 2048×2048 SAME-shape (the
# fingerprint's job) while ffn differs (dim-reval's job).
ROT_SMALL = ["tinyllama", "llama32_1b", "tinyllama", "llama32_1b",
             "tinyllama", "llama32_1b", "tinyllama", "llama32_1b"]

# ---- PART B: BOUND churn (PRIMARY: evict_weight + reclaim) → KL=0 + flat-mem ----
if HAS_EVICT:
    recs, bf, ff, oom = churn_mode("BOUND", ROT_BIG, CYCLES)
    kl0_all = bool(recs) and all(r["kl0"] for r in recs)
    nocorrupt = not any(r["allzero_corrupt"] for r in recs)
    leak = (bf - ff)
    res["parts"]["B_bound"] = {"all_kl0": kl0_all, "no_allzero": nocorrupt,
                               "n_cycles": len(recs), "oom": oom,
                               "base_free_mib": bf, "final_free_mib": ff,
                               "net_leak_mib": round(leak, 1),
                               "flat_mem": abs(leak) < 2048,
                               "PASS": kl0_all and nocorrupt and oom is None and abs(leak) < 2048,
                               "note": "12 big-model cycles completing on 80GB without OOM ⇒ evict+reclaim bounds the fp16 surrogate",
                               "cycles": recs}
else:
    res["parts"]["B_bound"] = {"PASS": None, "note": "evict API absent (pre-fix .so)"}

# ---- PART C: UNBOUND churn (fingerprint correctness ONLY, small models) ----
# KL=0 is the gate; memory is NOT bounded without evict (documented), so the
# fp16-surrogate leak is RECORDED, not failed. Small models avoid OOM.
recs, bf, ff, oom = churn_mode("UNBOUND", ROT_SMALL, min(CYCLES, 8))
kl0_all = bool(recs) and all(r["kl0"] for r in recs)
nocorrupt = not any(r["allzero_corrupt"] for r in recs)
leak = (bf - ff)
res["parts"]["C_unbound"] = {"all_kl0": kl0_all, "no_allzero": nocorrupt,
                             "n_cycles": len(recs), "oom": oom,
                             "base_free_mib": bf, "final_free_mib": ff,
                             "recorded_leak_mib": round(leak, 1),
                             "note": "correctness-only (fingerprint, no evict); leak EXPECTED+documented (v-next evict-on-free wiring bounds it)",
                             "PASS": kl0_all and nocorrupt and oom is None}

# NOTE: the concurrent safe-reclaim gate is validated by the dedicated
# marlin_gc_concurrent.py (preloads one model in the main thread, then races
# evict+reclaim against live launches) — the in-process 2-thread variant here
# was removed because transformers.from_pretrained is not thread-safe for
# CONCURRENT loads (meta-tensor lazy-init race), which masked the actual test.
res["parts"]["D_concurrent"] = {"PASS": None, "note": "validated separately by marlin_gc_concurrent.py"}

# ---- overall ----
gates = []
if res["parts"]["B_bound"]["PASS"] is not None: gates.append(res["parts"]["B_bound"]["PASS"])
gates.append(res["parts"]["C_unbound"]["PASS"])
res["OVERALL_PASS"] = all(gates)
out = os.environ.get("MARLIN_GC_CHURN_OUT", "/home/ubuntu/marlin_gc_churn.json")
json.dump(res, open(out, "w"), indent=2)
print("CHURN", "PASS" if res["OVERALL_PASS"] else "FAIL",
      "| B_bound=%s C_unbound=%s (concurrent: separate test)" %
      (res["parts"]["B_bound"]["PASS"], res["parts"]["C_unbound"]["PASS"]), flush=True)
sys.exit(0 if res["OVERALL_PASS"] else 1)
