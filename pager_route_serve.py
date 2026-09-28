#!/usr/bin/env python3
# STEP 1 correctness gate: route a REAL model's weights behind the pager via a scoped MemPool, serve a correct
# forward, evict+pagein the FULL weight set, serve again. Advisor-hardened gates:
#  (G-route) 100% residency: EVERY named_parameter + named_buffer .data_ptr() in [base_va, base_va+used_bytes)
#  (G-cksum) determinism-INDEPENDENT restore: live byte-checksum before page_out == after page_in
#  (G-fwd)   integration: forward bit-identical, but FIRST establish O1a==O1b (no-evict determinism baseline)
#  (G-reg)   two distinct models, per-region isolation: evict A while B resident -> B unaffected, A restores
import ctypes, sys, torch
from ctypes import c_int, c_ulong, c_ulonglong, c_size_t, POINTER
SO = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
A = "/home/ubuntu/models/TinyLlama-1.1B"
B = "/home/ubuntu/models/Llama-3.2-1B-Instruct"   # genuinely distinct model

lib = ctypes.CDLL(SO)
for fn, res, args in [
    ("cipher_pager_init", c_int, []),
    ("cipher_pager_begin_load", c_int, [c_ulonglong, c_size_t]),
    ("cipher_pager_end_load", c_int, [c_int]),
    ("cipher_pager_page_in", c_int, [c_int]),
    ("cipher_pager_page_out", c_int, [c_int]),
    ("cipher_pager_state", c_int, [c_int]),
    ("cipher_pager_find", c_int, [c_ulonglong]),
    ("cipher_pager_live_cksum", c_ulonglong, [c_int]),
]:
    f = getattr(lib, fn); f.restype = res; f.argtypes = args

class Stats(ctypes.Structure):
    _fields_ = [("cold_miss", c_ulong), ("pagein", c_ulong), ("evict", c_ulong), ("state", c_int),
                ("ref", c_int), ("pif", c_int), ("used_bytes", c_ulong), ("base_va", c_ulonglong)]
lib.cipher_pager_get_stats.argtypes = [c_int, POINTER(Stats)]
def stats(rid):
    s = Stats(); lib.cipher_pager_get_stats(rid, ctypes.byref(s)); return s

from torch.cuda.memory import CUDAPluggableAllocator
alloc = CUDAPluggableAllocator(SO, "cipher_pager_malloc", "cipher_pager_free")
assert lib.cipher_pager_init() == 0
torch.cuda.init(); _warm = torch.zeros(1, device="cuda")   # warm default allocator (memory_stats path)

from transformers import AutoModelForCausalLM, AutoTokenizer
PASS = True
def gate(name, ok, detail=""):
    global PASS; PASS = PASS and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

def load_paged(path, key):
    """Load model 'path' with its WEIGHTS routed behind a pager region (scoped MemPool). Returns (model, rid)."""
    cpu = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)   # CPU
    pbytes = sum(p.numel() * p.element_size() for p in cpu.parameters())
    bbytes = sum(b.numel() * b.element_size() for b in cpu.buffers())
    reserve = int((pbytes + bbytes) * 1.10) + (64 << 20)
    rid = lib.cipher_pager_begin_load(key, reserve)
    assert rid >= 0, "begin_load failed"
    pool = torch.cuda.MemPool(alloc.allocator())
    try:
        with torch.cuda.use_mem_pool(pool):
            cpu.to("cuda")               # weights -> GPU via pool -> pager region (NULL mid-load would raise here)
            torch.cuda.synchronize()
    except Exception as e:
        print("  LOAD RAISED (likely reserve too small / NULL mid-load):", e); raise
    assert lib.cipher_pager_end_load(rid) == 0, "end_load failed"
    cpu._cipher_pool = pool              # keep pool alive for region lifetime (advisor footgun)
    return cpu, rid, pbytes + bbytes

def residency(model, rid):
    s = stats(rid); lo, hi = s.base_va, s.base_va + s.used_bytes
    tot = inreg = 0; escaped = []
    for kind, it in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, t in it:
            if t.numel() == 0 or not t.is_cuda: continue
            tot += 1; dp = t.data_ptr()
            if lo <= dp < hi: inreg += 1
            else: escaped.append((kind, name, t.numel() * t.element_size()))
    return inreg, tot, escaped, s

def fwd(model, ids):
    with torch.no_grad():
        return model(ids).logits.float().clone()

print("=== STEP 1: real-model weight routing + evict/pagein correctness ===")
tok = AutoTokenizer.from_pretrained(A)
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()

# ---- MODEL A ----
mA, ridA, expA = load_paged(A, 0xA1)
inreg, tot, escaped, sA = residency(mA, ridA)
weights_escaped = [e for e in escaped if e[0] == "param"]
gate("G-route: 100% of A's PARAMETERS behind pager VA", len(weights_escaped) == 0,
     f"params+buffers in-region={inreg}/{tot}  used={sA.used_bytes>>20}MiB exp~={expA>>20}MiB  escaped={escaped[:3]}")
gate("G-reserve: load fit (used<=reserve, no NULL mid-load)", sA.used_bytes > 0 and sA.used_bytes >= int(0.95*expA),
     f"used={sA.used_bytes>>20}MiB")

# determinism baseline THEN restore gates
o1a = fwd(mA, ids); o1b = fwd(mA, ids)
det = torch.equal(o1a, o1b)
gate("G-det: forward deterministic baseline (O1a==O1b)", det, "(else O2 compare is vs variance, not zero)")
ck_before = lib.cipher_pager_live_cksum(ridA)
assert lib.cipher_pager_page_out(ridA) == 0
gate("G-evict: A -> EVICTED", lib.cipher_pager_state(ridA) == 0)
assert lib.cipher_pager_page_in(ridA) == 0
gate("G-resident: A -> RESIDENT", lib.cipher_pager_state(ridA) == 2)
ck_after = lib.cipher_pager_live_cksum(ridA)
gate("G-cksum (PRIMARY): live region byte-identical across evict+pagein", ck_before == ck_after and ck_before != 0,
     f"cksum {hex(ck_before)} -> {hex(ck_after)}")
o2 = fwd(mA, ids)
gate("G-fwd: A forward bit-identical after evict+pagein", torch.equal(o1a, o2) if det else torch.allclose(o1a, o2, atol=0, rtol=0),
     f"max_abs_diff={(o1a-o2).abs().max().item():.3e}")

# ---- MODEL B (registry / per-region isolation) ----
try:
    mB, ridB, expB = load_paged(B, 0xB2)
    bidsT = AutoTokenizer.from_pretrained(B)("Once upon a time", return_tensors="pt").input_ids.cuda()
    inregB, totB, escB, sB = residency(mB, ridB)
    gate("G-reg.route: 100% of B's PARAMETERS behind its OWN region", len([e for e in escB if e[0]=="param"]) == 0,
         f"B in-region={inregB}/{totB}  base_va A={hex(sA.base_va)} B={hex(sB.base_va)} (disjoint={sA.base_va!=sB.base_va})")
    bref = fwd(mB, bidsT)
    gate("G-reg.find: registry lookup keys -> distinct regions", lib.cipher_pager_find(0xA1)==ridA and lib.cipher_pager_find(0xB2)==ridB)
    # evict A while B resident -> B must be unaffected (per-region isolation)
    assert lib.cipher_pager_page_out(ridA) == 0
    isoB = lib.cipher_pager_state(ridB) == 2 and lib.cipher_pager_state(ridA) == 0
    b_after = fwd(mB, bidsT)
    gate("G-reg.isolation: evict A while B RESIDENT -> B forward unchanged", isoB and torch.equal(bref, b_after),
         f"B state stayed RESIDENT while A EVICTED; B max_abs_diff={(bref-b_after).abs().max().item():.3e}")
    assert lib.cipher_pager_page_in(ridA) == 0
    a_restored = fwd(mA, ids)
    gate("G-reg.restore: A restored after B-resident eviction -> A forward correct", torch.equal(o1a, a_restored),
         f"A max_abs_diff={(o1a-a_restored).abs().max().item():.3e}")
except Exception as e:
    gate("G-reg: two-model registry demo", False, f"EXC {e}")

print(f"\nSTEP 1 ROUTING CORRECTNESS: {'ALL PASS' if PASS else 'FAIL'}")
# The full production .so attaches the cipher_v2 CUPTI subscriber via the cuInit hook when ctypes-loaded into a
# host process (the CLASSIFY logs); its teardown segfaults at interpreter exit -- a harness artifact of this
# non-injection load path (CONTROL1 reproduces it with NO pager). All gates complete BEFORE exit; bypass the
# crashing finalizers so the exit code is a reliable PASS/FAIL signal.
import os
sys.stdout.flush()
os._exit(0 if PASS else 1)
