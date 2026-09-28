# D.7 step-4 core: prove the Marlin (model_id, w_ptr) re-key ISOLATES families
# and mis-route is BLOCKED — directly via the exported engine ABI (no inference).
import ctypes, sys
LIB = sys.argv[1]
m = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
m.cipher_rt_marlin_engine_bind_model.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
m.cipher_rt_marlin_engine_observe_weight.argtypes = [ctypes.c_void_p]
m.cipher_rt_marlin_engine_observe_weight.restype = ctypes.c_int
W = ctypes.c_void_p(0x7f0000abc000)   # a fixed fake weight pointer
A = 0xA11AA11AA11AA11A                 # family-A model_id
B = 0xB22BB22BB22BB22B                 # family-B model_id

fails = []
# 1) Bind W -> family A; observe 3x. Slot (A,W) count should reach 3.
m.cipher_rt_marlin_engine_bind_model(W, A)
ca = [m.cipher_rt_marlin_engine_observe_weight(W) for _ in range(3)]
print("family A observe counts:", ca)
if ca != [1,2,3]: fails.append("A counts not 1,2,3 (got %s)"%ca)

# 2) Rebind SAME W -> family B; observe 2x. If the re-key ISOLATES, slot (B,W) is
#    FRESH -> counts 1,2 (NOT continuing A's 4,5). That proves no cross-family bleed.
m.cipher_rt_marlin_engine_bind_model(W, B)
cb = [m.cipher_rt_marlin_engine_observe_weight(W) for _ in range(2)]
print("family B observe counts (same w_ptr):", cb)
if cb != [1,2]: fails.append("MIS-ROUTE BLEED: B saw A's cache (got %s, expected fresh 1,2)"%cb)

# 3) Re-bind back to A; A's slot must be INTACT at 3 -> next observe = 4 (isolation both ways).
m.cipher_rt_marlin_engine_bind_model(W, A)
ca2 = m.cipher_rt_marlin_engine_observe_weight(W)
print("family A observe after B interleave:", ca2)
if ca2 != 4: fails.append("A slot not preserved across B (got %s, expected 4)"%ca2)

# 4) model_id=0 (unbound/MODEL_UNKNOWN) is a DISTINCT slot from A/B (single-model default).
m.cipher_rt_marlin_engine_bind_model(W, 0)
c0 = m.cipher_rt_marlin_engine_observe_weight(W)
print("model_id=0 (default) observe:", c0)
if c0 != 1: fails.append("default slot not isolated (got %s, expected fresh 1)"%c0)

print("D7_KEYING_PASS" if not fails else "D7_KEYING_FAIL: " + "; ".join(fails))
sys.exit(1 if fails else 0)
