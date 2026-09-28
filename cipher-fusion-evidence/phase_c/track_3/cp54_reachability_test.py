#!/usr/bin/env python3
"""Track 3 SC3-2 — ctypes reachability GATE (design memo §3, Item-4 PUSH).

Question: can a Python tenant, with libcipher_rt loaded as the
CUDA_INJECTION64_PATH library, resolve libcipher_rt's exported C symbols to
the ALREADY-INJECTED instance (so a call mutates the live g_green_* state)?

Run via:
  CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
  CIPHER_QOS_CLASS=partition CIPHER_SM_COUNT=16 \
  python3 cp54_reachability_test.py

Proof method (functional, stronger than an address compare): after a real GPU
op the injected instance has BUILT its green context, so its module-static
cipher_rt_green_ctx_is_initialized() == 1 and sm_count() > 0. A handle that
reports is_initialized()==1 is provably the injected instance — a separately
loaded copy would report 0. PASS = at least one resolution technique reaches it.
"""
import ctypes, os, sys, json

RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"

import torch
# Force cuInit (InitializeInjection runs) + an explicit cuStreamCreate (the
# hook that triggers cipher_rt_green_ctx_ensure -> green context build).
torch.cuda.init()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    x = torch.ones(4096, device='cuda')
    x = (x * 2.0).relu()
torch.cuda.synchronize()

def probe(label, dll):
    try:
        f_init = dll.cipher_rt_green_ctx_is_initialized
        f_sm   = dll.cipher_rt_green_ctx_sm_count
        f_grp  = dll.cipher_rt_green_ctx_group_id
        f_init.restype = ctypes.c_int
        f_sm.restype   = ctypes.c_uint
        f_grp.restype  = ctypes.c_uint
        addr = ctypes.cast(f_init, ctypes.c_void_p).value
        init, sm, grp = f_init(), f_sm(), f_grp()
        print(f"  {label:28s} addr={hex(addr) if addr else None}"
              f" is_initialized={init} sm_count={sm} group_id={grp}")
        return {"addr": addr, "is_initialized": init, "sm_count": sm,
                "group_id": grp}
    except Exception as e:
        print(f"  {label:28s} FAILED — {type(e).__name__}: {e}")
        return None

print("reachability probes (after a real GPU op — green ctx should be built):")
res = {}
res["CDLL_None"] = probe("CDLL(None)", ctypes.CDLL(None))
try:
    h = ctypes.CDLL(RT, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    res["CDLL_RTLD_NOLOAD"] = probe("CDLL(path,RTLD_NOLOAD)", h)
except Exception as e:
    print(f"  {'CDLL(path,RTLD_NOLOAD)':28s} FAILED — {e}")
    res["CDLL_RTLD_NOLOAD"] = None
res["CDLL_path"] = probe("CDLL(path)", ctypes.CDLL(RT))

reached = [k for k, v in res.items() if v and v["is_initialized"] == 1]
ok = len(reached) > 0
print()
print("reached the injected instance (is_initialized==1) via:",
      reached or "NONE")
print("REACHABILITY:", "PASS" if ok else "FAIL")
out = {"probes": res, "reached_via": reached, "reachability_pass": ok}
json.dump(out, open("/home/ubuntu/cipher-fusion-evidence/phase_c/track_3/"
                    "sc3_reachability_result.json", "w"), indent=2)
sys.exit(0 if ok else 1)
