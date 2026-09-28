#!/usr/bin/env python3
# v2 Probe A (the PREREG correctness gate): per-call math of the SUCCESSOR FP8 actuator.
#   fwd: y = F.linear(x, W) = x @ W.T  — the (T,N) layout the engine hardcodes -> should be FP8-handled,
#        rel-err ~ per-tensor-FP8 noise (0.03-0.05)
#   bwd: dx = dy @ W — transa=N layout -> v2 must LAYOUT-REFUSE it (bwd_handled == 0, layout_refused
#        increments), rel-err ~ fp16 noise (<= 1e-3)
# Engagement comes ONLY from the caller env (CUDA_INJECTION64_PATH/CIPHER_FP8); this script never sets it.
# Lib path for counter reads comes from CIPHER_LIB (the successor .so), falling back to substrate-absent.
import os, sys, json, ctypes, torch

out = sys.argv[1]
LIB = os.environ.get("CIPHER_LIB", "/home/ubuntu/cipher_fp8v2_build/libcipher_rt.so")

def counters():
    try:
        lib = ctypes.CDLL(LIB, mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        vals = {}
        for sym in ("cipher_rt_fp8_calls_handled", "cipher_rt_fp8_calls_layout_refused",
                    "cipher_rt_fp8_calls_total", "cipher_rt_fp8_calls_skipped"):
            f = getattr(lib, sym); f.restype = ctypes.c_ulong
            vals[sym.replace("cipher_rt_fp8_calls_", "")] = int(f())
        return vals
    except OSError:
        return None  # substrate not loaded (vanilla arm)

torch.manual_seed(0)
res = {"arm": "fp8" if os.environ.get("CIPHER_FP8") else "vanilla", "lib": LIB, "tokens": 4096, "shapes": {}}
SHAPES = {"q_4096x4096": (4096, 4096), "kv_1024x4096": (1024, 4096), "gate_14336x4096": (14336, 4096),
          "down_4096x14336": (4096, 14336), "lmhead_32000x4096": (32000, 4096)}
for name, (OUT, IN) in SHAPES.items():
    W  = torch.randn(OUT, IN, dtype=torch.float16, device="cuda") * 0.02
    x0 = torch.randn(4096, IN, dtype=torch.float16, device="cuda")
    dy = torch.randn(4096, OUT, dtype=torch.float16, device="cuda") * 0.1
    ref_y  = x0.float() @ W.float().t()
    ref_dx = dy.float() @ W.float()
    rec = []
    for it in range(4):  # stability=2: iter0 skipped, iter1+ handled
        x = x0.clone().requires_grad_(True)
        c0 = counters()
        y = torch.nn.functional.linear(x, W)
        torch.cuda.synchronize(); c1 = counters()
        y.backward(dy)
        torch.cuda.synchronize(); c2 = counters()
        rel = lambda a, b: float((a.float() - b).norm() / b.norm())
        d = lambda a, b, kk: (b[kk] - a[kk]) if (a is not None and b is not None) else None
        rec.append({"iter": it,
                    "fwd_handled": d(c0, c1, "handled"), "fwd_refused": d(c0, c1, "layout_refused"),
                    "bwd_handled": d(c1, c2, "handled"), "bwd_refused": d(c1, c2, "layout_refused"),
                    "fwd_rel_err_vs_fp32": rel(y.detach(), ref_y),
                    "bwd_rel_err_vs_fp32": rel(x.grad, ref_dx)})
    res["shapes"][name] = rec
res["counters_final"] = counters()

# PREREG gate verdicts, computed in-probe so the JSON is self-judging
gate = {"fwd_handled_fires": True, "bwd_never_handled": True, "fwd_err_band": True, "bwd_fp16_exact": True}
for name, rec in res["shapes"].items():
    for r in rec[1:]:                      # post-stability iters
        if r["fwd_handled"] is not None and r["fwd_handled"] < 1: gate["fwd_handled_fires"] = False
        if r["fwd_handled"] and not (0.001 < r["fwd_rel_err_vs_fp32"] < 0.10): gate["fwd_err_band"] = False
    for r in rec:
        if r["bwd_handled"]: gate["bwd_never_handled"] = False
        if r["bwd_rel_err_vs_fp32"] > 1e-3: gate["bwd_fp16_exact"] = False
res["gate"] = gate
res["gate_pass"] = all(gate.values()) if res["arm"] == "fp8" else None
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "shapes"}, indent=1))
