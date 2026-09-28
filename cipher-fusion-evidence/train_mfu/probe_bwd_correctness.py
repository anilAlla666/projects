#!/usr/bin/env python3
# Panel-fix Probe A (M12): is the FP8-substituted GEMM mathematically correct PER CALL?
#   fwd: y = F.linear(x, W) = x @ W.T  — the (T,N) col-major layout the engine hardcodes
#   bwd: dx = dy @ W                   — the input-grad GEMM, DIFFERENT trans/ld layout
# Engagement comes ONLY from the caller env (CUDA_INJECTION64_PATH/CIPHER_FP8); this script
# never sets it. Each arm is compared against an fp32 ground truth of the SAME fp16 values.
# Counter deltas around each op identify exactly which calls were FP8-handled.
import os, sys, json, ctypes, torch

out = sys.argv[1]
def handled():
    try:
        lib = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        f = lib.cipher_rt_fp8_calls_handled; f.restype = ctypes.c_ulong
        return int(f())
    except OSError:
        return -1  # substrate not loaded (vanilla arm)

torch.manual_seed(0)
res = {"arm": "fp8" if os.environ.get("CIPHER_FP8") else "vanilla", "tokens": 4096, "shapes": {}}
SHAPES = {"q_4096x4096": (4096, 4096), "kv_1024x4096": (1024, 4096), "gate_14336x4096": (14336, 4096)}
for name, (OUT, IN) in SHAPES.items():
    W  = torch.randn(OUT, IN, dtype=torch.float16, device="cuda") * 0.02
    x0 = torch.randn(4096, IN, dtype=torch.float16, device="cuda")
    dy = torch.randn(4096, OUT, dtype=torch.float16, device="cuda") * 0.1
    ref_y  = x0.float() @ W.float().t()
    ref_dx = dy.float() @ W.float()
    rec = []
    for it in range(3):
        x = x0.clone().requires_grad_(True)
        h0 = handled()
        y = torch.nn.functional.linear(x, W)
        torch.cuda.synchronize(); h1 = handled()
        y.backward(dy)
        torch.cuda.synchronize(); h2 = handled()
        rel = lambda a, b: float((a.float() - b).norm() / b.norm())
        rec.append({"iter": it,
                    "fwd_handled": (h1 - h0) if h0 >= 0 else None,
                    "bwd_handled": (h2 - h1) if h0 >= 0 else None,
                    "fwd_rel_err_vs_fp32": rel(y.detach(), ref_y),
                    "bwd_rel_err_vs_fp32": rel(x.grad, ref_dx)})
    res["shapes"][name] = rec
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
