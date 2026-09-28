#!/usr/bin/env python3
# Panel-fix Probe B2 (M5/M11, profiler-free): bottom-up attribution of the FP8-arm slowdown.
# torch.profiler under the substrate returns 0 events (CUPTI client conflict, see profile_fp8.err),
# so instead: per-call wall timing of every Mistral-7B linear shape, fwd-only and fwd+bwd, in both
# arms; counter deltas label each call handled/skipped. Reconstruct step delta = sum(count x delta).
#   - steady HANDLED call delta  = act-quant kernels + FP8-Lt GEMM vs fp16 nvjet  (no prequant)
#   - thrash SKIPPED call delta  = per-step re-prequant churn (2x cudaMalloc + quant + sync + free)
# Engagement ONLY from caller env; fp32 ground truth not needed here (Probe A covered correctness).
import os, sys, json, ctypes, time, torch

out = sys.argv[1]
def ctr():
    try:
        lib = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
        r = {}
        for s in ("handled", "skipped"):
            f = getattr(lib, f"cipher_rt_fp8_calls_{s}"); f.restype = ctypes.c_ulong; r[s] = int(f())
        return r
    except OSError:
        return {"handled": -1, "skipped": -1}

torch.manual_seed(0)
M, ITERS = 4096, 30
# (name, OUT, IN, per_step_count) — Mistral-7B: 32 layers x {q,k,v,o,gate,up,down} + lm_head
SHAPES = [("q_o", 4096, 4096, 64), ("k_v", 1024, 4096, 64), ("gate_up", 14336, 4096, 64),
          ("down", 4096, 14336, 32), ("lm_head", 32000, 4096, 1)]
res = {"arm": "fp8" if os.environ.get("CIPHER_FP8") else "vanilla", "tokens": M, "iters": ITERS, "shapes": {}}
for name, OUT, IN, cnt in SHAPES:
    W  = torch.randn(OUT, IN, dtype=torch.float16, device="cuda") * 0.02
    x0 = torch.randn(M, IN, dtype=torch.float16, device="cuda")
    dy = torch.randn(M, OUT, dtype=torch.float16, device="cuda") * 0.1
    # fwd-only timing (per-call): in fp8 arm, non-square weights thrash fwd<->bwd dims only when
    # bwd also runs; time fwd-only AFTER a bwd to reproduce the training-step alternation.
    x = x0.clone().requires_grad_(True)
    for _ in range(3):
        y = torch.nn.functional.linear(x, W); y.backward(dy); x.grad = None
    torch.cuda.synchronize()
    c0 = ctr(); t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            y = torch.nn.functional.linear(x0, W)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / ITERS * 1e3; c1 = ctr()
    # fwd+bwd timing (per-call pair, reproduces the training fwd/bwd dim alternation)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x = x0.clone().requires_grad_(True)
        y = torch.nn.functional.linear(x, W)
        y.backward(dy)
    torch.cuda.synchronize()
    fb_ms = (time.perf_counter() - t0) / ITERS * 1e3; c2 = ctr()
    res["shapes"][name] = {
        "out": OUT, "in": IN, "count_per_step_fwd_or_bwd": cnt,
        "fwd_ms": round(fwd_ms, 3), "fwdbwd_ms": round(fb_ms, 3),
        "fwd_handled_per_iter": (c1["handled"] - c0["handled"]) / ITERS if c0["handled"] >= 0 else None,
        "fwd_skipped_per_iter": (c1["skipped"] - c0["skipped"]) / ITERS if c0["handled"] >= 0 else None,
        "fb_handled_per_iter": (c2["handled"] - c1["handled"]) / ITERS if c0["handled"] >= 0 else None,
        "fb_skipped_per_iter": (c2["skipped"] - c1["skipped"]) / ITERS if c0["handled"] >= 0 else None}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
