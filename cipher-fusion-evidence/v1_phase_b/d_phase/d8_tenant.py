#!/usr/bin/env python3
# D.8 FAIRNESS+SHIELD gate — single tenant worker (one process == one agent).
# Loads the D.8 staging libcipher_rt via ctypes (arms the cublasGemmEx GOT hook
# + the timing-only throttle), then runs a bf16 TinyLlama decode workload. The
# throttle is exercised through the ~18k cublasGemmEx calls per decode (D.7).
#
# Role decides the load shape (adversarial, per the D.8 memo §2):
#   noisy     — continuous, no think-time (the aggressor / throughput neighbor)
#   victim    — steady requests with think-time (Gate A: keeps quota share?)
#   highband  — small frequent requests, band>=1 (Gate B: p99 bounded?)
# Mode/env (CIPHER_FAIRNESS/CIPHER_SHIELD/CIPHER_SHIELD_BAND/CIPHER_FAIRNESS_THROTTLE_US)
# is inherited from the parent — set BEFORE this process starts. Default-unset
# == disarmed == pre-D.8 behavior.
import os, sys, json, time, ctypes, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True, choices=["noisy", "victim", "highband"])
ap.add_argument("--lib", required=True)
ap.add_argument("--model", default="/home/ubuntu/models/TinyLlama-1.1B")
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--think-ms", type=float, default=20.0)   # victim/highband inter-request gap
ap.add_argument("--gen-tokens", type=int, default=32)     # tokens per request
ap.add_argument("--saturate", type=int, default=0)        # noisy: if >0, run NxN bf16 matmul flood (FLOP-heavy aggressor) instead of decode
ap.add_argument("--out", required=True)
a = ap.parse_args()

# arm the substrate hook BEFORE torch/cuda init
lib = ctypes.CDLL(a.lib, mode=ctypes.RTLD_GLOBAL)
for fn, rt in [("cipher_rt_fairness_self_calls", ctypes.c_ulonglong),
               ("cipher_rt_fairness_self_yields", ctypes.c_ulonglong),
               ("cipher_rt_fairness_armed", ctypes.c_int),
               ("cipher_rt_fairness_self_band", ctypes.c_int)]:
    try:
        getattr(lib, fn).restype = rt
    except Exception:
        pass

import torch

import torch

# FLOP-heavy saturating aggressor (the realistic single-agent H100 hog): a
# continuous large bf16 matmul flood. This is the only way one agent can starve
# B=1 victims on an H100 (per the campaign's B=1=4-18%-HBM finding). It also
# explicitly participates in the ledger once per matmul batch (so it presents a
# HIGH call-count — the BEST case for should_yield to flag it). self_yields then
# answers: when the throttle DOES fire on a saturating aggressor, do victims
# recover? (Separately, self_gemm_calls via the natural cublasGemmEx hook shows
# whether a real big-GEMM aggressor would even be visible in production.)
if a.role == "noisy" and a.saturate > 0:
    lib.cipher_rt_fairness_record_and_maybe_throttle.restype = None
    N = a.saturate
    x = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    nat0 = int(lib.cipher_rt_fairness_self_calls())  # natural-hook calls before manual participation
    for _ in range(3):
        z = x @ y
    torch.cuda.synchronize()
    nat_hook = int(lib.cipher_rt_fairness_self_calls()) - nat0  # did the big matmul hit cublasGemmEx?
    iters = 0
    deadline = time.time() + a.seconds
    while time.time() < deadline:
        for _ in range(8):
            z = x @ y
            lib.cipher_rt_fairness_record_and_maybe_throttle()  # explicit ledger participation + throttle
        torch.cuda.synchronize()
        iters += 8
    res = {"role": a.role, "armed": int(lib.cipher_rt_fairness_armed()),
           "band": int(lib.cipher_rt_fairness_self_band()),
           "self_gemm_calls": int(lib.cipher_rt_fairness_self_calls()),
           "self_yields": int(lib.cipher_rt_fairness_self_yields()),
           "natural_hook_calls_per_3matmul": nat_hook,
           "matmul_iters": iters, "saturate_dim": N,
           "requests": iters, "tokens": 0, "tokens_per_s": 0.0,
           "lat_mean_ms": 0.0, "lat_p50_ms": 0.0, "lat_p99_ms": 0.0, "n_lat": 0}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[tenant noisy-saturate N={N}] armed={res['armed']} "
          f"gemms={res['self_gemm_calls']} yields={res['self_yields']} "
          f"natural_hook(3 matmul)={nat_hook} matmul_iters={iters}", flush=True)
    sys.exit(0)

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
prompt = "The history of computing began when"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

# warmup (also triggers the first-GEMM fairness init / env read)
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()

lat = []          # per-request wall latency (s)
reqs = 0
toks = 0
deadline = time.time() + a.seconds
think = a.think_ms / 1000.0
with torch.no_grad():
    while time.time() < deadline:
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=a.gen_tokens, do_sample=False)
        torch.cuda.synchronize()
        lat.append(time.time() - t0)
        toks += a.gen_tokens
        reqs += 1
        if a.role in ("victim", "highband") and think > 0:
            time.sleep(think)

wall = a.seconds
lat_sorted = sorted(lat)
def pct(p):
    if not lat_sorted:
        return 0.0
    i = min(len(lat_sorted) - 1, int(p / 100.0 * len(lat_sorted)))
    return lat_sorted[i]

res = {
    "role": a.role,
    "armed": int(lib.cipher_rt_fairness_armed()),
    "band": int(lib.cipher_rt_fairness_self_band()),
    "self_gemm_calls": int(lib.cipher_rt_fairness_self_calls()),
    "self_yields": int(lib.cipher_rt_fairness_self_yields()),
    "requests": reqs,
    "tokens": toks,
    "tokens_per_s": toks / wall,
    "lat_mean_ms": (sum(lat) / len(lat) * 1000.0) if lat else 0.0,
    "lat_p50_ms": pct(50) * 1000.0,
    "lat_p99_ms": pct(99) * 1000.0,
    "n_lat": len(lat),
}
json.dump(res, open(a.out, "w"), indent=2)
print(f"[tenant {a.role}] armed={res['armed']} band={res['band']} "
      f"gemms={res['self_gemm_calls']} yields={res['self_yields']} "
      f"reqs={reqs} tok/s={res['tokens_per_s']:.1f} "
      f"p99={res['lat_p99_ms']:.1f}ms", flush=True)
