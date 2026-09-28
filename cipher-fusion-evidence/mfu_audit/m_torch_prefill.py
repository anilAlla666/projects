#!/usr/bin/env python3
# Torch-eager HF Mistral-7B PREFILL forward bench — the matched baseline AND engaged lane for the FP8/actuator
# tests (same script; engagement comes only from env: CUDA_INJECTION64_PATH / LD_PRELOAD + CIPHER_* arming vars,
# set by the CALLER — this script never sets them). No vLLM. Clocks DEFAULT (no relock).
# MFU = tokens * 2*N_params / window / 989.5e12 (attention FLOPs excluded => slight lower bound; stated).
import os, sys, time, json, threading
TAG    = os.environ.get("MA_TAG", "vanilla")
B      = int(os.environ.get("MA_BATCH", "8"))
T      = int(os.environ.get("MA_SEQ", "2048"))
REPS   = int(os.environ.get("MA_REPS", "5"))
ATTN   = os.environ.get("MA_ATTN", "sdpa")           # sdpa default; eager available
RESULT = os.environ.get("MA_RESULT", f"/home/ubuntu/cipher-fusion-evidence/mfu_audit/m_torchpf_{TAG}.json")
NPARAMS, PEAK = 7_241_732_096, 989.5e12

import pynvml as NV
NV.nvmlInit(); H = NV.nvmlDeviceGetHandleByIndex(0)
_s = []; _stop = threading.Event()
def _smp():
    while not _stop.is_set():
        try:
            _s.append((time.monotonic(), NV.nvmlDeviceGetPowerUsage(H)/1000.0,
                       NV.nvmlDeviceGetClockInfo(H, NV.NVML_CLOCK_SM)))
        except Exception: pass
        _stop.wait(0.05)

import torch
from transformers import AutoModelForCausalLM
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16,
                                             attn_implementation=ATTN).cuda().eval()
ids = torch.randint(1000, 28000, (B, T), device="cuda")
with torch.no_grad():
    model(input_ids=ids)          # warmup (also lets any injected substrate classify steady-state)
    torch.cuda.synchronize()
    th = threading.Thread(target=_smp, daemon=True); th.start()
    try: e0 = NV.nvmlDeviceGetTotalEnergyConsumption(H)
    except Exception: e0 = None
    t0 = time.monotonic()
    for _ in range(REPS):
        model(input_ids=ids)
    torch.cuda.synchronize()
    t1 = time.monotonic()
    try: e1 = NV.nvmlDeviceGetTotalEnergyConsumption(H)
    except Exception: e1 = None
_stop.set(); th.join(timeout=2)
dt = t1 - t0
tok = B*T*REPS
pts = [(t, w) for (t, w, _) in _s if t0 <= t <= t1]
J = sum((pts[i+1][0]-pts[i][0])*0.5*(pts[i][1]+pts[i+1][1]) for i in range(len(pts)-1)) if len(pts) > 1 else None
cs = sorted(c for (t, _, c) in _s if t0 <= t <= t1)
res = {"tag": TAG, "pid": os.getpid(), "batch": B, "seq": T, "reps": REPS, "attn": ATTN,
       "dtype": "fp16", "engine": "transformers-eager-forward (no vLLM)",
       "flop_model": "2*N_params per prefill token; attention excluded => MFU lower bound",
       "n_params": NPARAMS, "peak_flops": PEAK,
       "tokens": tok, "window_s": dt, "prefill_tok_s": tok/dt,
       "mfu": tok*2*NPARAMS/dt/PEAK,
       "joules_trapezoid": J, "joules_counter": (e1-e0)/1000.0 if (e0 is not None and e1 is not None) else None,
       "tok_per_joule": (tok/J if J else None),
       "avg_power_w": (sum(w for _, w in pts)/len(pts) if pts else None),
       "clocks": ({"sm_median": cs[len(cs)//2], "sm_min": cs[0], "sm_max": cs[-1]} if cs else None),
       "env_engagement_vars": {k: os.environ.get(k) for k in
            ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "CIPHER_DISPATCH_LIVE") if os.environ.get(k)} or None}
# fired-counter proof: read the substrate's exported FP8 counters from the ALREADY-loaded .so (RTLD_NOLOAD
# = never loads it into a vanilla process; returns a handle only if injection already mapped it).
try:
    import ctypes
    lib = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_LAZY | os.RTLD_NOLOAD)
    for sym in ("cipher_rt_fp8_calls_total", "cipher_rt_fp8_calls_handled", "cipher_rt_fp8_calls_skipped"):
        fn = getattr(lib, sym); fn.restype = ctypes.c_ulong
        res[sym] = int(fn())
except OSError:
    res["fp8_counters"] = "substrate not loaded in this process (vanilla lane)"
except Exception as e:
    res["fp8_counter_err"] = repr(e)
with open(RESULT, "w") as f: json.dump(res, f, indent=1)
print("RESULT", json.dumps(res))
