#!/usr/bin/env python3
# R.A+R.B goodput driver: real eager Mistral-7B greedy decode (batch=1 default so a single-element GEMM
# corruption corrupts THE workload). Records per-token ids + timing + NVML-integrated energy over the
# MEASURED generate window so the analyzer can compute useful-goodput (matching-prefix) and tok/joule.
import sys, os, time, json, threading
MODEL  = os.environ.get("RV_MODEL", "mistralai/Mistral-7B-v0.1")
B      = int(os.environ.get("RV_BATCH", "1"))
OUTTOK = int(os.environ.get("RV_OUT_TOKENS", "128"))
WARM   = int(os.environ.get("RV_WARM_TOKENS", "0"))
UTIL   = float(os.environ.get("RV_UTIL", "0.85"))
MAXLEN = int(os.environ.get("RV_MAXLEN", "2048"))
RESULT = os.environ.get("RV_RESULT", "/home/ubuntu/cipher-fusion-evidence/snapshot/a2_rb/ab_driver_result.json")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_PLUGINS"] = ""  # CRITICAL substrate-line guard: cipher_vllm_kv/cipher_vllm_kvdedup auto-load via vllm.general_plugins entry points (easy-install.pth -> /home/ubuntu/cipher_vllm_plugin) and pull the cipher_v2 substrate into the engine; empty = load no plugins


import pynvml as N
N.nvmlInit(); H = N.nvmlDeviceGetHandleByIndex(0)
# NVML power sampler: collect (mono, power_W) at ~50ms; integrate (trapezoid) over the measured window.
_samples = []; _stop = threading.Event()
def _sampler():
    while not _stop.is_set():
        try: _samples.append((time.monotonic(), N.nvmlDeviceGetPowerUsage(H) / 1000.0))
        except Exception: pass
        _stop.wait(0.05)
def integrate_joules(t0, t1):
    pts = [(t, w) for (t, w) in _samples if t0 <= t <= t1]
    if len(pts) < 2: return None, None
    J = sum((pts[i+1][0]-pts[i][0]) * 0.5*(pts[i][1]+pts[i+1][1]) for i in range(len(pts)-1))
    avg_w = sum(w for _, w in pts)/len(pts)
    return J, avg_w

from vllm import LLM, SamplingParams

def _relock_clocks():
    # The -lgc 1980 lock is observed to release during vLLM engine init on this box (driver 580.105.08);
    # re-apply right before the measured window so the achieved clock is the locked one. Fail-soft.
    import subprocess
    try: subprocess.run(["sudo","-n","nvidia-smi","-lgc","1980,1980"], capture_output=True, timeout=10)
    except Exception: pass

llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=UTIL, max_model_len=MAXLEN,
          dtype="float16", disable_log_stats=True)
prompts = ["The history of computing began " * 4] * B
sp = SamplingParams(max_tokens=OUTTOK, min_tokens=OUTTOK, ignore_eos=True, temperature=0.0)
if WARM > 0:
    llm.generate(prompts, SamplingParams(max_tokens=WARM, min_tokens=WARM, ignore_eos=True, temperature=0.0), use_tqdm=False)

_relock_clocks(); time.sleep(0.5)
th = threading.Thread(target=_sampler, daemon=True); th.start()
t0w = time.time(); t0 = time.monotonic()
o = llm.generate(prompts, sp, use_tqdm=False)
t1 = time.monotonic(); t1w = time.time()
_stop.set(); th.join(timeout=2)
dt = t1 - t0
J, avg_w = integrate_joules(t0, t1)
seqs = [list(x.outputs[0].token_ids) for x in o]
ntok = sum(len(s) for s in seqs)
res = {"model": MODEL, "pid": os.getpid(), "batch": B, "out": OUTTOK,
       "gen_time_s": dt, "gen_wall_start": t0w, "gen_wall_end": t1w,
       "n_tokens_total": ntok, "joules": J, "avg_power_w": avg_w,
       "tok_s": ntok/dt, "tok_per_joule": (ntok/J if J else None),
       "n_power_samples": len(_samples), "token_ids": seqs}
with open(RESULT, "w") as f: json.dump(res, f)
print("DONE", json.dumps({"pid": os.getpid(), "batch": B, "out": OUTTOK, "gen_time_s": round(dt, 3),
      "tok_s": round(ntok/dt, 1), "joules": round(J, 1) if J else None,
      "avg_power_w": round(avg_w, 1) if avg_w else None, "first_seq_len": len(seqs[0])}))
N.nvmlShutdown()
