#!/usr/bin/env python3
# A1 baseline MFU/MBU/TPW bench — CLEAN vLLM, NO CIPHER substrate, NO LD_PRELOAD shim, eager.
# One workload per process (isolation discipline). Modes:
#   decode  RV_BATCH RV_OUT      — latency/throughput decode regimes
#   prefill RV_BATCH RV_PLEN RV_REPS — compute-bound prefill regime (distinct random-token prompts,
#                                      prefix caching DISABLED so reps cannot be cache-skipped)
# Energy: NVML power.draw sampled @50ms, trapezoid-integrated over the measured generate window
# (validated method, rc_ab/ab_driver.py); nvmlDeviceGetTotalEnergyConsumption delta recorded as cross-check.
# Clock + throttle reasons sampled alongside power; achieved clock reported per run.
import sys, os, time, json, threading, random
MODE   = sys.argv[1] if len(sys.argv) > 1 else "decode"
MODEL  = os.environ.get("RV_MODEL", "mistralai/Mistral-7B-v0.1")
B      = int(os.environ.get("RV_BATCH", "1"))
OUT    = int(os.environ.get("RV_OUT", "256"))
PLEN   = int(os.environ.get("RV_PLEN", "2048"))
REPS   = int(os.environ.get("RV_REPS", "3"))
RESULT = os.environ.get("RV_RESULT", f"/home/ubuntu/cipher-fusion-evidence/mfu_lane2/iso_{MODE}_B{B}.json")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_PLUGINS"] = ""  # CRITICAL substrate-line guard: cipher_vllm_kv/cipher_vllm_kvdedup auto-load via vllm.general_plugins entry points (easy-install.pth -> /home/ubuntu/cipher_vllm_plugin) and pull the cipher_v2 substrate into the engine; empty = load no plugins

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")  # deep_gemm not installed; fp8-only warmup path crashes engine init; prior validated runs (rc_pergpu/rc_monitor.py:159) used the same flag

# ---- fixed analytic constants (PREREG.md) ----
NPARAMS    = 7_241_732_096          # Mistral-7B-v0.1 exact (h4096 L32 kv8 inter14336 vocab32000)
PEAK_FLOPS = 989.5e12               # H100 SXM BF16/FP16 dense, no sparsity (NVIDIA datasheet)
PEAK_BW    = 3.35e12                # H100 SXM HBM3 (NVIDIA datasheet)
DTYPE_B    = 2                      # fp16

import pynvml as NV
NV.nvmlInit(); H = NV.nvmlDeviceGetHandleByIndex(0)
_samples = []; _stop = threading.Event()
def _throttle():
    for fn in ("nvmlDeviceGetCurrentClocksThrottleReasons", "nvmlDeviceGetCurrentClocksEventReasons"):
        try: return int(getattr(NV, fn)(H))
        except Exception: pass
    return -1
def _sampler():
    while not _stop.is_set():
        try:
            _samples.append((time.monotonic(), NV.nvmlDeviceGetPowerUsage(H)/1000.0,
                             NV.nvmlDeviceGetClockInfo(H, NV.NVML_CLOCK_SM), _throttle()))
        except Exception: pass
        _stop.wait(0.05)
def integrate(t0, t1):
    pts = [(t, w) for (t, w, _, _) in _samples if t0 <= t <= t1]
    if len(pts) < 2: return None, None, 0
    J = sum((pts[i+1][0]-pts[i][0])*0.5*(pts[i][1]+pts[i+1][1]) for i in range(len(pts)-1))
    return J, sum(w for _, w in pts)/len(pts), len(pts)
def clock_stats(t0, t1):
    cs = sorted(c for (t, _, c, _) in _samples if t0 <= t <= t1)
    rs = 0
    for (t, _, _, r) in _samples:
        if t0 <= t <= t1 and isinstance(r, int) and r > 0: rs |= r
    if not cs: return None
    try: plim = NV.nvmlDeviceGetEnforcedPowerLimit(H)/1000.0
    except Exception: plim = None
    return {"sm_clock_median": cs[len(cs)//2], "sm_clock_min": cs[0], "sm_clock_max": cs[-1],
            "throttle_reasons_or": hex(rs), "sw_power_cap": bool(rs & 0x4), "hw_slowdown": bool(rs & 0x8),
            "enforced_power_limit_w": plim}
def energy_counter():
    try: return NV.nvmlDeviceGetTotalEnergyConsumption(H)  # mJ since driver load
    except Exception: return None

from vllm import LLM, SamplingParams, TokensPrompt

def _relock_clocks():
    # The -lgc 1980 lock is observed to release during vLLM engine init on this box (driver 580.105.08);
    # re-apply right before the measured window so the achieved clock is the locked one. Fail-soft.
    import subprocess
    return  # MFU-audit run: clocks DEFAULT (verify-unlocked discipline); relock disabled

rng = random.Random(20260610)
def rand_prompt(n):  # exact-length prompt, ids well inside vocab, no specials
    return TokensPrompt(prompt_token_ids=[rng.randrange(1000, 28000) for _ in range(n)])

maxlen = (PLEN + 64) if MODE == "prefill" else 2048
llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.85, max_model_len=maxlen,
          dtype="float16", disable_log_stats=True, enable_prefix_caching=False)

res = {"mode": MODE, "model": MODEL, "pid": os.getpid(), "eager": True, "dtype": "float16",
       "prefix_caching": False, "n_params": NPARAMS, "peak_flops": PEAK_FLOPS, "peak_bw": PEAK_BW,
       "flop_model": "2*N_params per token (linear+lm_head matmuls; attention FLOPs excluded => MFU lower bound)",
       "byte_model": "weights N_params*2B streamed once per step; KV excluded => MBU lower bound",
       "tpw_model": "tokens / trapezoid-integrated NVML power.draw @50ms over measured window"}

th = threading.Thread(target=_sampler, daemon=True); th.start()
if MODE == "decode":
    PROMPT_LEN = 32
    prompts = [rand_prompt(PROMPT_LEN) for _ in range(B)]
    warm = SamplingParams(max_tokens=32, min_tokens=32, ignore_eos=True, temperature=0.0)
    llm.generate(prompts, warm, use_tqdm=False)
    sp = SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
    _relock_clocks(); time.sleep(0.5)
    e0 = energy_counter(); t0 = time.monotonic()
    o = llm.generate(prompts, sp, use_tqdm=False)
    t1 = time.monotonic(); e1 = energy_counter()
    ntok = sum(len(x.outputs[0].token_ids) for x in o)
    dt = t1 - t0
    J, avg_w, nsmp = integrate(t0, t1)
    tps = ntok/dt
    res.update({"batch": B, "out_tokens_each": OUT, "prompt_len_each": PROMPT_LEN,
                "n_tokens_generated": ntok, "window_s": dt, "tok_s": tps,
                "joules_trapezoid": J, "avg_power_w": avg_w, "n_power_samples": nsmp,
                "joules_counter": (e1-e0)/1000.0 if (e0 is not None and e1 is not None) else None,
                "mfu": tps*2*NPARAMS/PEAK_FLOPS,
                "mbu_weight_stream_lb": (tps/B)*NPARAMS*DTYPE_B/PEAK_BW,
                "tok_per_joule": (ntok/J if J else None), "clocks": clock_stats(t0, t1)})
elif MODE == "prefill":
    # warmup: one full prefill batch (distinct tokens from the measured ones)
    warm_p = [rand_prompt(PLEN) for _ in range(B)]
    sp1 = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate(warm_p, sp1, use_tqdm=False)
    batches = [[rand_prompt(PLEN) for _ in range(B)] for _ in range(REPS)]  # all-distinct random prompts
    _relock_clocks(); time.sleep(0.5)
    e0 = energy_counter(); t0 = time.monotonic()
    outs = [llm.generate(bp, sp1, use_tqdm=False) for bp in batches]
    t1 = time.monotonic(); e1 = energy_counter()
    dt = t1 - t0
    pref_tok = sum(len(x.prompt_token_ids) for o in outs for x in o)
    dec_tok = sum(len(x.outputs[0].token_ids) for o in outs for x in o)
    J, avg_w, nsmp = integrate(t0, t1)
    res.update({"batch": B, "prompt_len_each": PLEN, "reps": REPS,
                "n_prefill_tokens": pref_tok, "n_decode_tokens_in_window": dec_tok,
                "window_s": dt, "prefill_tok_s": pref_tok/dt,
                "window_caveat": f"window includes {REPS} single-token decode steps (~{REPS}x30ms) => prefill MFU slightly understated",
                "joules_trapezoid": J, "avg_power_w": avg_w, "n_power_samples": nsmp,
                "joules_counter": (e1-e0)/1000.0 if (e0 is not None and e1 is not None) else None,
                "mfu": (pref_tok*2*NPARAMS/dt)/PEAK_FLOPS,
                "prefill_tok_per_joule": (pref_tok/J if J else None), "clocks": clock_stats(t0, t1)})
else:
    raise SystemExit(f"unknown mode {MODE}")
_stop.set(); th.join(timeout=2)
with open(RESULT, "w") as f: json.dump(res, f, indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("flop_model", "byte_model", "tpw_model")}))
NV.nvmlShutdown()
