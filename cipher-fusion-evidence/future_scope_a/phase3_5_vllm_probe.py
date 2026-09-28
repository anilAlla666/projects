#!/usr/bin/env python3
"""FUTURE_SCOPE/A Phase 3.5 — vLLM + CIPHER DVFS actuator probe (THROWAWAY).

One vLLM 0.21.0 instance, greedy decode, graph mode. Records tok/s, GPU power
AND SM clock (nvidia-smi, 200 ms), byte-identical token_ids. Run with
CIPHER_VOLT engaged (via env, set by the sweep driver) to measure DVFS tok/W
lift beneath vLLM.

Env: PROMPT_KEY · TAG · OUT_JSON  (+ CIPHER_VOLT / CIPHER_VOLT_MHZ /
CUDA_INJECTION64_PATH set by the driver). Graph mode always (production path).
"""
import json
import os
import subprocess
import threading
import time

from vllm import LLM, SamplingParams

PROMPT_KEY = os.environ.get("PROMPT_KEY", "med")
TAG = os.environ.get("TAG", "run")
OUT = os.environ.get("OUT_JSON", "/tmp/p35_%s.json" % TAG)
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"

_BASE = "The history of computing spans several distinct eras, each defined by"
PROMPTS = {"short": "Explain GPU scheduling.", "med": _BASE,
           "long": (_BASE + " — ") * 12}


def sampler(stop, pw, clk):
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.sm",
                 "--format=csv,noheader,nounits"], timeout=2)
            p, c = out.decode().split("\n")[0].split(",")
            pw.append(float(p))
            clk.append(float(c))
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(0.2)


def main():
    prompt = PROMPTS[PROMPT_KEY]
    llm = LLM(model=MODEL, enforce_eager=False, gpu_memory_utilization=0.3,
              dtype="float16", max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))

    pw, clk, stop = [], [], threading.Event()
    threading.Thread(target=sampler, args=(stop, pw, clk),
                     daemon=True).start()
    t0 = time.time()
    out = llm.generate([prompt], sp)
    gen_s = time.time() - t0
    stop.set()
    time.sleep(0.25)

    o = out[0].outputs[0]
    ntok = len(o.token_ids)
    res = {"tag": TAG, "prompt_key": PROMPT_KEY,
           "injection": bool(os.environ.get("CUDA_INJECTION64_PATH")),
           "cipher_volt": os.environ.get("CIPHER_VOLT", ""),
           "cipher_volt_mhz": os.environ.get("CIPHER_VOLT_MHZ", ""),
           "gen_s": round(gen_s, 4), "n_tokens": ntok,
           "tok_s": round(ntok / gen_s, 2) if gen_s else 0,
           "power_mean_w": round(sum(pw) / len(pw), 1) if pw else None,
           "sm_clock_mean_mhz": round(sum(clk) / len(clk), 0) if clk else None,
           "sm_clock_min_mhz": min(clk) if clk else None,
           "sm_clock_max_mhz": max(clk) if clk else None,
           "n_samples": len(pw),
           "token_ids": list(o.token_ids)}
    res["tok_w"] = (round(res["tok_s"] / res["power_mean_w"], 4)
                    if res["power_mean_w"] else None)
    json.dump(res, open(OUT, "w"))
    print("RESULT %s" % json.dumps({k: res[k] for k in
          ("tag", "tok_s", "power_mean_w", "sm_clock_mean_mhz", "tok_w")}),
          flush=True)


if __name__ == "__main__":
    main()
