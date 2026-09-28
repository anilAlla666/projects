#!/usr/bin/env python3
"""FUTURE_SCOPE/A Phase 3 — vLLM performance-parity probe (THROWAWAY).

One vLLM 0.21.0 instance, greedy decode, instrumented: tok/s, TTFT + mean ITL
(from vLLM RequestOutput metrics when available), GPU power (nvidia-smi, 200 ms),
byte-identical token_ids. Run plain vs CUDA_INJECTION64_PATH=libcipher_rt.so to
measure parity.

Env: EAGER(0/1) · PROMPT_KEY(short|med|long) · TAG · OUT_JSON
Runs in the vLLM venv.
"""
import json
import os
import subprocess
import threading
import time

from vllm import LLM, SamplingParams

EAGER = os.environ.get("EAGER", "0") == "1"
PROMPT_KEY = os.environ.get("PROMPT_KEY", "med")
TAG = os.environ.get("TAG", "run")
OUT = os.environ.get("OUT_JSON", "/tmp/p3_%s.json" % TAG)
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"

_BASE = "The history of computing spans several distinct eras, each defined by"
PROMPTS = {
    "short": "Explain GPU scheduling.",
    "med":   _BASE,
    "long":  (_BASE + " — ") * 12,        # ~200-token context
}


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"], timeout=2)
            samples.append(float(out.decode().split("\n")[0].strip()))
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(0.2)


def main():
    prompt = PROMPTS[PROMPT_KEY]
    llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=0.3,
              dtype="float16", max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=128)

    # warmup (excluded)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))

    samples, stop = [], threading.Event()
    threading.Thread(target=power_sampler, args=(stop, samples),
                     daemon=True).start()
    t0 = time.time()
    out = llm.generate([prompt], sp)
    gen_s = time.time() - t0
    stop.set()
    time.sleep(0.25)

    o = out[0].outputs[0]
    ntok = len(o.token_ids)
    ttft = itl_mean = None
    m = getattr(out[0], "metrics", None)
    if m is not None:
        try:
            if m.first_token_time and m.arrival_time:
                ttft = m.first_token_time - m.arrival_time
            if m.last_token_time and m.first_token_time and ntok > 1:
                itl_mean = (m.last_token_time - m.first_token_time) / (ntok - 1)
        except Exception:                                     # noqa: BLE001
            pass

    res = {"tag": TAG, "eager": EAGER, "prompt_key": PROMPT_KEY,
           "injection": bool(os.environ.get("CUDA_INJECTION64_PATH")),
           "gen_s": round(gen_s, 4), "n_tokens": ntok,
           "tok_s": round(ntok / gen_s, 2) if gen_s else 0,
           "ttft_s": round(ttft, 4) if ttft else None,
           "itl_mean_ms": round(itl_mean * 1e3, 3) if itl_mean else None,
           "power_mean_w": round(sum(samples) / len(samples), 1)
           if samples else None,
           "power_n": len(samples),
           "token_ids": list(o.token_ids)}
    json.dump(res, open(OUT, "w"))
    print("RESULT %s" % json.dumps({k: res[k] for k in
          ("tag", "tok_s", "n_tokens", "ttft_s", "itl_mean_ms",
           "power_mean_w")}), flush=True)


if __name__ == "__main__":
    main()
