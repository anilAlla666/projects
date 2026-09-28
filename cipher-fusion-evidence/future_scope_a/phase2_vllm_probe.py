#!/usr/bin/env python3
"""FUTURE_SCOPE/A Phase 2 — vLLM single-instance transparency probe (THROWAWAY).

Runs ONE vLLM 0.21.0 instance, greedy decode of a fixed prompt, and records
output + timing. Run it (a) plain and (b) under
CUDA_INJECTION64_PATH=libcipher_rt.so — the token_ids must be byte-identical
(greedy decode is deterministic; the substrate must not perturb output).

Env: EAGER (0/1) · TAG · OUT_JSON
Runs in the vLLM venv: /home/ubuntu/vllm_env/bin/python
"""
import json
import os
import time

from vllm import LLM, SamplingParams

EAGER = os.environ.get("EAGER", "0") == "1"
TAG = os.environ.get("TAG", "run")
OUT = os.environ.get("OUT_JSON", "/tmp/p2_%s.json" % TAG)
MODEL = "/home/ubuntu/models/TinyLlama-1.1B"
PROMPT = ("The history of computing spans several distinct eras, "
          "each defined by")


def main():
    print("PROBE start tag=%s eager=%s pid=%d injection=%s"
          % (TAG, EAGER, os.getpid(),
             os.environ.get("CUDA_INJECTION64_PATH", "<none>")), flush=True)
    t_load = time.time()
    llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=0.3,
              dtype="float16", max_model_len=2048)
    load_s = time.time() - t_load

    sp = SamplingParams(temperature=0.0, max_tokens=128)
    t0 = time.time()
    out = llm.generate([PROMPT], sp)
    gen_s = time.time() - t0

    o = out[0].outputs[0]
    ntok = len(o.token_ids)
    res = {"tag": TAG, "eager": EAGER, "pid": os.getpid(),
           "injection": os.environ.get("CUDA_INJECTION64_PATH", ""),
           "load_s": round(load_s, 2), "gen_s": round(gen_s, 3),
           "n_tokens": ntok, "tok_s": round(ntok / gen_s, 1) if gen_s else 0,
           "text": o.text, "token_ids": list(o.token_ids)}
    json.dump(res, open(OUT, "w"))
    print("RESULT %s" % json.dumps({k: res[k] for k in
          ("tag", "eager", "n_tokens", "tok_s", "load_s")}), flush=True)
    print("TEXT: %r" % o.text[:180], flush=True)
    print("PROBE done tag=%s" % TAG, flush=True)


if __name__ == "__main__":
    main()
