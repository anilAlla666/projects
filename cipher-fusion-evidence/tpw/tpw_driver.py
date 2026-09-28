#!/usr/bin/env python3
# TPW measurement — decode tok/s + mean power.draw (during decode) -> tok/W, for fp16 vs 4-bit at a given
# power cap. Standalone vLLM; substrate NOT loaded (anchor_loaded probe). Power cap = DVFS stand-in.
import os, sys, time, json, ctypes, subprocess, threading
ARM = os.environ.get("TPW_ARM", "fp16")          # fp16 | gptq
BATCH = int(os.environ.get("TPW_BATCH", "16"))
PL = int(os.environ.get("TPW_PL", "700"))         # power limit W (informational; set by caller)
OUTJ = os.environ.get("TPW_OUTJSON", f"/home/ubuntu/cipher-fusion-evidence/tpw/tpw_{ARM}_pl{PL}.json")
OUT = int(os.environ.get("TPW_OUT", "256"))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
MODELS = {"fp16": "mistralai/Mistral-7B-v0.1", "gptq": "TheBloke/Mistral-7B-v0.1-GPTQ"}
MODEL = MODELS[ARM]
import torch
from vllm import LLM, SamplingParams

# power sampler (~10 Hz) — collected only during the timed window
power_samples = []
_run = threading.Event()
def sample_power():
    while True:
        if _run.is_set():
            try:
                w = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip()
                power_samples.append(float(w))
            except Exception: pass
        time.sleep(0.1)

kw = dict(model=MODEL, enforce_eager=False, gpu_memory_utilization=0.85, max_model_len=1024,
          disable_log_stats=True, compilation_config={"cudagraph_capture_sizes": [BATCH]})
if ARM == "fp16": kw["dtype"] = "float16"
llm = LLM(**kw)
quant = None
try: quant = llm.llm_engine.vllm_config.model_config.quantization
except Exception: pass

prompts = ["computing " * 16] * BATCH
sp = SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)  # warmup + capture

th = threading.Thread(target=sample_power, daemon=True); th.start()
best = None; best_pw = None
for _ in range(3):
    power_samples.clear(); _run.set()
    t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    _run.clear()
    gen = sum(len(x.outputs[0].token_ids) for x in o); tps = gen / dt
    pw = sum(power_samples)/len(power_samples) if power_samples else None
    if best is None or tps > best:
        best, best_pw = tps, pw

tokw = best / best_pw if best_pw else None
pl_now = subprocess.run(["nvidia-smi","--query-gpu=power.limit,clocks.sm","--format=csv,noheader"],
                        capture_output=True, text=True).stdout.strip()
anchor_loaded = True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError: anchor_loaded = False

res = dict(arm=ARM, quant_method=quant, batch=BATCH, power_limit_w=PL, out_tokens=OUT,
           decode_tok_s=round(best,1), mean_power_w=round(best_pw,1) if best_pw else None,
           tok_per_watt=round(tokw,4) if tokw else None, power_samples_n=len(power_samples),
           pl_clocks_now=pl_now, anchor_loaded=anchor_loaded)
print("TPW", json.dumps(res))
with open(OUTJ, "w") as f: json.dump(res, f, indent=1)
