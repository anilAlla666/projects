#!/usr/bin/env python3
# MBU experiment — decode tok/s + achieved HBM BW (MBU) for fp16 vs 4-bit GPTQ Mistral-7B,
# across batch, cudagraph ON. Standalone vLLM measurement; substrate NOT loaded (anchor_loaded probe).
import os, sys, time, json, ctypes, subprocess, threading
ARM = os.environ.get("MBU_ARM", "fp16")     # fp16 | gptq
BATCH = int(os.environ.get("MBU_BATCH", "1"))
OUTJ = os.environ.get("MBU_OUTJSON", f"/home/ubuntu/cipher-fusion-evidence/mbu/mbu_{ARM}_b{BATCH}.json")
OUT = int(os.environ.get("MBU_OUT", "128"))
PROMPT_TOK = int(os.environ.get("MBU_PROMPT", "16"))   # short ctx -> weights dominate
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODELS = {"fp16": "mistralai/Mistral-7B-v0.1", "gptq": "TheBloke/Mistral-7B-v0.1-GPTQ", "awq": "TheBloke/Mistral-7B-v0.1-AWQ", "fp8": "mistralai/Mistral-7B-v0.1"}
MODEL = MODELS[ARM]
# weight bytes per forward (decode): 7.24B params. fp16=2B/param; GPTQ 4-bit ~= 0.5B/param + scales/zeros (~0.56 eff)
PARAMS = 7.24e9
WEIGHT_BYTES = {"fp16": PARAMS * 2.0, "gptq": PARAMS * 0.56, "awq": PARAMS * 0.56, "fp8": PARAMS * 1.0}[ARM]
PEAK_BW = 3.35e12   # H100 80GB HBM3 peak ~3.35 TB/s

import torch
from vllm import LLM, SamplingParams

# background dmon sampler for mem% (BW utilization proxy)
dmon_samples = []
_stop = threading.Event()
def sample_dmon():
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if _stop.is_set(): break
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try: dmon_samples.append(int(parts[2]))   # mem% column
            except ValueError: pass
    p.terminate()

kw = dict(model=MODEL, enforce_eager=False, gpu_memory_utilization=0.85, max_model_len=1024,
          disable_log_stats=True, compilation_config={"cudagraph_capture_sizes": [BATCH]})
if ARM in ("fp16", "awq"):
    kw["dtype"] = "float16"   # awq kernel requires fp16
FORCE_QUANT = os.environ.get("MBU_QUANT", "")
if ARM == "fp8":
    kw["quantization"] = "fp8"
if FORCE_QUANT:
    kw["quantization"] = FORCE_QUANT
llm = LLM(**kw)
# detect actual quant kernel engaged
quant_method = None
try:
    quant_method = llm.llm_engine.vllm_config.model_config.quantization
except Exception:
    pass

prompt = "computing " * PROMPT_TOK
prompts = [prompt] * BATCH
sp = SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)   # warmup + capture

t = threading.Thread(target=sample_dmon, daemon=True); t.start()
time.sleep(1.0)
best = None
for _ in range(3):
    t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    gen = sum(len(x.outputs[0].token_ids) for x in o); tps = gen / dt
    best = tps if best is None or tps > best else best
_stop.set(); t.join(timeout=2)

# roofline: per decode STEP the model streams weights once + reads KV for all active seqs.
# decode steps for the timed run = OUT (each step advances all BATCH seqs by 1 token).
# tok/s(best) -> step/s = tps / BATCH ; step_time = BATCH/tps
step_time = BATCH / best
# bytes/step: weights (once) + KV read (per seq, ~ctx*2*layers*kv_dim*2B). short ctx -> small. approx weights-dominated.
kv_dim = 1024; layers = 32; ctx = PROMPT_TOK + OUT // 2  # avg ctx during decode
kv_bytes = BATCH * ctx * layers * kv_dim * 2 * 2   # K+V, 2 bytes
bytes_per_step = WEIGHT_BYTES + kv_bytes
achieved_bw = bytes_per_step / step_time
mbu = achieved_bw / PEAK_BW

dmon_mem = sorted(dmon_samples)
dmon_med = dmon_mem[len(dmon_mem)//2] if dmon_mem else None
clocks = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
anchor_loaded = True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError: anchor_loaded = False

res = dict(arm=ARM, model=MODEL, quant_method=quant_method, batch=BATCH, out_tokens=OUT, prompt_tok=PROMPT_TOK,
           decode_tok_s=round(best, 1), step_time_ms=round(step_time*1000, 3),
           weight_bytes_gb=round(WEIGHT_BYTES/1e9, 2), kv_bytes_mb=round(kv_bytes/1e6, 1),
           bytes_per_step_gb=round(bytes_per_step/1e9, 2),
           achieved_bw_tbs=round(achieved_bw/1e12, 3), mbu_roofline=round(mbu, 3),
           dmon_mem_pct_median=dmon_med, dmon_n=len(dmon_samples), peak_bw_tbs=PEAK_BW/1e12,
           clocks=clocks, anchor_loaded=anchor_loaded)
print("MBU", json.dumps(res))
with open(OUTJ, "w") as f: json.dump(res, f, indent=1)
