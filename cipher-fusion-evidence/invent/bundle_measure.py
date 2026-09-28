#!/usr/bin/env python3
# STAGED PER-MODEL BUNDLE MEASUREMENT (Llama-3.1-8B-Instruct, the model with public co-designed artifacts).
# Stages: dense bf16 -> FP8 -> FP8+EAGLE-3 spec. Measures decode tok/s + tok/W + PPL at batch 1 and saturated.
# This tests the per-model thesis: does co-design (FP8 + a TRAINED EAGLE-3 drafter) deliver MFU/MBU/TPW that
# the model-agnostic post-hoc path could not. Substrate NOT loaded (anchor_loaded probe).
import os, sys, time, json, ctypes, subprocess, threading, math
STAGE = os.environ["B_STAGE"]            # dense | fp8 | fp8_eagle
LOAD = os.environ.get("B_LOAD", "decode")  # decode (batch1) | sat (NREQ concurrent)
NREQ = int(os.environ.get("B_NREQ", "64"))
OUTJ = os.environ["B_OUTJSON"]
BASE = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
PARAMS = 8.03e9; PEAK = 989.5e12; PEAK_BW = 3.35e12
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch
from vllm import LLM, SamplingParams

pw, runflag = [], threading.Event()
def sample():
    while True:
        if runflag.is_set():
            try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
            except Exception: pass
        time.sleep(0.1)

kw = dict(model=BASE, enforce_eager=False, gpu_memory_utilization=0.82, max_model_len=2048, disable_log_stats=True)
if STAGE in ("fp8", "fp8_eagle"): kw["quantization"] = "fp8"
if STAGE == "fp8_eagle":
    kw["speculative_config"] = {"method": "eagle3", "model": EAGLE, "num_speculative_tokens": 5}
    kw["compilation_config"] = {"cudagraph_capture_sizes": [s for s in [1,2,4,8,16,24,32,48,64] ]}
llm = LLM(**kw)

if LOAD == "decode":
    prompts = ["Continue this story in detail: The old lighthouse keeper noticed something strange in the water."]
    B = 1
else:
    import random; rng = random.Random(7)
    base = "Write a detailed technical explanation of how distributed systems handle consensus. "
    prompts = [base * rng.choice([1,2,4]) for _ in range(NREQ)]; B = NREQ
sp = SamplingParams(max_tokens=200, min_tokens=200, ignore_eos=True, temperature=0.0)
llm.generate(prompts, sp, use_tqdm=False)  # warmup

th = threading.Thread(target=sample, daemon=True); th.start()
best = None; best_pw = None
for _ in range(3):
    pw.clear(); runflag.set(); t0 = time.perf_counter()
    o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0
    runflag.clear()
    gen = sum(len(x.outputs[0].token_ids) for x in o); tps = gen / dt
    mp = sum(pw)/len(pw) if pw else None
    if best is None or tps > best: best, best_pw = tps, mp

# PPL on a fixed held-out text (teacher-forced)
ppl = None
try:
    txts = ["The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells, generating ATP used as chemical energy. In 1969 the Apollo 11 mission landed the first humans on the Moon."]
    spp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    oo = llm.generate(txts, spp, use_tqdm=False)
    nlls=[];
    for x in oo:
        for d in (x.prompt_logprobs or [])[1:]:
            if d: nlls.append(-list(d.values())[0].logprob)
    ppl = round(math.exp(sum(nlls)/len(nlls)),4) if nlls else None
except Exception as e:
    ppl = "err:"+str(e)[:40]

# decode MFU/MBU (batch-1 weight-stream): bytes = params * (1 fp8 / 2 bf16)
wb = PARAMS * (1.0 if STAGE != "dense" else 2.0)
step_time = B / best
mbu = (wb + 0) / step_time / PEAK_BW if LOAD == "decode" else None
mfu = 2*PARAMS*best/PEAK
tokw = best/best_pw if best_pw else None
al = True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError: al = False
res = dict(stage=STAGE, load=LOAD, nreq=B, decode_tok_s=round(best,1), mean_power_w=round(best_pw,1) if best_pw else None,
           tok_per_watt=round(tokw,4) if tokw else None, mfu=round(mfu,4), mbu_batch1=round(mbu,3) if mbu else None,
           ppl=ppl, anchor_loaded=al)
print("BUNDLE", json.dumps(res)); json.dump(res, open(OUTJ,"w"), indent=1)
