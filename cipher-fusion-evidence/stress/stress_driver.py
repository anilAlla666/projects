#!/usr/bin/env python3
# STRESS TEST — do the TPW/MBU/spec claims hold under realistic SATURATED load with a MIXED workload?
# Real vLLM continuous batching, variable prompt+output lengths, many concurrent requests, power sampled
# during the run, p50/p99 latency. Substrate NOT loaded (anchor_loaded probe).
import os, sys, time, json, ctypes, subprocess, threading, random
ARM = os.environ.get("ST_ARM", "fp16")            # fp16 | fp8 | gptq
SPEC = os.environ.get("ST_SPEC", "0") == "1"
PL = int(os.environ.get("ST_PL", "700"))          # power limit W (set by caller)
NREQ = int(os.environ.get("ST_NREQ", "128"))      # concurrent request pool (saturates the GPU)
K = int(os.environ.get("ST_K", "5"))
WL = os.environ.get("ST_WL", "mixed")             # mixed | rag
OUTJ = os.environ["ST_OUTJSON"]
SEED = 20260611
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch
from vllm import LLM, SamplingParams

# power sampler during the timed window
pw, runflag = [], threading.Event()
def sample():
    while True:
        if runflag.is_set():
            try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
            except Exception: pass
        time.sleep(0.1)

kw = dict(model="mistralai/Mistral-7B-v0.1", enforce_eager=False, gpu_memory_utilization=0.85,
          max_model_len=4096, disable_log_stats=True)
if ARM == "fp16": kw["dtype"] = "float16"
if ARM == "fp8": kw["quantization"] = "fp8"
if ARM == "gptq": kw["model"] = "TheBloke/Mistral-7B-v0.1-GPTQ"
if SPEC:
    kw["speculative_config"] = {"method":"ngram","num_speculative_tokens":K,"prompt_lookup_max":4,"prompt_lookup_min":2}
    kw["compilation_config"] = {"cudagraph_capture_sizes":[8,16,24,32]}
llm = LLM(**kw)

# ---- realistic MIXED workload: variable prompt + output lengths, multiple task types ----
rng = random.Random(SEED)
filler = ("The system processes requests using continuous batching and streams model weights from memory "
          "at every decode step, while attention reads the key-value cache. ")
chat = ["What is the capital of France and why is it historically important?",
        "Explain how a hash table works in two paragraphs.",
        "Give me three tips for writing clean code."]
def make_req():
    t = rng.choice(["chat","instruct","rag","code"])
    if t == "chat":   p = rng.choice(chat); out = rng.choice([64,128,200])
    elif t == "instruct": p = "Write a detailed explanation of "+rng.choice(["TCP/IP","photosynthesis","gradient descent"])+":"; out = rng.choice([128,256])
    elif t == "rag":  # long context + extraction (spec-decode's favorable case)
        ctx = filler * rng.choice([4,8,16]); p = f"Context:\n{ctx}\nUsing only the context, list the facts about memory and decode verbatim:"; out = rng.choice([100,200])
    else:             p = "Write a Python function to merge two sorted lists, with comments:\n\ndef merge("; out = rng.choice([100,200])
    return p, out
reqs = [make_req() for _ in range(NREQ)]
prompts = [p for p,_ in reqs]
sps = [SamplingParams(max_tokens=o, min_tokens=o, ignore_eos=True, temperature=0.0) for _,o in reqs]

llm.generate(prompts[:8], sps[:8], use_tqdm=False)  # warmup
th = threading.Thread(target=sample, daemon=True); th.start()
pw.clear(); runflag.set()
t0 = time.perf_counter()
outs = llm.generate(prompts, sps, use_tqdm=False)    # SATURATED: all NREQ submitted, continuous batching
dt = time.perf_counter() - t0
runflag.clear()
gen = sum(len(o.outputs[0].token_ids) for o in outs)
tps = gen/dt
mean_pw = sum(pw)/len(pw) if pw else None
tokw = tps/mean_pw if mean_pw else None
clocks = subprocess.run(["nvidia-smi","--query-gpu=power.limit,clocks.sm","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
al = True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_NOLOAD)
except OSError: al = False
res = dict(arm=ARM, spec=SPEC, power_limit_w=PL, nreq=NREQ, wl=WL,
           total_gen_tokens=gen, wall_s=round(dt,2), agg_tok_s=round(tps,1),
           mean_power_w=round(mean_pw,1) if mean_pw else None, tok_per_watt=round(tokw,4) if tokw else None,
           pw_samples=len(pw), clocks=clocks, anchor_loaded=al)
print("STRESS", json.dumps(res))
json.dump(res, open(OUTJ,"w"), indent=1)
