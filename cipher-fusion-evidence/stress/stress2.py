#!/usr/bin/env python3
# STRESS v2 — fixes panel confounds: (1) multi-trial (median + spread), (2) longer/steadier window,
# (3) ISO-cudagraph for spec vs base (both enforce_eager when EAGER=1, isolating spec compute),
# (4) iso-power configs available. Substrate NOT loaded.
import os, sys, time, json, ctypes, subprocess, threading, random, statistics
ARM=os.environ.get("ST_ARM","fp16"); SPEC=os.environ.get("ST_SPEC","0")=="1"
PL=int(os.environ.get("ST_PL","700")); NREQ=int(os.environ.get("ST_NREQ","192"))
K=int(os.environ.get("ST_K","7")); EAGER=os.environ.get("ST_EAGER","0")=="1"
TRIALS=int(os.environ.get("ST_TRIALS","3")); OUTJ=os.environ["ST_OUTJSON"]; SEED=20260611
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from vllm import LLM, SamplingParams
pw, runflag = [], threading.Event()
def sample():
    while True:
        if runflag.is_set():
            try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
            except Exception: pass
        time.sleep(0.1)
kw=dict(model="mistralai/Mistral-7B-v0.1",enforce_eager=EAGER,gpu_memory_utilization=0.85,max_model_len=4096,disable_log_stats=True,enable_prefix_caching=False)
if ARM=="fp16": kw["dtype"]="float16"
if ARM=="fp8": kw["quantization"]="fp8"
if ARM=="gptq": kw["model"]="TheBloke/Mistral-7B-v0.1-GPTQ"
if SPEC: kw["speculative_config"]={"method":"ngram","num_speculative_tokens":K,"prompt_lookup_max":4,"prompt_lookup_min":2}
# iso-cudagraph note: when EAGER=1 BOTH spec and base run eager (no cudagraph) -> clean spec isolation.
llm=LLM(**kw)
rng=random.Random(SEED)
filler=("The system processes requests using continuous batching and streams model weights from memory at every decode step, while attention reads the key-value cache. ")
chat=["What is the capital of France and why is it historically important?","Explain how a hash table works in two paragraphs.","Give me three tips for writing clean code."]
def make_req():
    t=rng.choice(["chat","instruct","rag","code"])
    if t=="chat": p=rng.choice(chat); out=rng.choice([128,256,384])
    elif t=="instruct": p="Write a detailed explanation of "+rng.choice(["TCP/IP","photosynthesis","gradient descent"])+":"; out=rng.choice([256,384])
    elif t=="rag": ctx=filler*rng.choice([4,8,16]); p=f"Context:\n{ctx}\nUsing only the context, list the facts about memory and decode verbatim:"; out=rng.choice([200,300])
    else: p="Write a Python function to merge two sorted lists, with comments:\n\ndef merge("; out=rng.choice([200,300])
    return p,out
reqs=[make_req() for _ in range(NREQ)]
prompts=[p for p,_ in reqs]; sps=[SamplingParams(max_tokens=o,min_tokens=o,ignore_eos=True,temperature=0.0) for _,o in reqs]
llm.generate(prompts[:8],sps[:8],use_tqdm=False)  # warmup (distinct from timed set's tail; prefix-cache note below)
th=threading.Thread(target=sample,daemon=True); th.start()
trials=[]
for _ in range(TRIALS):
    pw.clear(); runflag.set(); t0=time.perf_counter()
    outs=llm.generate(prompts,sps,use_tqdm=False); dt=time.perf_counter()-t0; runflag.clear()
    gen=sum(len(o.outputs[0].token_ids) for o in outs)
    mp=sum(pw)/len(pw) if pw else None
    trials.append(dict(tok_s=round(gen/dt,1),wall=round(dt,2),power=round(mp,1) if mp else None,tokw=round((gen/dt)/mp,4) if mp else None,gen=gen,pw_n=len(pw)))
med_tokw=statistics.median([t["tokw"] for t in trials if t["tokw"]])
med_tps=statistics.median([t["tok_s"] for t in trials])
med_pw=statistics.median([t["power"] for t in trials if t["power"]])
clocks=subprocess.run(["nvidia-smi","--query-gpu=power.limit,clocks.sm","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
al=True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",mode=os.RTLD_NOLOAD)
except OSError: al=False
res=dict(arm=ARM,spec=SPEC,eager=EAGER,power_limit_w=PL,nreq=NREQ,trials=trials,
         median_tok_s=med_tps,median_power_w=med_pw,median_tok_per_watt=med_tokw,clocks=clocks,anchor_loaded=al)
print("STRESS2",json.dumps(res)); json.dump(res,open(OUTJ,"w"),indent=1)
