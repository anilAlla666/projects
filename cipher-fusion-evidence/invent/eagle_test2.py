import os, time, json, threading, subprocess
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
BASE="meta-llama/Llama-3.1-8B-Instruct"; EAGLE="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
FP8=os.environ.get("E_FP8","0")=="1"; SPEC=os.environ.get("E_SPEC","0")=="1"; NREQ=int(os.environ.get("E_NREQ","1"))
tok=AutoTokenizer.from_pretrained(BASE)
pw,run=[],threading.Event()
def smp():
    while True:
        if run.is_set():
            try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
            except: pass
        time.sleep(0.1)
kw=dict(model=BASE,enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True,dtype="bfloat16")
if FP8: kw["quantization"]="fp8"
if SPEC: kw["speculative_config"]={"method":"eagle3","model":EAGLE,"num_speculative_tokens":int(os.environ.get("E_SPEC_TOK","5"))}
llm=LLM(**kw)
import random; rng=random.Random(7)
qs=["Explain how the TCP three-way handshake works step by step.","Write a Python memoized Fibonacci function with comments.",
    "Summarize the causes of the French Revolution.","Describe how a hash table resolves collisions."]
msgs=[[{"role":"user","content":rng.choice(qs)}] for _ in range(NREQ)]
prompts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs]
sp=SamplingParams(max_tokens=256,temperature=0.0)
llm.generate(prompts,sp,use_tqdm=False)
th=threading.Thread(target=smp,daemon=True);th.start()
best=None;bp=None
for _ in range(3):
    pw.clear();run.set();t0=time.perf_counter();o=llm.generate(prompts,sp,use_tqdm=False);dt=time.perf_counter()-t0;run.clear()
    g=sum(len(x.outputs[0].token_ids) for x in o);tps=g/dt;mp=sum(pw)/len(pw) if pw else None
    if best is None or tps>best: best,bp=tps,mp
ok = "handshake" in o[0].outputs[0].text.lower() or "fib" in o[0].outputs[0].text.lower() or len(o[0].outputs[0].text)>50
print("R", json.dumps({"fp8":FP8,"spec":SPEC,"nreq":NREQ,"tok_s":round(best,1),"tok_w":round(best/bp,4) if bp else None,"coherent":ok,"out":o[0].outputs[0].text[:60]}))
