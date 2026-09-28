import os, time, json, threading, subprocess
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams
M=os.environ["S_MODEL"]; NREQ=int(os.environ.get("S_NREQ","32"))
FP8=os.environ.get("S_FP8","0")=="1"; SPEC=os.environ.get("S_SPEC","0")=="1"; OUT=os.environ["S_OUT"]
EAGLE="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
pw,run=[],threading.Event()
def smp():
    while True:
        if run.is_set():
            try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
            except: pass
        time.sleep(0.1)
kw=dict(model=M,enforce_eager=os.environ.get("S_EAGER","0")=="1",gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=True)
if FP8: kw["quantization"]="fp8"
if os.environ.get("S_QUANT"): kw["quantization"]=os.environ["S_QUANT"]
if SPEC: kw["speculative_config"]={"method":"eagle3","model":EAGLE,"num_speculative_tokens":5}
try:
    llm=LLM(**kw)
except Exception as e:
    json.dump({"model":M,"fp8":FP8,"spec":SPEC,"nreq":NREQ,"error":str(e)[:300]},open(OUT,"w")); raise SystemExit
import random; rng=random.Random(7)
base="The quick brown fox jumps over the lazy dog. Distributed systems handle consensus using "
prompts=[base*rng.choice([1,2,4]) for _ in range(NREQ)]
sp=SamplingParams(max_tokens=256,temperature=0.0,ignore_eos=True,min_tokens=256)
llm.generate(prompts,sp,use_tqdm=False)
th=threading.Thread(target=smp,daemon=True);th.start()
best=None;bp=None
for _ in range(3):
    pw.clear();run.set();t0=time.perf_counter();o=llm.generate(prompts,sp,use_tqdm=False);dt=time.perf_counter()-t0;run.clear()
    g=sum(len(x.outputs[0].token_ids) for x in o);tps=g/dt;mp=sum(pw)/len(pw) if pw else None
    if best is None or tps>best: best,bp=tps,mp
res={"model":M,"fp8":FP8,"spec":SPEC,"nreq":NREQ,"plugins_env":os.environ.get("VLLM_PLUGINS","UNSET"),"tok_s":round(best,1),
     "tok_w":round(best/bp,4) if bp else None,"mean_w":round(bp,1) if bp else None,
     "quant":str(getattr(llm.llm_engine.vllm_config.model_config,'quantization',None)),
     "out":o[0].outputs[0].text[:80]}
json.dump(res,open(OUT,"w"));print("S24",json.dumps(res))
