#!/usr/bin/env python3
# STEP 0 (G-O3): convert the roofline refutation into a MEASURED one. N distinct 4-bit models decode concurrently
# (the bursty-multiplex peak). Measure aggregate tok/s + GPU occupancy (nvidia-smi util%, NOT MFU) + DERIVED MFU
# (tok/s x 2*params / peak_bf16). The finding = the GAP: occupancy can be high while derived MFU stays ~1-2%
# (decode is HBM-bandwidth-bound; distinct models can't batch). + correctness: concurrent output == solo (KL=0).
import os, sys, time, threading, subprocess, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
MODELS=[("/home/ubuntu/models_int4/Mistral-7B-v0.1",7.2e9),("/home/ubuntu/models_int4/Qwen2-7B",7.6e9),("/home/ubuntu/models_int4/Llama-3.1-8B",8.0e9)]
PEAK=990e12  # H100 dense bf16 TFLOP/s (the standard MFU denominator; the dequant matmuls run bf16)
from transformers import AutoModelForCausalLM, AutoTokenizer
def load(p):
    return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, device_map="cuda")
ms=[]; tks=[]; ids=[]
for p,_ in MODELS:
    m=load(p); t=AutoTokenizer.from_pretrained("/home/ubuntu/models/"+os.path.basename(p))
    ms.append(m); tks.append(t); ids.append(t("Write a short story about a robot:", return_tensors="pt").input_ids.cuda())
GEN=48
def gen(i):
    with torch.no_grad():
        out=ms[i].generate(ids[i], max_new_tokens=GEN, do_sample=False, use_cache=True)
    return out

class UtilSampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.samples=[]; s.run_flag=True
    def run(s):
        while s.run_flag:
            try:
                u=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2)
                s.samples.append(int(u.stdout.strip().split("\n")[0]))
            except Exception: pass
            time.sleep(0.1)
def measure(fn, label):
    samp=UtilSampler(); samp.start(); torch.cuda.synchronize(); t0=time.time()
    outs=fn(); torch.cuda.synchronize(); dt=time.time()-t0
    samp.run_flag=False; time.sleep(0.05)
    occ = sum(samp.samples)/len(samp.samples) if samp.samples else -1
    toks=GEN*len(MODELS)
    tps=toks/dt
    avg_params=sum(p for _,p in MODELS)/len(MODELS)
    mfu=tps*2*avg_params/PEAK*100
    print(f"  [{label}] {toks} tok in {dt:.2f}s -> aggregate {tps:.0f} tok/s | occupancy(util%)={occ:.0f}% | DERIVED MFU={mfu:.2f}%",flush=True)
    return outs, tps, occ, mfu

print("=== G-O3 STEP 0: occupancy vs DERIVED MFU for distinct-model bursty multiplex ===",flush=True)
# warmup
for i in range(len(MODELS)): gen(i)
# SERIAL (one model's burst at a time -- back to back, no idle)
serial_outs,_,_,_ = measure(lambda: [gen(i) for i in range(len(MODELS))], "SERIAL back-to-back")
# CONCURRENT (all N bursting at once = the multiplex peak)
def conc():
    res=[None]*len(MODELS); ths=[]
    def w(i): res[i]=gen(i)
    for i in range(len(MODELS)): th=threading.Thread(target=w,args=(i,)); ths.append(th); th.start()
    for th in ths: th.join()
    return res
conc_outs,tps,occ,mfu = measure(conc, "CONCURRENT multiplex")
# correctness: concurrent output bit-identical to serial (multiplex must not perturb)
ok=all(torch.equal(serial_outs[i], conc_outs[i]) for i in range(len(MODELS)))
print(f"  correctness: concurrent output == solo for all {len(MODELS)} models = {ok}",flush=True)
print(f"\n  FINDING: 85% MFU needs ~{0.85*PEAK/(2*7.6e9):,.0f} tok/s; measured aggregate {tps:.0f} tok/s = DERIVED MFU {mfu:.2f}%.",flush=True)
print(f"  occupancy(util%)={occ:.0f}% can be high (GPU busy READING weights) but DERIVED MFU ~1-2% (decode is HBM-bandwidth-bound,",flush=True)
print(f"  distinct models can't batch). 85% MFU is COMPUTE saturation = batched/prefill (vLLM's regime), NOT distinct-model decode.",flush=True)
sys.stdout.flush(); os._exit(0)
