#!/usr/bin/env python3
# D.9 FP8 actuator — CIPHER-engaged torch large-batch MFU (Mistral-7B B=64 forward).
# Run under CUDA_INJECTION64_PATH with CIPHER_FP8 on (engaged) or unset (CIPHER-off bf16 baseline).
# MFU vs 989 bf16 ref, sustained (watts+clock sampled). Guard 2 = engaged must BEAT CIPHER-off.
import os, time, json, threading, subprocess, statistics, torch
from transformers import AutoModelForCausalLM
M="/home/ubuntu/models/Mistral-7B-v0.1"
PEAK=989.0e12
B,S=64,512
L,H,HD,PLIN = 32,32,128,7.241e9
def flops(B,S): T=B*S; return 2.0*PLIN*T + 2.0*L*B*H*(S**2)*HD
def smi():
    try:
        o=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader,nounits"],
                         capture_output=True,text=True,timeout=4).stdout.strip().splitlines()[0]
        s,p=o.split(","); return float(s),float(p)
    except Exception: return -1.0,-1.0
tag = "fp8" if os.environ.get("CIPHER_FP8") else "off"
print(f"[{tag}] loading Mistral-7B bf16 ...", flush=True)
ATTN=os.environ.get("D9_ATTN","sdpa")  # sdpa => flash, GEMM-bound forward (realistic train/serve)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation=ATTN).cuda().eval()
ids=torch.randint(1,31000,(B,S),device='cuda')
samp={"on":False,"d":[]}
def sampler():
    while samp["on"]:
        c,p=smi(); samp["d"].append((c,p)); time.sleep(0.2)
with torch.no_grad():
    for i in range(8): model(ids); torch.cuda.synchronize()   # warmup (FP8 engages after stability=2)
    samp["on"]=True; th=threading.Thread(target=sampler); th.start()
    best=1e9; N=30
    for i in range(N):
        t0=time.time(); model(ids); torch.cuda.synchronize(); dt=time.time()-t0; best=min(best,dt)
    # sustained window for power/clock (not a burst): keep going ~6s
    t_end=time.time()+6.0
    while time.time()<t_end: model(ids); torch.cuda.synchronize()
    samp["on"]=False; th.join()
tf=flops(B,S)/best/1e12
cl=[c for c,_ in samp["d"] if c>0]; pw=[p for _,p in samp["d"] if p>0]
res={"tag":tag,"B":B,"S":S,"fwd_ms":round(best*1e3,2),"TFLOPS":round(tf,1),
     "MFU_vs989_pct":round(100*tf*1e12/PEAK,1),
     "sus_clk_mhz":round(statistics.median(cl),0) if cl else -1,
     "med_watts":round(statistics.median(pw),0) if pw else -1,
     "max_watts":round(max(pw),0) if pw else -1}
json.dump(res,open(f"/home/ubuntu/d9_fp8_mfu_{tag}.json","w"),indent=2)
print(f"[{tag}] fwd={res['fwd_ms']}ms TFLOPS={res['TFLOPS']} MFU_vs989={res['MFU_vs989_pct']}% clk={res['sus_clk_mhz']}MHz watts={res['med_watts']}(max {res['max_watts']})",flush=True)
print(f"WROTE d9_fp8_mfu_{tag}.json",flush=True)
