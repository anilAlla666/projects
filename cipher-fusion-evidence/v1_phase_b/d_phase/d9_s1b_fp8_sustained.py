#!/usr/bin/env python3
# D.9 §1b measurement 1 — FP8 SUSTAINED (not burst). 60s clock-sampled loop per dtype.
# Kills the burst-1.85x assumption: FP8 at full tilt also draws max power -> throttles at 700W.
# Reports sustained fp16 & fp8 TFLOPS + sustained RATIO + clock 0/30/60s. CIPHER off.
import os, sys, json, time, subprocess, statistics
PEAK_BF16=989.0e12; PEAK_FP8=1979.0e12
def smi():
    try:
        o=subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,power.draw","--format=csv,noheader,nounits"],
                         capture_output=True,text=True,timeout=5).stdout.strip().splitlines()[0]
        s,p=o.split(","); return float(s),float(p)
    except Exception: return -1.0,-1.0
import torch
SECS=float(os.environ.get("SUSTAIN_SECS","45"))
def sustained(kind, M,N,K):
    a16=torch.randn(M,K,dtype=torch.float16,device="cuda"); b16=torch.randn(N,K,dtype=torch.float16,device="cuda")
    sa=torch.tensor(1.0,device="cuda"); sb=torch.tensor(1.0,device="cuda")
    if kind=="fp8":
        a=a16.to(torch.float8_e4m3fn); b=b16.to(torch.float8_e4m3fn)
        run=lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.float16)
    else:
        run=lambda: a16 @ b16.t()
    for _ in range(10): run()
    torch.cuda.synchronize()
    f=2.0*M*N*K; samples=[]; t_end=time.time()+SECS; last=0; ws=time.time(); wi=0
    while time.time()<t_end:
        for _ in range(30): run()
        torch.cuda.synchronize(); wi+=30; now=time.time()
        if now-last>=1.0:
            dt=now-ws; tf=(wi*f/dt)/1e12; cl,pw=smi()
            samples.append({"t":round(now-(t_end-SECS),1),"tflops":round(tf,1),"clk":cl,"pw":pw})
            last=now; ws=now; wi=0
    tail=[s["tflops"] for s in samples if s["t"]>=SECS*0.5]
    sus=statistics.median(tail) if tail else (samples[-1]["tflops"] if samples else 0)
    clk=statistics.median([s["clk"] for s in samples if s["t"]>=SECS*0.5] or [-1])
    def at(tt):
        return (min(samples,key=lambda s:abs(s["t"]-tt)) if samples else {})
    return {"sustained_tflops":round(sus,1),"sustained_clk":clk,"clk_0":at(1).get("clk"),"clk_mid":at(SECS/2).get("clk"),"clk_end":at(SECS-1).get("clk"),"pw_mid":at(SECS/2).get("pw")}

shapes=[(8192,14336,4096,"ffn_up_M8192"),(8192,4096,4096,"attn_proj_M8192"),(16384,14336,4096,"ffn_up_M16384")]
out={"shapes":[]}
for (M,N,K,tag) in shapes:
    f16=sustained("fp16",M,N,K); f8=sustained("fp8",M,N,K)
    ratio=f8["sustained_tflops"]/max(f16["sustained_tflops"],1e-9)
    row={"tag":tag,"M":M,"N":N,"K":K,
         "fp16_sus_TFLOPS":f16["sustained_tflops"],"fp16_sus_MFU":round(100*f16["sustained_tflops"]*1e12/PEAK_BF16,1),"fp16_clk":f16["sustained_clk"],
         "fp8_sus_TFLOPS":f8["sustained_tflops"],"fp8_sus_MFU_vs989":round(100*f8["sustained_tflops"]*1e12/PEAK_BF16,1),"fp8_clk":f8["sustained_clk"],
         "sustained_ratio":round(ratio,3),"fp16_clk_0_mid_end":[f16["clk_0"],f16["clk_mid"],f16["clk_end"]],"fp8_clk_0_mid_end":[f8["clk_0"],f8["clk_mid"],f8["clk_end"]],"fp8_pw_mid":f8["pw_mid"]}
    out["shapes"].append(row)
    print(f"[{tag}] fp16 SUS {f16['sustained_tflops']}TF@{f16['sustained_clk']}MHz | fp8 SUS {f8['sustained_tflops']}TF@{f8['sustained_clk']}MHz ({row['fp8_sus_MFU_vs989']}% vs989) | SUSTAINED ratio {round(ratio,2)}x", flush=True)
rs=[r["sustained_ratio"] for r in out["shapes"]]
out["median_sustained_ratio"]=round(statistics.median(rs),3)
print(f"MEDIAN SUSTAINED FP8/fp16 ratio = {out['median_sustained_ratio']}x  (burst was 1.85x)", flush=True)
json.dump(out, open("/home/ubuntu/d9_s1b_fp8_sustained.json","w"), indent=2)
print("WROTE d9_s1b_fp8_sustained.json", flush=True)
