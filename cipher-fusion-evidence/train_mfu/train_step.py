#!/usr/bin/env python3
# Real LoRA-finetune training step on Mistral-7B (NOT toy B=1): full fwd+bwd+AdamW, ≥40 steps, stable wi1 recipe
# (real coherent text, grad-clip 1.0, fp32 adapter params, eager attention). Records loss curve + step MFU + NVML
# energy + clock. Engagement (FP8/substrate) comes ONLY from the caller's env (CUDA_INJECTION64_PATH/CIPHER_FP8);
# this script never sets it and never monkeypatches the loop.
# FLOP MODEL (stated): LoRA finetune, base weights FROZEN => per token = fwd 2N + input-grad 2N = 4*N_params
# (no base weight-grad GEMM under LoRA; r=16 adapter GEMMs <1%). This is a FINETUNE, not a pretrain proxy.
# Full-pretrain would be 6N (fwd 2N + bwd 4N incl weight-grad); reported separately if a full-finetune point is run.
import os, sys, time, json, threading, argparse, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--pin-mhz", type=int, default=0)        # 0 = default clocks; else iso-clock pin via relock loop (caller)
ap.add_argument("--result", required=True)
ap.add_argument("--loss-out", default=None)
args = ap.parse_args()
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
MODEL="mistralai/Mistral-7B-v0.1"; NPARAMS=7_241_732_096; PEAK=989.5e12
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import pynvml as NV
NV.nvmlInit(); H=NV.nvmlDeviceGetHandleByIndex(0)
_s=[]; _stop=threading.Event()
def _thr():
    for fn in ("nvmlDeviceGetCurrentClocksThrottleReasons","nvmlDeviceGetCurrentClocksEventReasons"):
        try: return int(getattr(NV,fn)(H))
        except Exception: pass
    return -1
def _smp():
    while not _stop.is_set():
        try: _s.append((time.monotonic(), NV.nvmlDeviceGetPowerUsage(H)/1000.0, NV.nvmlDeviceGetClockInfo(H,NV.NVML_CLOCK_SM), _thr()))
        except Exception: pass
        _stop.wait(0.05)

torch.manual_seed(0)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation="sdpa")
lcfg=LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                lora_dropout=0.0,bias="none",task_type="CAUSAL_LM")
model=get_peft_model(model,lcfg); model.cuda(); model.train()
for n,p in model.named_parameters():
    if p.requires_grad: p.data=p.data.float()
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
tok=AutoTokenizer.from_pretrained(MODEL)
passage=("The history of computing began with mechanical calculators and evolved through vacuum tubes, transistors, "
         "integrated circuits, and now massively parallel accelerators used to train and serve large language models "
         "in modern datacenters. ")
ids=tok(passage*80, return_tensors="pt")["input_ids"][0][:args.seq]
FIXED=ids.unsqueeze(0).repeat(args.batch,1).cuda()
trainable=[p for p in model.parameters() if p.requires_grad]
def step():
    out=model(input_ids=FIXED,labels=FIXED); loss=out.loss
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
    return float(loss.detach())
for _ in range(args.warmup): step()
torch.cuda.synchronize()
th=threading.Thread(target=_smp,daemon=True); th.start()
try: e0=NV.nvmlDeviceGetTotalEnergyConsumption(H)
except Exception: e0=None
losses=[]; steptimes=[]; T0=time.monotonic()
for i in range(args.steps):
    torch.cuda.synchronize(); t0=time.perf_counter()
    losses.append(step())
    torch.cuda.synchronize(); steptimes.append(time.perf_counter()-t0)
T1=time.monotonic()
try: e1=NV.nvmlDeviceGetTotalEnergyConsumption(H)
except Exception: e1=None
_stop.set(); th.join(timeout=2)
pts=[(t,w) for (t,w,_,_) in _s if T0<=t<=T1]
J=sum((pts[k+1][0]-pts[k][0])*0.5*(pts[k][1]+pts[k+1][1]) for k in range(len(pts)-1)) if len(pts)>1 else None
cs=sorted(c for (t,_,c,_) in _s if T0<=t<=T1); rs=0
for (t,_,_,r) in _s:
    if T0<=t<=T1 and isinstance(r,int) and r>0: rs|=r
med=st.median(steptimes); tok_step=args.batch*args.seq; nan=sum(1 for l in losses if l!=l)
# FP8 fired-counter read (RTLD_NOLOAD: only succeeds if substrate already injected into THIS process)
fp8={}
try:
    import ctypes
    lib=ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so", mode=os.RTLD_LAZY|os.RTLD_NOLOAD)
    for sym in ("cipher_rt_fp8_calls_total","cipher_rt_fp8_calls_handled","cipher_rt_fp8_calls_skipped"):
        fn=getattr(lib,sym); fn.restype=ctypes.c_ulong; fp8[sym]=int(fn())
except OSError: fp8={"substrate_loaded":False}
except Exception as ex: fp8={"err":repr(ex)}
res={"mode":"lora_finetune","model":MODEL,"flop_model":"4*N_params/token (LoRA frozen base: fwd 2N + input-grad 2N; no base weight-grad; adapter <1%); FINETUNE not pretrain",
     "n_params":NPARAMS,"peak_flops":PEAK,"batch":args.batch,"seq":args.seq,"steps":args.steps,"pin_mhz":args.pin_mhz,
     "step_time_s_median":med,"step_time_s_mean":st.mean(steptimes),"tokens_per_step":tok_step,
     "mfu_median_step":(tok_step*4*NPARAMS/med)/PEAK,
     "window_s":T1-T0,"joules":J,"avg_power_w":(sum(w for _,w in pts)/len(pts) if pts else None),
     "tok_per_joule":(tok_step*args.steps/J if J else None),
     "loss_first":losses[0],"loss_last":losses[-1],"nan_steps":nan,"stability":("STABLE" if nan==0 else "NaN-DIVERGED"),
     "loss_curve":losses,
     "clocks":({"sm_med":cs[len(cs)//2],"sm_min":cs[0],"sm_max":cs[-1],"throttle_or":hex(rs),"sw_power_cap":bool(rs&0x4)} if cs else None),
     "peak_mem_GB":torch.cuda.max_memory_allocated()/1e9,"fp8_counters":fp8}
json.dump(res,open(args.result,"w"),indent=1)
if args.loss_out: json.dump(losses,open(args.loss_out,"w"))
print("RESULT",json.dumps({k:v for k,v in res.items() if k not in ("loss_curve","flop_model")}))
NV.nvmlShutdown()
