#!/usr/bin/env python3
# A1 train-regime bench — SCRATCH copy of wi1_trainingbadput/wi1_train.py (fork-1 untouched) extended with
# the NVML 50ms trapezoid energy sampler + MFU/TPW computation. CLEAN torch, NO CIPHER substrate, NO shim.
# Stability recipe preserved from wi1: real coherent text, grad-clip 1.0, fp32 adapter params, eager attention.
# Train FLOP model (stated): FLOPs/token = 4*N_params  (forward 2N + input-grad 2N; base weights FROZEN under
# LoRA so weight-grad GEMMs are skipped; LoRA r=16 adapter terms <1% extra; attention FLOPs excluded).
import os, time, json, threading, argparse, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--result", default=None)
args = ap.parse_args()
MODEL = "mistralai/Mistral-7B-v0.1"
RESULT = args.result or f"/home/ubuntu/cipher-fusion-evidence/mfu_audit/m_train_B{args.batch}.json"
NPARAMS, PEAK_FLOPS = 7_241_732_096, 989.5e12

import pynvml as NV
NV.nvmlInit(); H = NV.nvmlDeviceGetHandleByIndex(0)
_samples = []; _stop = threading.Event()
def _throttle():
    for fn in ("nvmlDeviceGetCurrentClocksThrottleReasons", "nvmlDeviceGetCurrentClocksEventReasons"):
        try: return int(getattr(NV, fn)(H))
        except Exception: pass
    return -1
def _sampler():
    while not _stop.is_set():
        try:
            _samples.append((time.monotonic(), NV.nvmlDeviceGetPowerUsage(H)/1000.0,
                             NV.nvmlDeviceGetClockInfo(H, NV.NVML_CLOCK_SM), _throttle()))
        except Exception: pass
        _stop.wait(0.05)

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="eager")
lcfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.cuda(); model.train()
for n, p in model.named_parameters():
    if p.requires_grad: p.data = p.data.float()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
tok = AutoTokenizer.from_pretrained(MODEL)
_passage = ("The history of computing began with mechanical calculators and evolved through vacuum tubes, "
            "transistors, integrated circuits, and now massively parallel accelerators used to train and serve "
            "large language models in modern datacenters. ")
_ids = tok(_passage * 40, return_tensors="pt")["input_ids"][0][:args.seq]
FIXED_IDS = _ids.unsqueeze(0).repeat(args.batch, 1).cuda()
trainable = [p for p in model.parameters() if p.requires_grad]

def one_step():
    out = model(input_ids=FIXED_IDS, labels=FIXED_IDS)
    loss = out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    return float(loss.detach())

for _ in range(args.warmup): one_step()
torch.cuda.synchronize()

th = threading.Thread(target=_sampler, daemon=True); th.start()
step_times = []; losses = []
try: e0 = NV.nvmlDeviceGetTotalEnergyConsumption(H)
except Exception: e0 = None
T0 = time.monotonic()
for s in range(args.steps):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    losses.append(one_step())
    torch.cuda.synchronize(); step_times.append(time.perf_counter() - t0)
T1 = time.monotonic()
try: e1 = NV.nvmlDeviceGetTotalEnergyConsumption(H)
except Exception: e1 = None
_stop.set(); th.join(timeout=2)

pts = [(t, w) for (t, w, _, _) in _samples if T0 <= t <= T1]
J = sum((pts[i+1][0]-pts[i][0])*0.5*(pts[i][1]+pts[i+1][1]) for i in range(len(pts)-1)) if len(pts) > 1 else None
cs = sorted(c for (t, _, c, _) in _samples if T0 <= t <= T1)
rs = 0
for (t, _, _, r) in _samples:
    if T0 <= t <= T1 and isinstance(r, int) and r > 0: rs |= r
tok_step = args.batch*args.seq
med = st.median(step_times)
window = T1 - T0
tokens_total = tok_step*args.steps
nan_steps = sum(1 for l in losses if l != l)
res = {"mode": "train_lora", "model": MODEL, "pid": os.getpid(), "lora_r": 16, "dtype": "fp16",
       "attn": "eager", "batch": args.batch, "seq": args.seq, "steps": args.steps,
       "flop_model": "4*N_params per token (fwd 2N + input-grad 2N; frozen base => no weight-grad; LoRA r=16 <1%; attention excluded)",
       "n_params": NPARAMS, "peak_flops": PEAK_FLOPS,
       "step_time_s_median": med, "step_time_s_mean": st.mean(step_times),
       "step_time_s_p90": sorted(step_times)[int(0.9*len(step_times))-1],
       "tokens_per_step": tok_step, "train_tok_s_window": tokens_total/window,
       "mfu_median_step": (tok_step*4*NPARAMS/med)/PEAK_FLOPS,
       "window_s": window, "joules_trapezoid": J,
       "joules_counter": (e1-e0)/1000.0 if (e0 is not None and e1 is not None) else None,
       "avg_power_w": (sum(w for _, w in pts)/len(pts) if pts else None), "n_power_samples": len(pts),
       "train_tok_per_joule": (tokens_total/J if J else None),
       "loss_first": losses[0], "loss_last": losses[-1], "nan_loss_steps": nan_steps,
       "stability": ("STABLE" if nan_steps == 0 else "NaN-DIVERGED"),
       "clocks": ({"sm_clock_median": cs[len(cs)//2], "sm_clock_min": cs[0], "sm_clock_max": cs[-1],
                   "throttle_reasons_or": hex(rs), "sw_power_cap": bool(rs & 0x4)} if cs else None),
       "peak_mem_GB": torch.cuda.max_memory_allocated()/1e9}
with open(RESULT, "w") as f: json.dump(res, f, indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "flop_model"}))
