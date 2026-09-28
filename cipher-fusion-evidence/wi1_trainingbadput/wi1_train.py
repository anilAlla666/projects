#!/usr/bin/env python3
# WI-1: real Mistral-7B LoRA finetune step loop on one H100. Measures real step time (fwd+bwd+optimizer)
# and checkpoint write time. fp16 to match the R.A cuBLAS detector (which assumes fp16 GEMMs). Convergence
# is irrelevant; step TIMING + the rollback accounting must be real. Optional persistent-SDC injection +
# within-N detection is driven by the LD_PRELOADed wi1_shim.so (env RV_*).
import os, sys, time, json, argparse
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--ckpt-c", type=int, default=100)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--result", default="/home/ubuntu/cipher-fusion-evidence/wi1_trainingbadput/wi1_steptime.json")
ap.add_argument("--ckpt-dir", default="/home/ubuntu/cipher-fusion-evidence/wi1_trainingbadput/ckpt")
args = ap.parse_args()
MODEL = "mistralai/Mistral-7B-v0.1"

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="eager")
lcfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.cuda(); model.train()
# fp16 LoRA: keep adapter params fp32 for the optimizer, base stays fp16
for n,p in model.named_parameters():
    if p.requires_grad: p.data = p.data.float()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
V = model.config.vocab_size
os.makedirs(args.ckpt_dir, exist_ok=True)
# Real coherent text (repeated) so the fp16 forward stays in-distribution and does not diverge to NaN —
# training-divergence NaN is orthogonal to the SDC measurement and would confound the fp16 detector.
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

# warmup
for _ in range(args.warmup): one_step()
torch.cuda.synchronize()

step_times = []; ckpt_times = []
t_all0 = time.perf_counter()
for s in range(args.steps):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss = one_step()
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    step_times.append(dt)
    if (s+1) % args.ckpt_c == 0:
        tc0 = time.perf_counter()
        sd = {k: v for k,v in model.state_dict().items() if "lora" in k.lower()}
        torch.save(sd, os.path.join(args.ckpt_dir, f"step{s+1}.pt"))
        ckpt_times.append(time.perf_counter() - tc0)
torch.cuda.synchronize()
import statistics as st
res = {"model": MODEL, "lora_r": 16, "batch": args.batch, "seq": args.seq, "steps": args.steps,
       "dtype": "fp16", "step_time_s_mean": st.mean(step_times), "step_time_s_median": st.median(step_times),
       "step_time_s_min": min(step_times), "step_time_s_p90": sorted(step_times)[int(0.9*len(step_times))-1],
       "ckpt_c": args.ckpt_c, "ckpt_write_s": (st.mean(ckpt_times) if ckpt_times else None),
       "n_trainable_M": sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,
       "peak_mem_GB": torch.cuda.max_memory_allocated()/1e9,
       "last_loss": loss}
with open(args.result, "w") as f: json.dump(res, f, indent=2)
print("WI1_STEP", json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in res.items() if k!="last_loss"}))
