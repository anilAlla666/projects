#!/usr/bin/env python3
# P0c: training-step kernel-time decomposition (pure torch, in-process torch.profiler; no substrate).
# Buckets fwd+bwd+optimizer CUDA kernels: GEMM / attention / pointwise / optimizer-elementwise / other.
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.profiler import profile, ProfilerActivity
from collections import defaultdict
B,S=2,2048
m=AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",torch_dtype=torch.float16,attn_implementation="sdpa")
lc=LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_dropout=0.0,bias="none",task_type="CAUSAL_LM")
m=get_peft_model(m,lc); m.cuda(); m.train()
for n,p in m.named_parameters():
    if p.requires_grad: p.data=p.data.float()
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5)
tok=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
ids=tok("The history of computing began "*400,return_tensors="pt")["input_ids"][0][:S].unsqueeze(0).repeat(B,1).cuda()
tr=[p for p in m.parameters() if p.requires_grad]
def step():
    o=m(input_ids=ids,labels=ids); l=o.loss; opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(tr,1.0); opt.step(); return float(l.detach())
for _ in range(5): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
ev=prof.key_averages()
kern=defaultdict(float)
for e in ev:
    if e.device_time_total>0: kern[e.key]+=e.device_time_total
tot=sum(kern.values())
def bucket(n):
    s=n.lower()
    if any(t in s for t in ("flash","fmha","fwd_kernel","attention","sdpa","scaled_dot","softmax","_attn","mha")): return "ATTENTION"
    if any(t in s for t in ("nvjet","gemm","cutlass","wgmma","cublas","sm90_xmma","s16816","ampere_")): return "GEMM"
    if any(t in s for t in ("adam","fused_adam","optimizer","l2norm","clip","foreach","_amp","grad_scale")): return "OPTIMIZER"
    if any(t in s for t in ("rmsnorm","rms_norm","norm","silu","swiglu","gelu","activation","rotary","rope","elementwise","add","mul","cast","convert","copy","memcpy","memset","fill","cat","_to_copy")): return "POINTWISE"
    return "OTHER"
buck=defaultdict(float)
for n,d in kern.items(): buck[bucket(n)]+=d
out={"workload":f"LoRA-finetune Mistral-7B B={B}xS={S} fwd+bwd+AdamW, 3 steps profiled, sdpa","total_kernel_ms":round(tot/1000,1),"n_kernels":len(kern),
     "buckets_pct":{b:round(100*buck[b]/tot,2) for b in ("GEMM","ATTENTION","POINTWISE","OPTIMIZER","OTHER")},
     "top":{b:[{"k":n[:80],"pct":round(100*d/tot,2)} for n,d in sorted([(n,d) for n,d in kern.items() if bucket(n)==b],key=lambda x:-x[1])[:6]] for b in ("GEMM","ATTENTION","POINTWISE","OPTIMIZER","OTHER")}}
json.dump(out,open("train_gemmshare.json","w"),indent=1)
for b in ("GEMM","ATTENTION","POINTWISE","OPTIMIZER","OTHER"):
    print(f"{b}: {out['buckets_pct'][b]}%")
    for t in out["top"][b][:4]: print(f"   {t['pct']:5.2f}% {t['k']}")
