#!/usr/bin/env python3
# Panel-fix Probe B (M5/M11): kernel + host-API decomposition of the FP8-ENGAGED training step.
# Same model/step as profile_train.py (vanilla P0c profile); engagement ONLY from caller env.
# Adds: FP8QUANT bucket (cipher_fp8_* NVRTC kernels), CPU activity for cuda*-API host time
# (cudaMalloc/cudaFree/cudaStreamSynchronize churn), and wall-vs-GPU-kernel-time gap.
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.profiler import profile, ProfilerActivity
from collections import defaultdict
B, S = 2, 2048
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, attn_implementation="sdpa")
lc = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
m = get_peft_model(m, lc); m.cuda(); m.train()
for n, p in m.named_parameters():
    if p.requires_grad: p.data = p.data.float()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
ids = tok("The history of computing began "*400, return_tensors="pt")["input_ids"][0][:S].unsqueeze(0).repeat(B,1).cuda()
tr = [p for p in m.parameters() if p.requires_grad]
def step():
    o = m(input_ids=ids, labels=ids); l = o.loss; opt.zero_grad(set_to_none=True); l.backward()
    torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); return float(l.detach())
for _ in range(5): step()
torch.cuda.synchronize()
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
wall = time.perf_counter() - t0
ev = prof.key_averages()
kern = defaultdict(float); api = {}
for e in ev:
    if e.device_time_total > 0: kern[e.key] += e.device_time_total
    if e.key.startswith("cuda") and e.cpu_time_total > 0:
        api[e.key] = {"count": e.count, "cpu_ms_total": round(e.cpu_time_total/1e3, 2)}
tot = sum(kern.values())
def bucket(n):
    s = n.lower()
    if "cipher_fp8" in s: return "FP8QUANT"
    if any(t in s for t in ("flash","fmha","fwd_kernel","attention","sdpa","scaled_dot","softmax","_attn","mha")): return "ATTENTION"
    if any(t in s for t in ("nvjet","gemm","cutlass","wgmma","cublas","sm90_xmma","s16816","ampere_","matmul")): return "GEMM"
    if any(t in s for t in ("adam","fused_adam","optimizer","l2norm","clip","foreach","_amp","grad_scale")): return "OPTIMIZER"
    if any(t in s for t in ("rmsnorm","rms_norm","norm","silu","swiglu","gelu","activation","rotary","rope","elementwise","add","mul","cast","convert","copy","memcpy","memset","fill","cat","_to_copy")): return "POINTWISE"
    return "OTHER"
bk = defaultdict(float); top = defaultdict(list)
for k, v in kern.items():
    b = bucket(k); bk[b] += v; top[b].append((k, v))
res = {"workload": "FP8-ENGAGED LoRA-finetune Mistral-7B B=2xS=2048 fwd+bwd+AdamW, 3 steps profiled, sdpa",
       "wall_s_3steps": round(wall, 4), "gpu_kernel_s_3steps": round(tot/1e6, 4),
       "host_gap_s_3steps": round(wall - tot/1e6, 4),
       "buckets_pct": {b: round(100*v/tot, 2) for b, v in sorted(bk.items(), key=lambda x: -x[1])},
       "buckets_ms_per_step": {b: round(v/1e3/3, 1) for b, v in sorted(bk.items(), key=lambda x: -x[1])},
       "cuda_api_host": {k: v for k, v in sorted(api.items(), key=lambda x: -x[1]["cpu_ms_total"])[:12]},
       "top_kernels": {b: [{"k": k[:90], "ms_per_step": round(v/1e3/3, 1)} for k, v in sorted(l, key=lambda x: -x[1])[:6]] for b, l in top.items()}}
json.dump(res, open("fp8_arm_profile.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("wall_s_3steps","gpu_kernel_s_3steps","host_gap_s_3steps","buckets_pct","cuda_api_host")}, indent=1))
