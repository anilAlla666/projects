# Measure the forward's GEMM vs non-GEMM split (structural; plain bf16, no CIPHER) — is it GEMM-bound?
import torch
from transformers import AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
M="/home/ubuntu/models/Mistral-7B-v0.1"
m=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids=torch.randint(1,31000,(64,512),device='cuda')
with torch.no_grad():
    for _ in range(5): m(ids); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): m(ids)
        torch.cuda.synchronize()
ka=prof.key_averages()
tot=sum(e.self_device_time_total for e in ka)
def bucket(name):
    n=name.lower()
    if any(s in n for s in ['gemm','cutlass','ampere','sm90','volta','nvjet','matmul','wgrad','dgrad','s16816','f16f16','cublas']): return 'GEMM'
    if any(s in n for s in ['attention','flash','fmha','mha','efficient','fused_attn','sdpa']): return 'ATTN'
    if any(s in n for s in ['norm','rms','softmax','silu','mul','add','elementwise','rope','cat','copy','index']): return 'NORM/ELEM'
    if any(s in n for s in ['cipher_fp8','absmax','fp8_quant']): return 'FP8_QUANT'
    return 'OTHER'
agg={}
for e in ka:
    agg[bucket(e.key)]=agg.get(bucket(e.key),0)+e.self_device_time_total
print(f"total CUDA self time: {tot/1e3:.1f} ms (5 forwards)")
for k in ['GEMM','FP8_QUANT','ATTN','NORM/ELEM','OTHER']:
    v=agg.get(k,0); print(f"  {k:10s} {v/1e3:8.1f} ms  {100*v/tot:5.1f}%")
print("\nTop 12 kernels by self CUDA time:")
for e in sorted(ka,key=lambda x:-x.self_device_time_total)[:12]:
    print(f"  {100*e.self_device_time_total/tot:5.1f}%  {e.self_device_time_total/1e3:7.1f}ms  {e.key[:70]}")
