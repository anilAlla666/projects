import os, time, torch
from transformers import AutoModelForCausalLM
M="/home/ubuntu/models/Mistral-7B-v0.1"
print("loading Mistral-7B bf16...", flush=True)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
B,S=64,512
ids=torch.randint(1,31000,(B,S),device='cuda')
with torch.no_grad():
    for i in range(3):
        out=model(ids); torch.cuda.synchronize()
print("FORWARD_DONE logits", tuple(out.logits.shape), flush=True)
