import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M="/home/ubuntu/models/Mistral-7B-v0.1"
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).cuda().eval()
ids=tok(["The history of computing spans"]*8, return_tensors="pt", padding=True).input_ids.cuda()
with torch.no_grad():
    out=m.generate(ids, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize(); print("done")
