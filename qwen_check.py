import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p="Qwen/Qwen2-7B"
t=AutoTokenizer.from_pretrained(p); 
if t.pad_token is None: t.pad_token=t.eos_token
m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16).cuda().eval()
ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
with torch.no_grad(): o=m.generate(ids,max_new_tokens=16,do_sample=False)
print("QWEN tokens=", o[0,ids.shape[1]:].tolist()[:8])
