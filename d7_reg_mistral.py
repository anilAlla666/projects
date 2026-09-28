import os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p="/home/ubuntu/models/Mistral-7B-v0.1"; TAG=os.environ["TAG"]; GEN=16
t=AutoTokenizer.from_pretrained(p); t.pad_token=t.eos_token
m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16).cuda().eval()
ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
logs=[]
with torch.no_grad():
    o=m(ids,use_cache=True); past=o.past_key_values; lg=o.logits[:,-1,:].float()
    for _ in range(GEN):
        logs.append(lg[0].cpu().numpy()); nt=int(lg.argmax(-1))
        o=m(torch.tensor([[nt]],device='cuda'),past_key_values=past,use_cache=True); past=o.past_key_values; lg=o.logits[:,-1,:].float()
np.save("/home/ubuntu/regm_%s.npy"%TAG,np.stack(logs)); print("REGM %s saved"%TAG,flush=True)
