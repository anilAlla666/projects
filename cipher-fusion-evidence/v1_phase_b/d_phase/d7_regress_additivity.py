# D.7 additivity regression: d10 build (re-key, default-OFF, NO bind_model) must be
# byte-identical to the pre-D anchor on prior models -> proves the re-key doesn't regress.
import os, ctypes, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS=[("tinyllama","/home/ubuntu/models/TinyLlama-1.1B"),("mistral","/home/ubuntu/models/Mistral-7B-v0.1")]
GEN=16; TAG=os.environ["TAG"]   # "anchor" or "d10"
for name,path in MODELS:
    t=AutoTokenizer.from_pretrained(path); 
    if t.pad_token is None: t.pad_token=t.eos_token
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16).cuda().eval()
    ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
    logs=[]
    with torch.no_grad():
        out=m(ids,use_cache=True); past=out.past_key_values; lg=out.logits[:,-1,:].float()
        for _ in range(GEN):
            logs.append(lg[0].cpu().numpy()); nt=int(lg.argmax(-1))
            out=m(torch.tensor([[nt]],device='cuda'),past_key_values=past,use_cache=True); past=out.past_key_values
            lg=out.logits[:,-1,:].float()
    np.save("/home/ubuntu/reg_%s_%s.npy"%(name,TAG), np.stack(logs))
    del m; torch.cuda.empty_cache()
    print("REG %s %s saved"%(name,TAG),flush=True)
