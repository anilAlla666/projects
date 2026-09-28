# Clean SOLO ref (separate process, no shared Marlin state): family bf16+Marlin+bind,
# greedy 16 steps capturing per-step LOGITS + gold tokens + marlin_sub. Saves npy+json.
import os, ctypes, hashlib, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
lib=ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
lib.cipher_rt_marlin_engine_bind_model.argtypes=[ctypes.c_void_p,ctypes.c_ulonglong]
lib.cipher_rt_marlin_calls_bf16_substituted.restype=ctypes.c_ulong
NAME=os.environ["FAMILY"]; PATH=os.environ["FPATH"]; GEN=16
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8],'little')|1
t=AutoTokenizer.from_pretrained(PATH,trust_remote_code=True)
if t.pad_token is None: t.pad_token=t.eos_token
m=AutoModelForCausalLM.from_pretrained(PATH,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
for _,p in m.named_parameters():
    if p.dim()==2: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(p.data_ptr()),mid(NAME))
ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
s0=lib.cipher_rt_marlin_calls_bf16_substituted()
gold=[]; logs=[]
with torch.no_grad():
    out=m(ids,use_cache=True); past=out.past_key_values
    lg=out.logits[:,-1,:].float()
    for _ in range(GEN):
        logs.append(lg[0].cpu().numpy()); nt=int(lg.argmax(-1)); gold.append(nt)
        out=m(torch.tensor([[nt]],device='cuda'),past_key_values=past,use_cache=True); past=out.past_key_values
        lg=out.logits[:,-1,:].float()
sub=lib.cipher_rt_marlin_calls_bf16_substituted()-s0
np.save("/home/ubuntu/d7_solo_%s.npy"%NAME, np.stack(logs))
json.dump({"gold":gold,"sub":int(sub)}, open("/home/ubuntu/d7_solo_%s.json"%NAME,"w"))
print("SOLO %s sub=%d gold=%s saved"%(NAME,sub,gold[:6]),flush=True)
