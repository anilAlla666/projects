# Clean co-residence test: load 4 families ONCE (no free/reload churn), bind each to its
# model_uuid, run each greedy. Compare to known solo refs. Tests co-residence isolation
# without the load/free confound that polluted the prior run.
import ctypes, torch, hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
lib=ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
lib.cipher_rt_marlin_engine_bind_model.argtypes=[ctypes.c_void_p,ctypes.c_ulonglong]
lib.cipher_rt_marlin_calls_bf16_substituted.restype=ctypes.c_ulong
FAM=[("mistral","/home/ubuntu/models/Mistral-7B-v0.1"),("qwen2","Qwen/Qwen2-7B"),
     ("phi2","microsoft/phi-2"),("tinyllama","/home/ubuntu/models/TinyLlama-1.1B")]
# solo refs (mistral from Phase A clean-first; qwen2 from isolated test; others recompute solo-in-sep below not needed)
SOLO={"mistral":[1234,293,28725,1430,395,871],"qwen2":[2714,300,11,1817,31871,553]}
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8],'little')|1
def mkmodel(p):
    t=AutoTokenizer.from_pretrained(p,trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    return t,AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
M={}
for n,p in FAM:
    t,m=mkmodel(p)
    for _,pp in m.named_parameters():
        if pp.dim()==2: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(pp.data_ptr()),mid(n))
    M[n]=(t,m)
print("loaded %d families co-resident (no churn)"%len(M),flush=True)
for n,(t,m) in M.items():
    ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
    s0=lib.cipher_rt_marlin_calls_bf16_substituted()
    with torch.no_grad(): o=m.generate(ids,max_new_tokens=16,do_sample=False)
    sub=lib.cipher_rt_marlin_calls_bf16_substituted()-s0
    toks=o[0,ids.shape[1]:].tolist()
    ref=SOLO.get(n); 
    tag = ("match=%d/%d"%(sum(a==b for a,b in zip(ref,toks)),len(ref))) if ref else "no-ref"
    print("CORES %-9s sub=%d %s toks=%s"%(n,sub,tag,toks[:6]),flush=True)
print("CORES_CLEAN_DONE",flush=True)
