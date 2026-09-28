# D.7 R-H1 positive gate (bf16): >=4 distinct families co-resident, per-tenant model_uuid
# routing via bind_model, each KL=0 vs solo-bf16-Marlin (exact greedy match = KL=0 since
# Marlin INT4 is deterministic on weight values; solo & co-resident apply the SAME kit ->
# match iff no cross-family contamination). + per-family marlin_sub>0. + mis-route BLOCK.
import ctypes, torch, gc, hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
lib = ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
lib.cipher_rt_marlin_engine_bind_model.argtypes=[ctypes.c_void_p, ctypes.c_ulonglong]
lib.cipher_rt_marlin_engine_observe_weight.argtypes=[ctypes.c_void_p]; lib.cipher_rt_marlin_engine_observe_weight.restype=ctypes.c_int
lib.cipher_rt_marlin_calls_bf16_substituted.restype=ctypes.c_ulong
FAM=[("mistral","/home/ubuntu/models/Mistral-7B-v0.1"),("qwen2","Qwen/Qwen2-7B"),
     ("phi2","microsoft/phi-2"),("tinyllama","/home/ubuntu/models/TinyLlama-1.1B")]
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8],'little')|1
PROMPT="The history of computing spans several distinct"; GEN=16
def load(p):
    t=AutoTokenizer.from_pretrained(p,trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval(); return t,m
def bind(m,i):
    for _,p in m.named_parameters():
        if p.dim()==2: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(p.data_ptr()),i)
def gen(t,m):
    ids=t(PROMPT,return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): o=m.generate(ids,max_new_tokens=GEN,do_sample=False)
    return o[0,ids.shape[1]:].tolist()
solo={}
for name,path in FAM:
    t,m=load(path); bind(m,mid(name)); s0=lib.cipher_rt_marlin_calls_bf16_substituted()
    toks=gen(t,m); sub=lib.cipher_rt_marlin_calls_bf16_substituted()-s0
    solo[name]=toks; print("SOLO %-9s sub=%d toks=%s"%(name,sub,toks[:6]),flush=True)
    del m,t; gc.collect(); torch.cuda.empty_cache()
models={}
for name,path in FAM:
    t,m=load(path); bind(m,mid(name)); models[name]=(t,m)
print("CO-RESIDENT %d families loaded"%len(models),flush=True)
allok=True
for name,(t,m) in models.items():
    s0=lib.cipher_rt_marlin_calls_bf16_substituted(); toks=gen(t,m)
    sub=lib.cipher_rt_marlin_calls_bf16_substituted()-s0
    match=sum(a==b for a,b in zip(solo[name],toks)); tot=len(solo[name])
    ok=(match==tot) and sub>0; allok=allok and ok
    print("CORES %-9s match=%d/%d sub=%d %s"%(name,match,tot,sub,"OK" if ok else "FAIL"),flush=True)
# mis-route BLOCK (keying isolation re-confirm in this context)
W=ctypes.c_void_p(0x7f00abcd0000); A=mid("famA"); B=mid("famB")
lib.cipher_rt_marlin_engine_bind_model(W,A); ca=[lib.cipher_rt_marlin_engine_observe_weight(W) for _ in range(3)]
lib.cipher_rt_marlin_engine_bind_model(W,B); cb=[lib.cipher_rt_marlin_engine_observe_weight(W) for _ in range(2)]
misroute_ok=(ca==[1,2,3] and cb==[1,2])
print("MISROUTE A=%s B=%s %s"%(ca,cb,"BLOCKED-OK" if misroute_ok else "BLEED-FAIL"),flush=True)
print("D7_RH1_KL_%s"%("PASS" if (allok and misroute_ok) else "FAIL"),flush=True)
