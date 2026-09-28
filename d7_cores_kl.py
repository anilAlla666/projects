# CO-RESIDENT real-KL: load all families ONCE, bind each, teacher-force each on its SOLO gold,
# capture logits, compute KL(solo||cores) + max-abs-logit-diff per step. + mis-route + sub.
import ctypes, hashlib, json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
lib=ctypes.CDLL("/usr/lib/cipher/libcipher_rt.so")
lib.cipher_rt_marlin_engine_bind_model.argtypes=[ctypes.c_void_p,ctypes.c_ulonglong]
lib.cipher_rt_marlin_engine_observe_weight.argtypes=[ctypes.c_void_p]; lib.cipher_rt_marlin_engine_observe_weight.restype=ctypes.c_int
lib.cipher_rt_marlin_calls_bf16_substituted.restype=ctypes.c_ulong
FAM=[("mistral","/home/ubuntu/models/Mistral-7B-v0.1"),("qwen2","Qwen/Qwen2-7B"),
     ("phi2","microsoft/phi-2"),("tinyllama","/home/ubuntu/models/TinyLlama-1.1B")]
def mid(n): return int.from_bytes(hashlib.md5(n.encode()).digest()[:8],'little')|1
M={}
for n,p in FAM:
    t=AutoTokenizer.from_pretrained(p,trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    mm=AutoModelForCausalLM.from_pretrained(p,dtype=torch.bfloat16,trust_remote_code=True).cuda().eval()
    for _,pp in mm.named_parameters():
        if pp.dim()==2: lib.cipher_rt_marlin_engine_bind_model(ctypes.c_void_p(pp.data_ptr()),mid(n))
    M[n]=(t,mm)
print("co-resident %d families loaded"%len(M),flush=True)
ALLOK=True
for n,(t,mm) in M.items():
    try: d=json.load(open("/home/ubuntu/d7_solo_%s.json"%n)); solo_log=np.load("/home/ubuntu/d7_solo_%s.npy"%n)
    except Exception as e: print("CORES %s NO-SOLO-REF %s"%(n,e),flush=True); ALLOK=False; continue
    gold=d["gold"]; ids=t("The history of computing spans several distinct",return_tensors="pt").input_ids.cuda()
    s0=lib.cipher_rt_marlin_calls_bf16_substituted(); cl=[]
    with torch.no_grad():
        out=mm(ids,use_cache=True); past=out.past_key_values; lg=out.logits[:,-1,:].float()
        for tstep in range(len(gold)):
            cl.append(lg[0].cpu().numpy())
            out=mm(torch.tensor([[gold[tstep]]],device='cuda'),past_key_values=past,use_cache=True); past=out.past_key_values
            lg=out.logits[:,-1,:].float()
    sub=lib.cipher_rt_marlin_calls_bf16_substituted()-s0
    cl=np.stack(cl); sl=solo_log[:len(cl)]
    maxdiff=float(np.max(np.abs(cl-sl)))
    lp=torch.log_softmax(torch.tensor(sl),-1); lq=torch.log_softmax(torch.tensor(cl),-1)
    kl=float((lp.exp()*(lp-lq)).sum(-1).max())
    ok=(maxdiff<1e-3) and sub>0   # logit-identity (FP-tol) + Marlin engaged
    ALLOK=ALLOK and ok
    print("CORES %-9s sub=%d max_logit_diff=%.3e KL=%.3e %s"%(n,sub,maxdiff,kl,"OK" if ok else "FAIL"),flush=True)
# mis-route in-context
W=ctypes.c_void_p(0x7f00abcd0000)
lib.cipher_rt_marlin_engine_bind_model(W,mid("famA")); ca=[lib.cipher_rt_marlin_engine_observe_weight(W) for _ in range(3)]
lib.cipher_rt_marlin_engine_bind_model(W,mid("famB")); cb=[lib.cipher_rt_marlin_engine_observe_weight(W) for _ in range(2)]
mr=(ca==[1,2,3] and cb==[1,2]); print("MISROUTE A=%s B=%s %s"%(ca,cb,"BLOCKED" if mr else "BLEED"),flush=True)
print("D7_TIGHT_%s"%("PASS" if (ALLOK and mr) else "FAIL"),flush=True)
