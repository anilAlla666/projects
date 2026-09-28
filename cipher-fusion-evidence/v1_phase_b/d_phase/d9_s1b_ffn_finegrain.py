#!/usr/bin/env python3
# D.9 §1b — FIRST-PARTY verify of the granularity confound: FFN-only FP8 at PRODUCTION
# granularity (per-CHANNEL weight scale + per-TOKEN activation scale, rowwise torch._scaled_mm)
# vs the crude per-tensor (66%/64%). If finer scaling still DECLINEs (far from 100% token-id),
# the 0% coverage is a contract x E4M3-numerics property, NOT a crude-scaling artifact.
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
PROMPTS=[
 "The capital of France is","Water boils at a temperature of","The first president of the United States was",
 "In 1969, humans first landed on the","Photosynthesis is the process by which plants","The speed of light in a vacuum is approximately",
 "A prime number is a number that","The mitochondria is the powerhouse of the","To make a cup of tea, first you",
 "The theory of relativity was developed by","An example of a renewable energy source is","The largest planet in our solar system is",
 "Shakespeare wrote many famous plays including","The chemical symbol for gold is","DNA stands for",
 "The Great Wall of China was built to","A triangle has three","The currency used in Japan is the",
 "Machine learning is a subset of","The human heart has four","Mount Everest is the tallest",
 "The process of converting water to vapor is called","In computer science, an algorithm is","The Pacific is the largest",
 "A balanced diet should include","The French Revolution began in the year","Gravity causes objects to",
 "The smallest unit of life is the","Electricity flows through a conductor because","The author of Romeo and Juliet is",
 "A noun is a word that names a","The boiling point of water decreases at high","Solar panels convert sunlight into",
 "The square root of sixteen is","Bees are important because they","The longest river in the world is",
 "An atom consists of protons, neutrons, and","To solve a quadratic equation you can use","The opposite of hot is",
 "A democracy is a system of government in which","The freezing point of water is","Photosystem II is part of",
 "The capital of Japan is","A byte consists of eight","The force that pulls objects toward Earth is",
 "Recursion in programming is when a function","The primary colors are red, blue, and","The largest ocean on Earth is the",
 "A haiku is a form of poetry that","The inventor of the telephone was",
]
def make_rowwise_fp8_fwd(lin):
    K=lin.in_features; w=lin.weight.data                       # [N,K]
    wscale=(w.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32)  # [N,1] per-channel
    w8=(w/wscale).to(torch.float8_e4m3fn); bias=lin.bias
    sb=wscale.t().contiguous()                                  # [1,N] per-channel
    def fwd(x):
        shp=x.shape; x2=x.reshape(-1,K).contiguous()
        ascale=(x2.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32)  # [M,1] per-token
        x8=(x2/ascale).to(torch.float8_e4m3fn)
        try:
            o=torch._scaled_mm(x8, w8.t(), scale_a=ascale, scale_b=sb, out_dtype=torch.float16, use_fast_accum=True)
        except Exception:
            o=torch._scaled_mm(x8, w8.t(), scale_a=ascale, scale_b=sb, out_dtype=torch.float16)
        if bias is not None: o=o+bias
        return o.reshape(*shp[:-1], o.shape[-1])
    return fwd

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
enc=tok(PROMPTS,return_tensors="pt",padding=True).to("cuda")
def gen(ds):
    with torch.no_grad():
        o=m.generate(**enc,max_new_tokens=24,do_sample=ds,temperature=(0.8 if ds else None),top_p=(0.95 if ds else None),top_k=(50 if ds else None),pad_token_id=tok.pad_token_id)
    return o[:,enc["input_ids"].shape[1]:].cpu()
torch.manual_seed(0); rg=gen(False); torch.manual_seed(0); rs=gen(True)
# apply rowwise FP8 to FFN linears only
sel=[(n,mod) for n,mod in m.named_modules() if isinstance(mod,torch.nn.Linear) and any(k in n for k in ["mlp.gate","mlp.up","mlp.down"])]
orig={id(mod):mod.forward for _,mod in sel}
for _,mod in sel: mod.forward=make_rowwise_fp8_fwd(mod)
torch.manual_seed(0); g=gen(False); torch.manual_seed(0); s=gen(True)
for _,mod in sel: mod.forward=orig[id(mod)]
gr=float((g==rg).all(dim=1).float().mean()*100); sr=float((s==rs).all(dim=1).float().mean()*100)
out={"granularity":"per_channel_weight+per_token_activation(rowwise _scaled_mm)","class":"FFN","n_layers":len(sel),
     "n_prompts":len(PROMPTS),"greedy_pct":round(gr,1),"sampled_pct":round(sr,1),"FP8_SAFE":(gr==100.0 and sr==100.0),
     "crude_per_tensor_ref":{"greedy":66.0,"sampled":64.0}}
print(f"[FFN rowwise-FP8] {len(sel)} linears -> greedy {round(gr,1)}% / sampled {round(sr,1)}% token-identical -> {'SAFE' if out['FP8_SAFE'] else 'DECLINE'} (crude per-tensor was 66/64)",flush=True)
json.dump(out,open("/home/ubuntu/d9_s1b_ffn_finegrain_verify.json","w"),indent=2); print("WROTE d9_s1b_ffn_finegrain_verify.json",flush=True)
