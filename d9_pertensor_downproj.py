#!/usr/bin/env python3
# D.9 confound-closer: per-TENSOR weight-scale FP8 (scalar amax, NOT per-channel rowwise),
# drop_downproj coverage (down_proj stays fp16). This is the ONE unmeasured cell on the only
# path to REACHED: per-tensor speed (modeled 85.9% MFU) at the within-bar coverage. Question:
# does per-tensor drop_downproj PPL stay <=0.3%? per-channel drop_downproj was +0.2465% PASS;
# per-tensor all-fp8 was +0.44% vs per-channel all-fp8 +0.37% (~0.07pp worse). Measure it.
import os, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"; WIN=2048; NWIN=50
def make_fwd_pertensor(lin):
    K=lin.in_features; w=lin.weight.data
    # PER-TENSOR scalar weight scale (single amax over whole weight) -- the per-tensor scheme
    ws=(w.abs().amax()/448.0).clamp(min=1e-12).to(torch.float32).reshape(1,1)
    w8=(w/ws).to(torch.float8_e4m3fn); bias=lin.bias
    def fwd(x):
        shp=x.shape; x2=x.reshape(-1,K).contiguous()
        # PER-TENSOR activation scale (single scalar) -- matches per-tensor scheme
        a=(x2.abs().amax()/448.0).clamp(min=1e-12).to(torch.float32).reshape(1,1)
        x8=(x2/a).to(torch.float8_e4m3fn)
        try: o=torch._scaled_mm(x8,w8.t(),scale_a=a,scale_b=ws,out_dtype=torch.float16,use_fast_accum=True)
        except Exception: o=torch._scaled_mm(x8,w8.t(),scale_a=a,scale_b=ws,out_dtype=torch.float16)
        if bias is not None: o=o+bias
        return o.reshape(*shp[:-1],o.shape[-1])
    return fwd
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
wt=datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="test"); text="\n\n".join([t for t in wt["text"] if t.strip()])
ids=tok(text,return_tensors=None)["input_ids"]; windows=[torch.tensor([ids[i*WIN:(i+1)*WIN]],device="cuda") for i in range(NWIN)]
def ppl():
    nll=0.0;n=0
    with torch.no_grad():
        for w in windows:
            lsm=torch.log_softmax(m(w).logits[0].float(),-1); tgt=w[0,1:]
            nll+=float(-lsm[:-1].gather(1,tgt.unsqueeze(1)).sum()); n+=tgt.numel()
    return math.exp(nll/n)
def linears(pred):
    out=[]
    for n,mod in m.named_modules():
        if not isinstance(mod,torch.nn.Linear) or "lm_head" in n: continue
        if pred(n): out.append((n,mod))
    return out
def apply(sel):
    o={id(x):x.forward for _,x in sel}
    for _,x in sel: x.forward=make_fwd_pertensor(x)
    return o
def restore(sel,o):
    for _,x in sel: x.forward=o[id(x)]
ppl16=ppl(); print(f"[fp16] PPL={ppl16:.4f}",flush=True)
# all_fp8 per-tensor (sanity vs the +0.44% vLLM number) then drop_downproj per-tensor (the unmeasured cell)
for name,pred in [("all_fp8_pertensor",lambda n:True),("drop_downproj_pertensor",lambda n:"mlp.down" not in n)]:
    sel=linears(pred); o=apply(sel); p=ppl(); restore(sel,o)
    pd=100.0*(p-ppl16)/ppl16
    print(f"[{name}] n={len(sel)} PPL={p:.4f} delta={pd:+.4f}% ({'PASS<=0.3' if pd<=0.3 else 'FAIL>0.3'})",flush=True)
    if name=="all_fp8_pertensor": all_pd=pd
    else: dd_pd=pd; dd_ppl=p
out={"method":"per_TENSOR W scalar + per_TENSOR A scalar FP8 E4M3 (HF forward)","fp16_ppl":ppl16,
     "all_fp8_pertensor_delta_pct":round(all_pd,4),
     "drop_downproj_pertensor_ppl":round(dd_ppl,4),"drop_downproj_pertensor_delta_pct":round(dd_pd,4),
     "drop_downproj_pertensor_within_bar":dd_pd<=0.3}
json.dump(out,open("/home/ubuntu/d9_pertensor_downproj_result.json","w"),indent=2)
print("WROTE d9_pertensor_downproj_result.json",flush=True)
