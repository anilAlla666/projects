#!/usr/bin/env python3
# D.9 BUILD step 3 — PER-LAYER bounded-quality back-off (contingent: all-fp8 per-channel was
# +0.37% PPL, narrowly over the 0.3% bar). Find the smallest FP8 decline-set that brings PPL<=0.3%,
# report COVERAGE (GEMM-time fraction still FP8) + MFU at that coverage (harmonic blend of the
# measured all-fp8 MFU 98.3% and fp16 baseline 64%). PPL on 50x2048; MMLU on the chosen config.
import os, sys, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"; WIN=2048; NWIN=50
MMLU_SUBJECTS=["abstract_algebra","anatomy","astronomy","college_computer_science","high_school_mathematics"]; MMLU_PER=100
# GEMM-time weights (per all-layers): FFN gate/up/down 27% each; attn q/o 7.6% each, k/v 1.9% each. Per layer-index = 1/32 of total.
def make_fwd(lin):
    K=lin.in_features; w=lin.weight.data; ws=(w.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32)
    w8=(w/ws).to(torch.float8_e4m3fn); bias=lin.bias; sb=ws.t().contiguous()
    def fwd(x):
        shp=x.shape; x2=x.reshape(-1,K).contiguous(); a=(x2.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32)
        x8=(x2/a).to(torch.float8_e4m3fn)
        try: o=torch._scaled_mm(x8,w8.t(),scale_a=a,scale_b=sb,out_dtype=torch.float16,use_fast_accum=True)
        except Exception: o=torch._scaled_mm(x8,w8.t(),scale_a=a,scale_b=sb,out_dtype=torch.float16)
        if bias is not None: o=o+bias
        return o.reshape(*shp[:-1],o.shape[-1])
    return fwd
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
wt=datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="test"); text="\n\n".join([t for t in wt["text"] if t.strip()])
ids=tok(text,return_tensors=None)["input_ids"]; windows=[torch.tensor([ids[i*WIN:(i+1)*WIN]],device="cuda") for i in range(NWIN)]
ABCD=[tok(" "+L,add_special_tokens=False)["input_ids"][-1] for L in ["A","B","C","D"]]
mmlu=[]
for subj in MMLU_SUBJECTS:
    ds=datasets.load_dataset("cais/mmlu",subj,split="test")
    for r in list(ds)[:MMLU_PER]: mmlu.append(((r["question"].strip()+"\n"+"".join(f"{chr(65+i)}. {c}\n" for i,c in enumerate(r["choices"]))+"Answer:"),r["answer"]))
def ppl():
    nll=0.0;n=0
    with torch.no_grad():
        for w in windows:
            lsm=torch.log_softmax(m(w).logits[0].float(),-1); tgt=w[0,1:]
            nll+=float(-lsm[:-1].gather(1,tgt.unsqueeze(1)).sum()); n+=tgt.numel()
    return math.exp(nll/n)
def mmlu_acc():
    c=0
    with torch.no_grad():
        for p,ans in mmlu:
            e=tok(p,return_tensors="pt").to("cuda"); lg=m(**e).logits[0,-1]
            if int(torch.tensor([float(lg[t]) for t in ABCD]).argmax())==ans: c+=1
    return 100.0*c/len(mmlu)
def linears(pred):  # pred(name,layer_idx)->bool : engage FP8?
    out=[]
    for n,mod in m.named_modules():
        if not isinstance(mod,torch.nn.Linear) or "lm_head" in n: continue
        li=int(n.split(".layers.")[1].split(".")[0]) if ".layers." in n else -1
        if pred(n,li): out.append((n,mod))
    return out
def apply(sel):
    o={id(x):x.forward for _,x in sel}
    for _,x in sel: x.forward=make_fwd(x)
    return o
def restore(sel,o):
    for _,x in sel: x.forward=o[id(x)]
def cov(sel):  # GEMM-time fraction covered
    w=0.0
    for n,_ in sel:
        if "mlp.gate" in n or "mlp.up" in n or "mlp.down" in n: w+=0.27/32
        elif "q_proj" in n or "o_proj" in n: w+=0.076/32
        elif "k_proj" in n or "v_proj" in n: w+=0.019/32
    return w
ppl16=ppl(); print(f"[fp16] PPL={ppl16:.4f}",flush=True)
# candidate decline-sets (drop these from FP8), increasing coverage cost
CONFIGS=[
 ("all_fp8", lambda n,li: True),
 ("drop_L0", lambda n,li: li!=0),
 ("drop_L0_1", lambda n,li: li not in (0,1)),
 ("drop_L0_3", lambda n,li: li not in (0,1,2,3)),
 ("drop_downproj", lambda n,li: "mlp.down" not in n),
 ("drop_L0_and_downproj", lambda n,li: li!=0 and "mlp.down" not in n),
]
results=[]
for name,pred in CONFIGS:
    sel=linears(pred); o=apply(sel); p=ppl(); restore(sel,o)
    pd=100.0*(p-ppl16)/ppl16; c=cov(sel)
    # MFU at coverage (harmonic blend: fp8 rate 0.983, fp16 rate 0.64, on the fat B=8 S=2048 shape)
    mfu=1.0/((c/0.983)+((1-c)/0.64))*100
    results.append({"config":name,"n_linears":len(sel),"coverage_pct":round(100*c,1),"ppl":round(p,4),"ppl_delta_pct":round(pd,4),"within_ppl_bar":pd<=0.3,"proj_MFU_vs989_at_cov":round(mfu,1)})
    print(f"[{name}] cov={100*c:.1f}% PPL_delta={pd:+.4f}% ({'<=0.3 PASS' if pd<=0.3 else 'FAIL'}) projMFU@cov={mfu:.1f}%",flush=True)
# choose highest-coverage within-bar config; MMLU on it
within=[r for r in results if r["within_ppl_bar"]]
chosen=max(within,key=lambda r:r["coverage_pct"]) if within else None
mmlu_chosen=None
if chosen:
    pred=dict(CONFIGS)[chosen["config"]]; sel=linears(pred); o=apply(sel); mmlu_chosen=mmlu_acc(); restore(sel,o)
    print(f"[chosen {chosen['config']}] MMLU={mmlu_chosen:.2f}%",flush=True)
mmlu16=None
out={"fp16_ppl":ppl16,"configs":results,"chosen":chosen,"chosen_mmlu":mmlu_chosen,
     "note":"projMFU = harmonic blend of measured all-fp8 vLLM MFU 98.3% and fp16 64% at the FP8 coverage; >=85% if coverage high enough"}
json.dump(out,open("/home/ubuntu/d9_perlayer_backoff_result.json","w"),indent=2)
print(f"\nCHOSEN within-bar config: {chosen['config'] if chosen else 'NONE'} -> coverage {chosen['coverage_pct'] if chosen else 0}% PPL {chosen['ppl_delta_pct'] if chosen else '-'}% projMFU {chosen['proj_MFU_vs989_at_cov'] if chosen else '-'}%",flush=True)
print("WROTE d9_perlayer_backoff_result.json",flush=True)
