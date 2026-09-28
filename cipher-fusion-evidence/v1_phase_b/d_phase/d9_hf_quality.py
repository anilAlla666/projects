#!/usr/bin/env python3
# D.9 BUILD — PER-CHANNEL FP8 quality on the HF forward (pre-registered method, env-forced path:
# llm-compressor breaks vLLM in-container). All-fp8 = per-CHANNEL weight + per-TOKEN activation
# rowwise FP8 E4M3 on ALL linears (q/k/v/o/gate/up/down). TRUE PPL + TRUE full-vocab KL + MMLU.
# LOCKED bar (D9_BUILD_PREREG.md): PPL<=0.3% HARD, MMLU<=0.5% HARD, KL<=0.01 nats DIAGNOSTIC.
import os, sys, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
WIN=2048; NWIN=50
MMLU_SUBJECTS=["abstract_algebra","anatomy","astronomy","college_computer_science","high_school_mathematics"]; MMLU_PER=100

def make_rowwise_fp8_fwd(lin):
    K=lin.in_features; w=lin.weight.data
    wscale=(w.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32)   # per-channel
    w8=(w/wscale).to(torch.float8_e4m3fn); bias=lin.bias; sb=wscale.t().contiguous()
    def fwd(x):
        shp=x.shape; x2=x.reshape(-1,K).contiguous()
        ascale=(x2.abs().amax(dim=1,keepdim=True)/448.0).clamp(min=1e-12).to(torch.float32) # per-token
        x8=(x2/ascale).to(torch.float8_e4m3fn)
        try: o=torch._scaled_mm(x8, w8.t(), scale_a=ascale, scale_b=sb, out_dtype=torch.float16, use_fast_accum=True)
        except Exception: o=torch._scaled_mm(x8, w8.t(), scale_a=ascale, scale_b=sb, out_dtype=torch.float16)
        if bias is not None: o=o+bias
        return o.reshape(*shp[:-1], o.shape[-1])
    return fwd

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).cuda().eval()
# ---- eval sets ----
wt=datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join([t for t in wt["text"] if t.strip()])
ids=tok(text,return_tensors=None)["input_ids"]
windows=[torch.tensor([ids[i*WIN:(i+1)*WIN]],device="cuda") for i in range(NWIN)]
def fmt(q,ch):
    s=q.strip()+"\n"
    for i,c in enumerate(ch): s+=f"{chr(65+i)}. {c}\n"
    return s+"Answer:"
mmlu=[]
for subj in MMLU_SUBJECTS:
    ds=datasets.load_dataset("cais/mmlu",subj,split="test")
    for r in list(ds)[:MMLU_PER]: mmlu.append((fmt(r["question"],r["choices"]),r["answer"]))
ABCD=[tok(" "+L,add_special_tokens=False)["input_ids"][-1] for L in ["A","B","C","D"]]

def run():  # returns (ppl, mmlu_acc, per_window_logits_list_cpu)  -- logits kept for KL
    nll=0.0;ntok=0; logits_store=[]
    with torch.no_grad():
        for w in windows:
            lg=m(w).logits[0]                      # [WIN,V] fp32-ish
            lsm=torch.log_softmax(lg.float(),-1)
            tgt=w[0,1:]; lp=lsm[:-1].gather(1,tgt.unsqueeze(1)).squeeze(1)
            nll+=float(-lp.sum()); ntok+=tgt.numel()
            if len(logits_store)<10: logits_store.append(lsm[:-1].half().cpu())  # KL on first 10 windows (mem bound)
        # MMLU
        correct=0
        for prompt,ans in mmlu:
            e=tok(prompt,return_tensors="pt").to("cuda")
            lg=m(**e).logits[0,-1]
            sub=torch.tensor([float(lg[t]) for t in ABCD])
            if int(sub.argmax())==ans: correct+=1
    return math.exp(nll/ntok), 100.0*correct/len(mmlu), logits_store

ppl16,acc16,lsm16=run()
print(f"[fp16] PPL={ppl16:.4f} MMLU={acc16:.2f}%", flush=True)
# apply per-channel FP8 to ALL linears
sel=[(n,mod) for n,mod in m.named_modules() if isinstance(mod,torch.nn.Linear) and "lm_head" not in n]
orig={id(mod):mod.forward for _,mod in sel}
for _,mod in sel: mod.forward=make_rowwise_fp8_fwd(mod)
ppl8,acc8,lsm8=run()
for _,mod in sel: mod.forward=orig[id(mod)]
print(f"[fp8 per-channel, {len(sel)} linears] PPL={ppl8:.4f} MMLU={acc8:.2f}%", flush=True)
# TRUE full-vocab KL(p_fp16 || q_fp8), mean per-token
kl=0.0;n=0
for a,b in zip(lsm16,lsm8):
    p=a.float().exp(); kl+=float((p*(a.float()-b.float())).sum(-1).sum()); n+=a.shape[0]
kl/=n
ppl_delta=100.0*(ppl8-ppl16)/ppl16; mmlu_delta=acc8-acc16
ppl_pass=ppl_delta<=0.3; mmlu_pass=abs(mmlu_delta)<=0.5; within=ppl_pass and mmlu_pass
out={"method":"per_channel_W+per_token_A rowwise FP8 E4M3 (HF forward, all linears)","n_linears":len(sel),
     "fp16":{"ppl":ppl16,"mmlu":acc16},"fp8":{"ppl":ppl8,"mmlu":acc8},
     "ppl_delta_pct":round(ppl_delta,4),"mmlu_delta_abs":round(mmlu_delta,3),"kl_true_nats":round(kl,5),
     "ppl_pass(<=0.3%)":ppl_pass,"mmlu_pass(<=0.5%)":mmlu_pass,"kl_diag(<=0.01)":kl<=0.01,"WITHIN_BAR":within,
     "ppl_tokens":NWIN*WIN,"mmlu_n":len(mmlu)}
json.dump(out,open("/home/ubuntu/d9_hf_quality_result.json","w"),indent=2)
print(f"\n=== PER-CHANNEL QUALITY GATE ===\nPPL delta={ppl_delta:+.4f}% (<=0.3 {'PASS' if ppl_pass else 'FAIL'}) | MMLU delta={mmlu_delta:+.3f}pp (<=0.5 {'PASS' if mmlu_pass else 'FAIL'}) | TRUE KL={kl:.5f} nats (<=0.01 diag {'ok' if kl<=0.01 else 'over'})", flush=True)
print(f"WITHIN BAR (PPL+MMLU): {within}", flush=True); print("WROTE d9_hf_quality_result.json", flush=True)
