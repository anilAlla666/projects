#!/usr/bin/env python3
# D.9 BUILD step 1 — ALL-FP8 QUALITY (measured FIRST, per pre-reg D9_BUILD_PREREG.md).
# Single vLLM engine for fp16-ref and fp8. WikiText-2 PPL (prompt_logprobs), MMLU 500q
# (letter-logprob), output-KL (top-K diagnostic). LOCKED bar: PPL<=0.3% HARD, MMLU<=0.5% HARD,
# KL<=0.01 nats DIAGNOSTIC. Reports vLLM's actual fp8 scheme. Run: --quant fp16 then --quant fp8.
import os, sys, json, math
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
QUANT=sys.argv[1] if len(sys.argv)>1 else "fp16"
FP8_PATH=sys.argv[2] if len(sys.argv)>2 else None   # per-channel FP8_DYNAMIC checkpoint (pre-reg method)
WIN=2048; NWIN=50
MMLU_SUBJECTS=["abstract_algebra","anatomy","astronomy","college_computer_science","high_school_mathematics"]
MMLU_PER=100

tok=AutoTokenizer.from_pretrained(MODEL)
import datasets
# ---- PPL windows (WikiText-2-raw test, fixed 50x2048) ----
wt=datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join([t for t in wt["text"] if t.strip()])
allids=tok(text, return_tensors=None)["input_ids"]
windows=[allids[i*WIN:(i+1)*WIN] for i in range(NWIN)]
assert all(len(w)==WIN for w in windows), "not enough tokens"
ntok=NWIN*WIN
# ---- MMLU 500q ----
def fmt(q,choices):
    s=q.strip()+"\n"
    for i,c in enumerate(choices): s+=f"{chr(65+i)}. {c}\n"
    return s+"Answer:"
mmlu=[]
for subj in MMLU_SUBJECTS:
    ds=datasets.load_dataset("cais/mmlu",subj,split="test")
    for r in list(ds)[:MMLU_PER]:
        mmlu.append((fmt(r["question"],r["choices"]), r["answer"]))  # answer=0..3
print(f"[{QUANT}] PPL tokens={ntok} | MMLU q={len(mmlu)} | scheme=reporting...", flush=True)

# fp8 = pre-registered PER-CHANNEL weight + PER-TOKEN dynamic activation, via an llm-compressor
# FP8_DYNAMIC checkpoint (vLLM auto-detects compressed-tensors). Online quantization="fp8" is
# per-TENSOR (cruder) and was measured separately (per-tensor result kept in the report).
mdl = FP8_PATH if (QUANT=="fp8" and FP8_PATH) else MODEL
kw=dict(model=mdl, dtype="float16", gpu_memory_utilization=0.9, enforce_eager=False,
        max_num_seqs=16, max_model_len=4096, enable_prefix_caching=False, disable_log_stats=True)
if QUANT=="fp8" and not FP8_PATH: kw["quantization"]="fp8"   # fallback: online per-tensor
llm=LLM(**kw)
scheme="fp16" if QUANT=="fp16" else ("per_channel_W+per_token_A(FP8_DYNAMIC ckpt)" if FP8_PATH else "per_tensor_W+per_token_A(online)")

# ---- PPL + per-token logprob capture ----
sp_pl=SamplingParams(max_tokens=1, prompt_logprobs=20, temperature=0.0)
outs=llm.generate([{"prompt_token_ids":w} for w in windows], sp_pl, use_tqdm=False)
nll_sum=0.0; nll_n=0
pos_lp=[]   # per (win,pos): {"actual": logprob_of_actual, "top": {tid: lp}}
for wi,o in enumerate(outs):
    pls=o.prompt_logprobs  # list len WIN; [0] is None
    w=windows[wi]
    for i in range(1,WIN):
        d=pls[i]
        if d is None: continue
        actual=w[i]
        lp=d.get(actual)
        lp=lp.logprob if lp is not None else min(v.logprob for v in d.values())-2.0
        nll_sum+= -lp; nll_n+=1
        pos_lp.append({"a":lp,"t":{int(t):float(v.logprob) for t,v in d.items()}})
ppl=math.exp(nll_sum/nll_n)
# ---- MMLU letter-logprob ----
sp_m=SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)
mo=llm.generate([m[0] for m in mmlu], sp_m, use_tqdm=False)
correct=0
LET={" A":0,"A":0," B":1,"B":1," C":2,"C":2," D":3,"D":3}
for (prompt,ans),o in zip(mmlu,mo):
    lps=o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
    best=None;bestlp=-1e9
    for tid,lo in lps.items():
        dt=lo.decoded_token.strip() if hasattr(lo,'decoded_token') and lo.decoded_token else ""
        if dt in ("A","B","C","D"):
            if lo.logprob>bestlp: bestlp=lo.logprob; best="ABCD".index(dt)
    if best==ans: correct+=1
acc=100.0*correct/len(mmlu)
res={"quant":QUANT,"scheme":scheme,"ppl":ppl,"ppl_tokens":ntok,"mmlu_acc":acc,"mmlu_n":len(mmlu)}
print(f"[{QUANT}] scheme={scheme} | PPL={ppl:.4f} | MMLU={acc:.2f}% ({correct}/{len(mmlu)})", flush=True)
if QUANT=="fp16":
    json.dump({"res":res,"pos_lp":pos_lp}, open("/home/ubuntu/d9q_fp16.json","w"))
    print("WROTE d9q_fp16.json (ref)", flush=True)
else:
    ref=json.load(open("/home/ubuntu/d9q_fp16.json")); rres=ref["res"]; rpos=ref["pos_lp"]
    # KL diagnostic (top-K, p_fp16 || q_fp8) over PPL positions
    kl_sum=0.0;kl_n=0;dact=0.0
    for r,c in zip(rpos,pos_lp):
        p={int(t):lp for t,lp in r["t"].items()}; q={int(t):lp for t,lp in c["t"].items()}
        qfloor=min(q.values())-2.0 if q else -30.0
        Z=sum(math.exp(lp) for lp in p.values())  # top-K mass of fp16
        for t,lp16 in p.items():
            p16=math.exp(lp16)/Z if Z>0 else 0.0
            lq=q.get(t,qfloor)
            kl_sum+=p16*(lp16-lq); kl_n+=1
        dact+=abs(r["a"]-c["a"])
    kl=kl_sum/max(1,len(rpos)); dact/=max(1,len(rpos))
    ppl_delta=100.0*(res["ppl"]-rres["ppl"])/rres["ppl"]
    mmlu_delta=res["mmlu_acc"]-rres["mmlu_acc"]
    ppl_pass=ppl_delta<=0.3; mmlu_pass=abs(mmlu_delta)<=0.5
    within_bar=ppl_pass and mmlu_pass   # PPL+MMLU bind; KL diagnostic
    out={"fp16":rres,"fp8":res,"ppl_delta_pct":round(ppl_delta,4),"mmlu_delta_abs":round(mmlu_delta,3),
         "kl_topk_diag_nats":round(kl,5),"actual_token_logprob_delta":round(dact,5),
         "ppl_pass(<=0.3%)":ppl_pass,"mmlu_pass(<=0.5%)":mmlu_pass,"WITHIN_BAR":within_bar,
         "kl_diag_note":"top-K(20) approx, non-binding"}
    json.dump(out, open("/home/ubuntu/d9_build_quality_result.json","w"), indent=2)
    print(f"\n=== QUALITY GATE ===\nfp16 PPL={rres['ppl']:.4f} MMLU={rres['mmlu_acc']:.2f}% | fp8 PPL={res['ppl']:.4f} MMLU={res['mmlu_acc']:.2f}%", flush=True)
    print(f"PPL delta={ppl_delta:+.4f}% (gate<=0.3 -> {'PASS' if ppl_pass else 'FAIL'}) | MMLU delta={mmlu_delta:+.3f}pp (gate<=0.5 -> {'PASS' if mmlu_pass else 'FAIL'}) | KL_diag={kl:.5f} nats", flush=True)
    print(f"WITHIN BAR (PPL+MMLU): {within_bar}", flush=True)
    print("WROTE d9_build_quality_result.json", flush=True)
