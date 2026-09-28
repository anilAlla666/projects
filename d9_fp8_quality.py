#!/usr/bin/env python3
# D.9 FP8 actuator — CIPHER-engaged quality (transparent, unmodified model). LOCKED eval sets
# (D9_BUILD_PREREG.md): WikiText-2 50x2048, MMLU 500q, true KL. Run under CUDA_INJECTION64_PATH:
#   CIPHER_FP8 on  -> tag 'fp8' (engaged, per-tensor scalar FP8 at the cublasGemmEx intercept)
#   CIPHER_FP8 unset -> tag 'ref' (CIPHER-off bf16 reference; the app's own output)
# Saves PPL/MMLU + first-K window logprobs; offline script computes PPL/MMLU delta + KL.
import os, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
M="/home/ubuntu/models/Mistral-7B-v0.1"
WIN=2048; NWIN=50; KLW=4
MMLU_SUBJ=["abstract_algebra","anatomy","astronomy","college_computer_science","high_school_mathematics"]; MMLU_PER=100
tag="fp8" if os.environ.get("CIPHER_FP8") else "ref"
tok=AutoTokenizer.from_pretrained(M)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
wt=datasets.load_dataset("wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join([t for t in wt["text"] if t.strip()])
ids=tok(text,return_tensors=None)["input_ids"]
windows=[torch.tensor([ids[i*WIN:(i+1)*WIN]],device="cuda") for i in range(NWIN)]
def fmt(q,ch):
    s=q.strip()+"\n"
    for i,c in enumerate(ch): s+=f"{chr(65+i)}. {c}\n"
    return s+"Answer:"
mmlu=[]
for subj in MMLU_SUBJ:
    ds=datasets.load_dataset("cais/mmlu",subj,split="test")
    for r in list(ds)[:MMLU_PER]: mmlu.append((fmt(r["question"],r["choices"]),r["answer"]))
ABCD=[tok(" "+L,add_special_tokens=False)["input_ids"][-1] for L in ["A","B","C","D"]]
nll=0.0; ntok=0; kl_store=[]
with torch.no_grad():
    for wi,w in enumerate(windows):
        lg=m(w).logits[0]
        lsm=torch.log_softmax(lg.float(),-1)
        tgt=w[0,1:]; lp=lsm[:-1].gather(1,tgt.unsqueeze(1)).squeeze(1)
        nll+=float(-lp.sum()); ntok+=tgt.numel()
        if wi<KLW: kl_store.append(lsm[:-1].half().cpu())
    correct=0
    for prompt,ans in mmlu:
        e=tok(prompt,return_tensors="pt").to("cuda")
        lg=m(**e).logits[0,-1]
        sub=torch.tensor([float(lg[t]) for t in ABCD])
        if int(sub.argmax())==ans: correct+=1
ppl=math.exp(nll/ntok); acc=100.0*correct/len(mmlu)
torch.save(kl_store, f"/home/ubuntu/d9_q_{tag}_logprobs.pt")
json.dump({"tag":tag,"ppl":ppl,"mmlu":acc,"ppl_tokens":NWIN*WIN,"mmlu_n":len(mmlu)},
          open(f"/home/ubuntu/d9_fp8_quality_{tag}.json","w"),indent=2)
print(f"[{tag}] PPL={ppl:.4f} MMLU={acc:.2f}% (tokens={NWIN*WIN} mmlu={len(mmlu)})",flush=True)
print(f"WROTE d9_fp8_quality_{tag}.json + logprobs",flush=True)
