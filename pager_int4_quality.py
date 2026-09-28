#!/usr/bin/env python3
# INT4 QUALITY (gate 2, lossy-bounded NOT KL=0): NF4-runtime quality vs fp16 for Mistral-7B. INDICATIVE only --
# CIPHER-Marlin-format quality is the production-relevant number; this NF4 figure settles the density-axis "INT4
# quality unmeasured" condition with an indicative delta, NOT a shipped-quality claim.
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
P="/home/ubuntu/models/Mistral-7B-v0.1"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
tok=AutoTokenizer.from_pretrained(P)
PROMPTS=["The history of science is a long and winding road that",
         "In a distant galaxy, a lone explorer discovered",
         "The fundamental theorem of calculus states that",
         "Once upon a time, in a small village nestled in the mountains,"]
def load(q):
    kw=dict(torch_dtype=torch.float16, device_map="cuda")
    if q: kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(P, **kw)
def logits(m, text):
    ids=tok(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad(): return m(ids).logits.float()[0], ids[0]
m16=load(False)
ref=[logits(m16,p) for p in PROMPTS]
ppl16=[]
for (lg,ids),p in zip(ref,PROMPTS):
    lp=F.log_softmax(lg[:-1],-1); nll=-lp[range(len(ids)-1),ids[1:]].mean(); ppl16.append(torch.exp(nll).item())
del m16; import gc; gc.collect(); torch.cuda.empty_cache()
m4=load(True)
kls=[]; ppl4=[]
for (lg16,ids),p in zip(ref,PROMPTS):
    lg4,_=logits(m4,p)
    # mean per-token KL(fp16 || 4bit) over positions
    pq=F.log_softmax(lg16,-1); qq=F.log_softmax(lg4,-1)
    kl=(pq.exp()*(pq-qq)).sum(-1).mean().item(); kls.append(kl)
    lp=F.log_softmax(lg4[:-1],-1); nll=-lp[range(len(ids)-1),ids[1:]].mean(); ppl4.append(torch.exp(nll).item())
import statistics as st
print(f"NF4-runtime QUALITY (Mistral-7B, INDICATIVE -- not the Marlin-format shipped number):")
print(f"  mean per-token KL(fp16||nf4) = {st.mean(kls):.4f} nats  (per prompt: {[round(k,3) for k in kls]})")
print(f"  perplexity fp16 = {st.mean(ppl16):.3f}  nf4 = {st.mean(ppl4):.3f}  ratio = {st.mean(ppl4)/st.mean(ppl16):.3f}x")
print(f"  -> NF4 quality is LOSSY-BOUNDED (KL>0 expected); usable-if-tolerable. Production claim needs Marlin-format eval.")
sys.stdout.flush(); os._exit(0)
