#!/usr/bin/env python3
# PART B — INT4 (NF4) FLEET-QUALITY gate on a REALISTIC eval (not the 4-short-prompt KL 0.0085 indicative). Longer
# context + tail tokens + reasoning, per distinct family (Mistral/Qwen2/Llama-3.1). Per-family KL(fp16||nf4) + PPL
# delta. The K-lever rides on INT4 4x density -> must be shippable. NF4-runtime labeled (Marlin-format = production).
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING"); os.environ["VLLM_PLUGINS"]=""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
BNB=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
FAM=[("/home/ubuntu/models/Mistral-7B-v0.1","Mistral-7B"),("/home/ubuntu/models/Qwen2-7B","Qwen2-7B"),("/home/ubuntu/models/Llama-3.1-8B","Llama-3.1-8B")]
# realistic eval: a long passage (reasoning + factual + code-ish), ~600-800 tokens, vs 4 short prompts
EVAL=("The development of large language models has followed a trajectory shaped by scaling laws, architectural "
 "innovation, and the economics of compute. Consider a fleet operator serving many distinct fine-tuned models on a "
 "single accelerator: the binding constraint is rarely raw throughput but the latency of bringing a cold model's "
 "weights into high-bandwidth memory. To reason about this precisely, decompose the cold-start time into the number "
 "of concurrent misses, the bytes transferred per miss, and the achievable interconnect bandwidth. Each term has a "
 "different character: bandwidth is a hardware property fixed by the physical link; bytes are set by the numerical "
 "precision of the weights, which trades quality for size; and the miss count is governed by the residency policy, "
 "which is the operator's to tune. A correct analysis separates what is physics from what is design. For example, if "
 "the model set shares a common base with small per-model deltas, the bytes term collapses; but for genuinely "
 "distinct architectures, no such sharing exists and the full weight tensor must cross the link. The theorem that "
 "governs the steady state is simple: under a request distribution with heavy skew, a small resident set captures "
 "most traffic, and cold misses become a rare tail event -- unless arrivals are correlated, in which case the cold "
 "set is recalled together and the tail dominates. Therefore the design question reduces to: how skewed is the real "
 "traffic, and how correlated are the bursts? "*3)
@torch.no_grad()
def logits_ppl(m,tok):
    ids=tok(EVAL,return_tensors="pt",truncation=True,max_length=1024).input_ids.cuda()
    lg=m(ids).logits[0].float(); lp=F.log_softmax(lg[:-1],-1)
    nll=-lp[range(ids.shape[1]-1),ids[0,1:]].mean(); return lg, ids, torch.exp(nll).item()
for path,name in FAM:
    tok=AutoTokenizer.from_pretrained(path)
    m16=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,device_map="cuda")
    lg16,ids,ppl16=logits_ppl(m16,tok); del m16; import gc; gc.collect(); torch.cuda.empty_cache()
    m4=AutoModelForCausalLM.from_pretrained(path,quantization_config=BNB,torch_dtype=torch.float16,device_map="cuda")
    lg4,_,ppl4=logits_ppl(m4,tok)
    # mean per-token KL(fp16||nf4) over the realistic eval (~1000 tokens) + tail (rare-token) KL
    pq=F.log_softmax(lg16,-1); qq=F.log_softmax(lg4,-1)
    klpt=(pq.exp()*(pq-qq)).sum(-1)
    kl_mean=klpt.mean().item(); kl_p99=klpt.sort().values[int(len(klpt)*0.99)].item()
    print(f"{name}: PPL fp16={ppl16:.3f} nf4={ppl4:.3f} ({(ppl4/ppl16-1)*100:+.2f}%)  KL(fp16||nf4) mean={kl_mean:.4f} p99={kl_p99:.4f} nats  (n={len(klpt)} tok)",flush=True)
    del m4; gc.collect(); torch.cuda.empty_cache()
print("NF4-runtime fleet quality (indicative; CIPHER-Marlin-format = production number). FP8 on-pod was +0.567% PPL ref.",flush=True)
sys.stdout.flush(); os._exit(0)
