#!/usr/bin/env python3
# G-O3 STEP 3 -- quality gate. PPL (and KL/logit-delta proxy) FP8-substitution vs bf16 baseline on held-out text.
# Gate: <=0.3% PPL (0.37% amended). FP8 per-tensor has a KNOWN quality delta (prior +0.567% fails 0.37%; per-channel
# +0.37% passes but has no fused cuBLASLt path here) -- the gate is the bound, not bit-exact. n>64 so FP8 engages on
# the held-out forward. Warm past FP8_STABILITY first. CIPHER_FP8 / CIPHER_FP8_SHARE_ACT via env.
import os, sys, ctypes, math
import torch
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL=os.environ.get("GO3_MODEL","/home/ubuntu/models/Mistral-7B-v0.1")
TAG=os.environ.get("GO3_TAG","run")
lib=ctypes.CDLL(SO)
for s in ("cipher_rt_fp8_calls_handled","cipher_rt_fp8_weights_quantized","cipher_rt_fp8_is_active"):
    getattr(lib,s).restype=ctypes.c_ulong
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda").eval()
# held-out: REAL wikitext-2 test (standard PPL benchmark, non-repetitive diverse prose).
from datasets import load_dataset
ds=load_dataset("wikitext","wikitext-2-raw-v1",split="test")
text="\n\n".join(t for t in ds["text"] if t.strip())
ids=tok(text,return_tensors="pt").input_ids.cuda()[:, :2048]
WARM=os.environ.get("GO3_WARM_PROMPT","The history of artificial intelligence and high performance computing. ")
wids=tok(WARM,return_tensors="pt").input_ids; wids=wids.repeat(1,(256//wids.shape[1])+1)[:, :256].cuda()
@torch.no_grad()
def ppl_and_logits():
    out=m(ids).logits  # (1,S,V)
    lp=torch.log_softmax(out[0,:-1].float(),dim=-1)
    nll=-lp[torch.arange(ids.shape[1]-1),ids[0,1:]].mean().item()
    return math.exp(nll), out[0,:-1].float()
# warm up so FP8 substitution is live (stability + prequant) -- only matters when CIPHER_FP8=1
with torch.no_grad():
    for _ in range(4): _=m(wids).logits
    torch.cuda.synchronize()
ppl,logits=ppl_and_logits()
torch.cuda.synchronize()
print(f"[go3-s3 {TAG}] fp8_active={lib.cipher_rt_fp8_is_active()} handled={lib.cipher_rt_fp8_calls_handled()} wq={lib.cipher_rt_fp8_weights_quantized()} PPL={ppl:.5f}",flush=True)
# stash logits for KL vs baseline (saved to file keyed by tag)
torch.save(logits.cpu(), f"/tmp/go3_logits_{TAG}.pt")
print(f"GO3_PPL {TAG} ppl={ppl:.6f} handled={lib.cipher_rt_fp8_calls_handled()}",flush=True)
sys.stdout.flush(); os._exit(0)
