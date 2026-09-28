#!/usr/bin/env python3
# Gate-critical probe before the multiplexer build:
# (A) CORRECTNESS: does batched generate(B=N) give per-agent BIT-IDENTICAL output vs solo? (batching must not
#     perturb any agent -- the G-O8 per-agent KL=0 gate). Test same-prompt-xN and different-prompts(padded).
# (B) CAPACITY: real batched-DECODE tok/s vs B WITH KV at realistic context (256-tok prompt + 64 decode) -- the
#     number M2's prefill (S=8, 13.77x) stood in for. Decode+KV scales flatter and KV competes with weight residency.
import os, sys, time, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
P="/home/ubuntu/models_int4/Mistral-7B-v0.1"
from transformers import AutoModelForCausalLM, AutoTokenizer
m=AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.float16, device_map="cuda")
tok=AutoTokenizer.from_pretrained("/home/ubuntu/models/Mistral-7B-v0.1")
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"   # left-pad for correct decode alignment
GB=1<<30
def gen(input_ids, attn, n=32):
    with torch.no_grad():
        return m.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=n, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)

print("=== (A) per-agent bit-identical under batching ===",flush=True)
# A1: same prompt x B -> every row must equal the solo run
pa="The history of artificial intelligence began"
e=tok([pa], return_tensors="pt", padding=True); ia,aa=e.input_ids.cuda(),e.attention_mask.cuda()
solo=gen(ia,aa,32)[0]
B=4
eb=tok([pa]*B, return_tensors="pt", padding=True); ib,ab=eb.input_ids.cuda(),eb.attention_mask.cuda()
batched=gen(ib,ab,32)
a1=all(torch.equal(batched[i], solo) for i in range(B))
print(f"  A1 same-prompt x{B}: every batched row == solo -> {a1}",flush=True)
# A2: different prompts (padded) -> each sequence's output must equal its own solo
prompts=["The capital of France is","Once upon a time in a distant","def fibonacci(n):","The theory of relativity states"]
solos=[]
for p in prompts:
    e=tok([p],return_tensors="pt",padding=True); solos.append(gen(e.input_ids.cuda(),e.attention_mask.cuda(),32)[0])
eb=tok(prompts,return_tensors="pt",padding=True); ib,ab=eb.input_ids.cuda(),eb.attention_mask.cuda()
bat=gen(ib,ab,32)
# left-pad: each row's last (len(prompt)+32) tokens are the meaningful ones; compare the generated tail
def tail(x,k): return x[-k:]
a2_each=[]
for i,p in enumerate(prompts):
    plen=tok([p],return_tensors="pt").input_ids.shape[1]
    so=solos[i]; ba=bat[i]
    # compare the 32 generated tokens (the tail)
    a2_each.append(torch.equal(tail(so,32), tail(ba,32)))
a2=all(a2_each)
print(f"  A2 different-prompts(padded) batched B={len(prompts)}: each generated tail == solo -> {a2}  ({a2_each})",flush=True)

print("=== (B) batched DECODE tok/s vs B, WITH KV at realistic context (256-tok prompt + 64 decode) ===",flush=True)
ctx=256; dec=64
base_prompt=("data "*ctx)
def decode_tps(B):
    e=tok([base_prompt]*B, return_tensors="pt", padding=True, truncation=True, max_length=ctx)
    ii,am=e.input_ids.cuda(),e.attention_mask.cuda()
    torch.cuda.synchronize(); t=time.time()
    with torch.no_grad():
        out=m.generate(input_ids=ii, attention_mask=am, max_new_tokens=dec, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt=time.time()-t
    return B*dec/dt, dt, torch.cuda.max_memory_allocated()/GB
base=None
for B in [1,2,4,8,15]:
    torch.cuda.reset_peak_memory_stats()
    tps,dt,peak=decode_tps(B)
    if base is None: base=tps
    print(f"  B={B:2d}: {tps:.0f} decode-tok/s  ({tps/base:.2f}x vs B=1)  wall={dt*1000:.0f}ms  peakHBM={peak:.1f}GiB",flush=True)
print("  (decode+KV lift vs the M2 prefill 13.77x at B=15; peakHBM shows KV competing with the 4-bit weight residency)",flush=True)
sys.stdout.flush(); os._exit(0)
