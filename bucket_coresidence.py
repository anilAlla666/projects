#!/usr/bin/env python3
# The ONE non-tautological empirical add (advisor): do K pre-captured batch-size BUCKET graphs co-reside in
# memory AND switch at ~0 cost? (the bucket+padding mechanism's physical realizability on cu13).
# Bursty membership churn = pick the nearest pre-captured bucket per active-count + replay; NO recapture.
# Decision rests on components already measured (recapture ~27ms = per-event infeasible; replay rates per bucket).
import os, time, random, torch
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
M="/home/ubuntu/models/Mistral-7B-v0.1"
BUCKETS=[8,16,32,64]
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16,device_map="cuda").eval()
base_mem=torch.cuda.memory_allocated()/2**30
print(f"model loaded: {base_mem:.1f} GiB")

def capture_bucket(B):
    ids=tok(["The history of computing spans several distinct eras, each shaped by"]*B,return_tensors="pt").input_ids.cuda()
    plen=ids.shape[1]
    c=StaticCache(config=model.config,max_batch_size=B,max_cache_len=plen+80,device="cuda",dtype=torch.float16)
    with torch.no_grad(): o=model(input_ids=ids,cache_position=torch.arange(plen,device="cuda"),past_key_values=c,use_cache=True,return_dict=True)
    iid=o.logits[:,-1:].argmax(-1); cp=torch.tensor([plen],device="cuda")
    outl=torch.empty(B,1,model.config.vocab_size,device="cuda",dtype=torch.float16)
    side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(4):
            with torch.no_grad(): o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
            outl.copy_(o.logits); iid.copy_(outl.argmax(-1)); cp+=1
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(): o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
        outl.copy_(o.logits); iid.copy_(outl.argmax(-1))
    return {"B":B,"g":g,"cp":cp,"c":c,"iid":iid,"outl":outl}

# 1) capture all buckets, hold resident
graphs={}
for B in BUCKETS:
    graphs[B]=capture_bucket(B); torch.cuda.synchronize()
    print(f"  bucket B={B:>3} captured & resident; total GPU mem={torch.cuda.memory_allocated()/2**30:.1f} GiB")
print(f"K={len(BUCKETS)} bucket graphs co-resident in {torch.cuda.memory_allocated()/2**30:.1f} GiB (of 80) — fits with room")

def bucket_for(active): return min(b for b in BUCKETS if b>=active) if active<=max(BUCKETS) else max(BUCKETS)

# 2) per-bucket replay rate (padding economics: the rate you get = the bucket you padded UP to)
print("\nper-bucket replay rate (you pay the bucket's rate regardless of how many slots are active):")
for B in BUCKETS:
    G=graphs[B]; torch.cuda.synchronize(); t0=time.time()
    for _ in range(64): G["cp"]+=1; G["g"].replay()
    torch.cuda.synchronize(); ms=(time.time()-t0)/64*1000
    print(f"  bucket B={B:>3}: {ms:5.2f} ms/step ({1000/ms:.0f} steps/s)")

# 3) FIXED vs SWITCHING (churn): switch overhead ~0?  (bursty active-count random walk over a window)
def run_window(seconds, switching):
    for b in BUCKETS: graphs[b]["cp"].fill_(20)   # in-place reset captured cp to a safe in-bounds slot (timing test; work identical)
    torch.cuda.synchronize()
    random.seed(0); active=8; steps=0; switches=0; t_end=time.time()+seconds; nextchurn=time.time()
    cur=bucket_for(active)
    while time.time()<t_end:
        if switching and time.time()>=nextchurn:
            active=random.choice([4,8,12,20,28,40,60])          # bursty membership change
            nb=bucket_for(active)
            if nb!=cur: switches+=1; cur=nb                       # pick pre-captured bucket — NO recapture
            nextchurn=time.time()+random.uniform(0.035,0.2)       # realistic 100-agent churn cadence
        G=graphs[cur]; G["g"].replay(); steps+=1   # cp fixed in-bounds; replay does identical work (timing test)
        if steps%8==0: torch.cuda.synchronize()
    torch.cuda.synchronize()
    return steps, switches
sf,_=run_window(4.0, False); print(f"\nFIXED bucket (no churn): {sf/4.0:.0f} steps/s")
ss,sw=run_window(4.0, True);  print(f"SWITCHING under bursty churn ({sw} bucket-switches): {ss/4.0:.0f} steps/s")
print(f"switch overhead vs fixed: {100*(1-(ss/4.0)/(sf/4.0)):+.1f}% (≈0 => bucket-switch is free; eager ceiling=43 steps/s)")
