#!/usr/bin/env python3
# Single-instance, apples-to-apples: does graph replay (the reachable "GPU-resident replay" on cu13)
# beat the ~43 steps/s EAGER CPU-dispatch ceiling in the SM-pack harness's own regime (manual loop, B=8/32)?
# steps/s = decode steps per second (launch-elimination signal). NOT a multi-agent multiplex claim.
import os, time, torch
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
M="/home/ubuntu/models/Mistral-7B-v0.1"
B=int(os.environ.get("B","8")); K=int(os.environ.get("K","128"))
tok=AutoTokenizer.from_pretrained(M); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16,device_map="cuda").eval()
ids=tok(["The history of computing spans several distinct eras, each shaped by"]*B,return_tensors="pt").input_ids.cuda()
plen=ids.shape[1]

def prefill():
    c=StaticCache(config=model.config,max_batch_size=B,max_cache_len=plen+K+4,device="cuda",dtype=torch.float16)
    with torch.no_grad(): o=model(input_ids=ids,cache_position=torch.arange(plen,device="cuda"),past_key_values=c,use_cache=True,return_dict=True)
    return c, o.logits[:,-1:].argmax(-1)

# EAGER manual decode loop
c,nt=prefill(); iid=nt.clone(); cp=torch.tensor([plen],device="cuda")
for _ in range(4):  # warm
    with torch.no_grad(): o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
    iid=o.logits[:,-1:].argmax(-1); cp=cp+1
torch.cuda.synchronize()
c,nt=prefill(); iid=nt.clone(); cp=torch.tensor([plen],device="cuda"); etoks=[]
t0=time.time()
with torch.no_grad():
    for _ in range(K):
        o=model(input_ids=iid,cache_position=cp,past_key_values=c,use_cache=True,return_dict=True)
        iid=o.logits[:,-1:].argmax(-1); etoks.append(iid[:,0]); cp=cp+1
torch.cuda.synchronize(); e_dt=time.time()-t0

# GRAPH replay (capture one decode step, replay K)
c,nt=prefill(); iid=nt.clone(); cp=torch.tensor([plen],device="cuda")
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
gtoks=[]
torch.cuda.synchronize(); t0=time.time()
for _ in range(K):
    cp+=1; g.replay(); gtoks.append(iid[:,0].clone())
torch.cuda.synchronize(); g_dt=time.time()-t0

e_sps=K/e_dt; g_sps=K/g_dt
# coherence sanity: are graph-replay tokens non-garbage (not all-identical spam)?
gt=torch.stack(gtoks,1)[0].tolist(); et=torch.stack(etoks,1)[0].tolist()
print(f"B={B} K={K}")
print(f"  EAGER  steps/s={e_sps:.1f}  (vs ~43 ceiling)  sample='{tok.decode(et[:16])}'")
print(f"  GRAPH  steps/s={g_sps:.1f}  speedup={g_sps/e_sps:.2f}x  sample='{tok.decode(gt[:16])}'")
print(f"  GRAPH beats 43-steps/s ceiling: {g_sps>43}  | graph coherent(non-spam): {len(set(gt))>3}")
