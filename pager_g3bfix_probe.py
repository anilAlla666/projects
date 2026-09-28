#!/usr/bin/env python3
# INCREMENT-3b-fix PROBE: does a NEVER-EVICTED model's persistent captured graph survive OTHER models' churn?
# A=Qwen2 kept resident/never-evicted; B,C churn the pager (restore->serve->evict) around it. Serve A interleaved,
# NO re-capture. If A stays KL=0 -> others' churn is harmless; the tax is PER-SWAP (only own evict/restore invalidates
# the graph -> mechanism = own-weight cuMemMap remap invalidates the captured graph's weight reference). If A
# corrupts -> others' churn corrupts resident graphs (per-serve, deeper). CIPHER_RT_DISABLE_AUTO_INIT=1.
import os, sys, torch
os.environ["HF_DEACTIVATE_ASYNC_LOAD"]="1"; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
torch.manual_seed(0)
from cipher_engine import CipherPager, PagerGraphModel
N=48; PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"
pager=CipherPager()
# A stays resident the whole time (never evicted)
A=PagerGraphModel(pager,"/home/ubuntu/models/Qwen2-7B",0xC2).load(); A.capture(PROMPT,N); soloA=A.eager_burst(N)
print("[setup] A=Qwen2 captured, RESIDENT (never evicted)", flush=True)
# B,C will churn around A
B=PagerGraphModel(pager,"/home/ubuntu/models/Llama-3.1-8B",0xC3).load(); B.capture(PROMPT,N); soloB=B.eager_burst(N); B.evict()
C=PagerGraphModel(pager,"/home/ubuntu/models/TinyLlama-1.1B",0xC5).load(); C.capture(PROMPT,N); soloC=C.eager_burst(N); C.evict()
print("[setup] B=Llama, C=Tiny captured+evicted. A still resident.", flush=True)

def serveA(tag):
    try:
        g=A.serve_burst(N); mt=sum(a==b for a,b in zip(g,soloA))
        print(f"  serve A [{tag}]: {mt}/{N} {'KL=0' if mt==N else 'DIVERGES@'+str(next((i for i,(x,y) in enumerate(zip(g,soloA)) if x!=y),-1))}", flush=True); return mt==N
    except Exception as ex:
        print(f"  serve A [{tag}]: FAULT {type(ex).__name__}", flush=True); return False
VMM_ONLY=os.environ.get("VMM_ONLY","0")=="1"   # churn with ONLY pager VMM ops (no graph replay) to isolate the trigger
def churn(M,solo,nm):
    M.restore()                                 # page_in = cuMemCreate+cuMemMap (VMM)
    if not VMM_ONLY:
        try: M.serve_burst(N)                   # graph replay
        except Exception: pass
    M.evict()                                   # page_out = cuMemUnmap+cuMemRelease (VMM)

allok=True
allok &= serveA("baseline, no churn yet")
for rnd in range(5):
    churn(B,soloB,"B"); allok &= serveA(f"after B churn rnd{rnd}")
    churn(C,soloC,"C"); allok &= serveA(f"after C churn rnd{rnd}")
print(f"\n[VERDICT] A never-evicted under {'5'} rounds of B+C churn: {'KL=0 SURVIVED -> others-churn HARMLESS; tax is PER-SWAP (own evict/restore only); mechanism = own-weight remap invalidates the captured graph' if allok else 'A CORRUPTED -> others-churn corrupts resident graphs (per-serve, deeper mechanism)'}", flush=True)
print(f"[GATE probe17] {'PASS-PER-SWAP' if allok else 'FAIL-PER-SERVE'}", flush=True)
sys.stdout.flush(); os._exit(0)
