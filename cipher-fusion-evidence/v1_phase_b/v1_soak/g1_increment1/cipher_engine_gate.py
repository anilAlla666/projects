#!/usr/bin/env python3
# INCREMENT-1 CORRECTNESS GATE via the deliverable module (cipher_engine.PagerGraphModel).
# One condition per subprocess (mode argv), one capture, one monotonic burst vs eager-static (KL=0 = greedy match).
#   resident  : graph-decode over pager-resident weights == eager           (expect KL=0)
#   pagecycle : evict -> restore BEFORE any replay, then burst == eager      (expect KL=0; captured graph survives remap)
#   negctrl   : zero 2 down_proj at pager VA, then burst != eager            (expect DIVERGE; KL=0 is non-vacuous)
# Run with CIPHER_RT_DISABLE_AUTO_INIT=1 (legacy compute actuators are not capture-safe; out of the engine path).
import os, sys, torch
from cipher_engine import CipherPager, PagerGraphModel

P=sys.argv[1] if len(sys.argv)>1 else "/home/ubuntu/models/TinyLlama-1.1B"
N=int(sys.argv[2]) if len(sys.argv)>2 else 64
MODE=sys.argv[3] if len(sys.argv)>3 else "resident"
PROMPT="The history of artificial intelligence began in the 1950s, when researchers first"

eng=PagerGraphModel(CipherPager(), P).load()
es=eng.capture(PROMPT, N) or None
es=eng.eager_burst(N)          # reference computed before any page cycle / perturb (eager-before-replay is safe)

note=""; real_evict=True
if MODE=="pagecycle":
    GB=1<<30
    ckb=eng.region_cksum(); vab=eng.base_va()
    s0=eng.pg.stats(eng.rid); free0=torch.cuda.mem_get_info()[0]
    ro=eng.evict()                                   # RESIDENT -> EVICTED (must PHYSICALLY free HBM)
    s1=eng.pg.stats(eng.rid); free1=torch.cuda.mem_get_info()[0]
    ri=eng.restore()                                 # EVICTED -> RESIDENT (must re-pin HBM)
    s2=eng.pg.stats(eng.rid); free2=torch.cuda.mem_get_info()[0]
    cka=eng.region_cksum(); vaa=eng.base_va()
    # UNFAKEABLE: physical free must JUMP ~model size after evict, then DROP back after restore; state/counters must move
    freed_gb=(free1-free0)/GB; repinned_gb=(free1-free2)/GB
    real_evict = (s1.state==0) and (s1.evict_cnt>s0.evict_cnt) and (freed_gb>0.5) and (s2.state==2) and (s2.pagein_cnt>s1.pagein_cnt) and (repinned_gb>0.5)
    note=(f"evict={ro} restore={ri} VA stable={vab==vaa} bytes_restored={ckb==cka} | "
          f"state {s0.state}->{s1.state}(EVICTED)->{s2.state}(RESIDENT) evict_cnt {s0.evict_cnt}->{s1.evict_cnt} pagein_cnt {s1.pagein_cnt}->{s2.pagein_cnt} | "
          f"HBM freed_on_evict={freed_gb:.2f}GB repinned_on_restore={repinned_gb:.2f}GB REAL_EVICT={real_evict}")
elif MODE=="negctrl":
    s=eng.pg.stats(eng.rid); nz=0
    with torch.no_grad():
        for nm_,p in eng.m.named_parameters():
            if "mlp.down_proj" in nm_ and s.base_va<=p.data_ptr()<s.base_va+s.used_bytes:
                p.data.zero_(); nz+=1
                if nz>=2: break
    torch.cuda.synchronize(); note=f"zeroed {nz} down_proj at pager VA"

gs=eng.serve_burst(N)
match=sum(a==b for a,b in zip(gs,es)); fd=next((i for i,(x,y) in enumerate(zip(gs,es)) if x!=y),-1)
expect_kl0 = MODE in ("resident","pagecycle")
ok=((match==N) if expect_kl0 else (match<N)) and real_evict   # pagecycle also requires a PHYSICALLY-real evict
print(f"[RESULT mode={MODE} model={os.path.basename(P)} N={N}] serve_burst vs eager: {match}/{N} (1stdiff@{fd})  "
      f"{'KL=0' if match==N else 'DIVERGES'}  {note}", flush=True)
print(f"[GATE mode={MODE}] expect={'KL=0' if expect_kl0 else 'DIVERGE'} -> {'PASS' if ok else 'FAIL'}", flush=True)
sys.stdout.flush(); os._exit(0 if ok else 1)
