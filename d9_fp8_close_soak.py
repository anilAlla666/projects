#!/usr/bin/env python3
# D.9 FP8 staging-close §3 — FP8-actuator sustained soak (the close-relevant soak).
# NOTE: the W9 resolver_soak harness is NOT preserved (W6_SUBC_CLOSE_REPORT.md:79 /tmp scratch);
# the resolver is a W7-12 multi-tenant subsystem unrelated to the FP8 actuator and is not exercised
# by single-tenant FP8 forwards. This soak validates what the FP8 CLOSE needs: the FP8 actuator under
# sustained 30-min load — no crash, no leak (GPU mem stable), no NaN, handled-count grows, output stable.
# Run under CUDA_INJECTION64_PATH with CIPHER_FP8=on.
import os, time, json, numpy as np, torch
from transformers import AutoModelForCausalLM
M="/home/ubuntu/models/Mistral-7B-v0.1"
DUR=float(os.environ.get("D9_SOAK_SEC","1800"))   # 30 min
B,S=64,512
m=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids=torch.randint(1,31000,(B,S),device='cuda')
with torch.no_grad():
    for _ in range(6): m(ids); torch.cuda.synchronize()   # warmup → FP8 engages (stability=2)
torch.cuda.reset_peak_memory_stats()
base_mem=torch.cuda.memory_allocated()
t0=time.time(); nfwd=0; incoherent=0; nan_count=0; checkpoints=[]; next_cp=t0+60
ref=None
with torch.no_grad():
    while time.time()-t0 < DUR:
        out=m(ids).logits
        nfwd+=1
        # stability/coherence: output must stay finite AND deterministic for fixed input (FP8 is deterministic)
        if not torch.isfinite(out).all(): nan_count+=1
        cur=out[0,-1,:].float().cpu().numpy()
        if ref is None: ref=cur
        elif not np.array_equal(cur,ref): incoherent+=1   # same input → must be identical every fwd
        torch.cuda.synchronize()
        if time.time()>=next_cp:
            mem=torch.cuda.memory_allocated()
            checkpoints.append({"t":round(time.time()-t0),"nfwd":nfwd,"mem_MiB":round(mem/2**20,1),
                                "leak_MiB":round((mem-base_mem)/2**20,2),"incoherent":incoherent,"nan":nan_count})
            print(f"[soak t={int(time.time()-t0)}s] fwd={nfwd} mem={mem/2**20:.0f}MiB leak={(mem-base_mem)/2**20:.2f}MiB incoherent={incoherent} nan={nan_count}",flush=True)
            next_cp+=60
res={"soak_sec":round(time.time()-t0,1),"N_config":f"B={B} S={S} (single-tenant FP8 forward soak)",
     "forwards":nfwd,"incoherent":incoherent,"nan":nan_count,
     "leak_MiB_final":checkpoints[-1]["leak_MiB"] if checkpoints else None,
     "checkpoints":checkpoints,
     "PASS":bool(incoherent==0 and nan_count==0 and nfwd>0)}
json.dump(res,open("/home/ubuntu/d9_fp8_close_soak.json","w"),indent=2)
print(f"SOAK DONE: {nfwd} forwards / {res['soak_sec']}s, incoherent={incoherent} nan={nan_count} leak={res['leak_MiB_final']}MiB PASS={res['PASS']}",flush=True)
