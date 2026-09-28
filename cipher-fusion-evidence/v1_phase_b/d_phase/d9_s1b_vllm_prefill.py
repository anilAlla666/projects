#!/usr/bin/env python3
# D.9 §1b measurement 2 — vLLM forward PREFILL baseline (not HF-eager). CIPHER off.
# Kills the "50.9% HF-eager baseline" assumption: vLLM fuses epilogue + CUDA-graphs.
# Measures prefill MFU on the vLLM engine across the fat-shape ladder, isolating prefill
# via t(max_tokens=1) - decode_step (decode_step = t(mt=2)-t(mt=1)). MFU vs 989.
import os, sys, json, time
PEAK=989.0e12; MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
P_LINEAR=7.241e9; L,H,HD=32,32,128
import torch
from vllm import LLM, SamplingParams

def flops(B,S):
    T=B*S
    return 2.0*P_LINEAR*T + 2.0*L*B*H*(S**2)*HD   # linear + causal attn

_CALL=[0]
def fresh_ids(B,S):
    # DISTINCT prompts every call (prefix-cache disabled too) — re-running identical prompts
    # would be a KV-cache HIT, not a real prefill (gave impossible >100% MFU).
    _CALL[0]+=1; base=_CALL[0]*1_000_003
    return [{"prompt_token_ids":[ (base + i*131 + j*7) % 31000 + 1 for j in range(S)]} for i in range(B)]
def measure(llm, B, S, reps=3):
    def t_for(mt):
        sp=SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True)
        llm.generate(fresh_ids(B,S), sampling_params=sp, use_tqdm=False)  # warmup (distinct)
        torch.cuda.synchronize()
        best=1e9
        for _ in range(reps):
            t0=time.time(); llm.generate(fresh_ids(B,S), sampling_params=sp, use_tqdm=False); torch.cuda.synchronize()
            best=min(best, time.time()-t0)
        return best
    t1=t_for(1); t2=t_for(2)
    dec=max(t2-t1, 0.0); prefill=max(t1-dec, t1*0.5)   # guard
    fl=flops(B,S); tf=fl/prefill/1e12; mfu=100*tf*1e12/PEAK
    return {"B":B,"S":S,"tokens":B*S,"t_mt1_ms":round(t1*1e3,1),"decode_step_ms":round(dec*1e3,2),
            "prefill_ms":round(prefill*1e3,1),"prefill_TFLOPS":round(tf,1),"prefill_MFU_vs989":round(mfu,1)}

shapes=[(8,2048),(16,2048),(8,4096),(16,4096)]
out={"configs":[]}
for eager in [True, False]:   # True = vLLM kernels no cudagraph; False = full vLLM (cudagraph)
    try:
        llm=LLM(model=MODEL, dtype="float16", enforce_eager=eager, gpu_memory_utilization=0.9,
                max_num_seqs=16, max_num_batched_tokens=70000, enable_chunked_prefill=False,
                enable_prefix_caching=False, disable_log_stats=True)
    except Exception as e:
        out["configs"].append({"enforce_eager":eager,"llm_error":str(e)}); print(f"[vLLM eager={eager}] LLM ERR {e}",flush=True); continue
    rows=[]
    for (B,S) in shapes:
        try:
            r=measure(llm,B,S); rows.append(r)
            print(f"[vLLM eager={eager} B={B} S={S}] prefill {r['prefill_ms']}ms = {r['prefill_MFU_vs989']}% MFU(vs989)  (t_mt1={r['t_mt1_ms']}ms dec={r['decode_step_ms']}ms)",flush=True)
        except Exception as e:
            rows.append({"B":B,"S":S,"error":str(e)}); print(f"[vLLM eager={eager} B={B} S={S}] ERR {e}",flush=True)
    out["configs"].append({"enforce_eager":eager,"shapes":rows})
    del llm
    try: torch.cuda.empty_cache()
    except Exception: pass
json.dump(out, open("/home/ubuntu/d9_s1b_vllm_prefill.json","w"), indent=2)
print("WROTE d9_s1b_vllm_prefill.json", flush=True)
