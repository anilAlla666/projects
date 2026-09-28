#!/usr/bin/env python3
# PROBE 1: do 2+ independent in-process graph/decode engines COEXIST in ONE process (one CUDA context)?
# The earlier "won't fit" was a gpu_memory_utilization artifact -- re-test honestly with explicit low util.
# 2 small DISTINCT models, VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process engine, not subprocess).
import os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"   # in-process engine (else each LLM spawns a subprocess)
os.environ["VLLM_USE_DEEP_GEMM"]="0"; os.environ["VLLM_PLUGINS"]=""; os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
def used(): f,t=torch.cuda.mem_get_info(); return (t-f)/(1<<30)
A="/home/ubuntu/models/TinyLlama-1.1B"; B="/home/ubuntu/models/Llama-3.2-1B-Instruct"
if __name__=="__main__":
    from vllm import LLM, SamplingParams
    sp=SamplingParams(max_tokens=16, temperature=0.0)
    try:
        t=time.time(); llm_a=LLM(model=A, gpu_memory_utilization=0.25, max_model_len=2048, enforce_eager=True); ta=time.time()-t
        print(f"COEXIST: engine A (TinyLlama) loaded in-process in {ta:.0f}s, HBM={used():.1f}GiB",flush=True)
        t=time.time(); llm_b=LLM(model=B, gpu_memory_utilization=0.25, max_model_len=2048, enforce_eager=True); tb=time.time()-t
        print(f"COEXIST: engine B (Llama-3.2-1B) loaded in SAME process in {tb:.0f}s, HBM={used():.1f}GiB",flush=True)
        oa=llm_a.generate(["The capital of France is"], sp, use_tqdm=False)
        ob=llm_b.generate(["The capital of France is"], sp, use_tqdm=False)
        # also: does A still work AFTER B loaded (no clobber)?
        oa2=llm_a.generate(["The capital of France is"], sp, use_tqdm=False)
        ok = bool(oa and ob and oa2)
        ident = (oa[0].outputs[0].token_ids == oa2[0].outputs[0].token_ids)
        print(f"COEXIST VERDICT: 2 in-process engines coexist + both serve = {ok}; A unchanged after B load = {ident}",flush=True)
        print(f"  A: {oa[0].outputs[0].text[:50]!r}",flush=True)
        print(f"  B: {ob[0].outputs[0].text[:50]!r}",flush=True)
        print(f"  -> coexistence (both resident) {'WORKS' if ok else 'FAILS'}; consolidation (page idle out) needs pager-managed memory = separate question",flush=True)
    except Exception as e:
        import traceback
        print(f"COEXIST FAILED: {str(e)[:300]}",flush=True)
        print("  binding limit (diagnose for the CIPHER engine build):",flush=True)
        traceback.print_exc()
    sys.stdout.flush(); os._exit(0)
