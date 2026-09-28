#!/usr/bin/env python3
# G3 Marlin on vLLM batched DECODE (B>=8) — the product-path measurement.
#   ARM=off : baseline (substrate present, CIPHER_MARLIN unset -> passthrough). Saves ref output tokens.
#   ARM=on  : CIPHER_MARLIN=on. handled>0 from worker stderr (MATMUL exit totals); tok/s, tok/W;
#             output-token match vs OFF ref (the user's literal "batched output == Marlin-OFF ref").
# enforce_eager=True is MANDATORY: CUDA-graph replay bypasses the live cuBLAS GOT-patch (prod caveat).
# Arming: CUDA_INJECTION64_PATH=libcipher_rt.so (driver cuInit hook arms the GOT in the V1 worker).
import os, json, time, threading
import numpy as np
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

def main():
    import pynvml; pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
    from vllm import LLM, SamplingParams
    ARM   = os.environ["ARM"]
    MODEL = os.environ.get("MODEL", "/home/ubuntu/models/TinyLlama-1.1B")
    B     = int(os.environ.get("B", "8"))
    NEW   = int(os.environ.get("NEW", "64"))
    OUT   = os.environ.get("OUT", "/home/ubuntu/marlin_vllm")
    GMU   = float(os.environ.get("GMU", "0.85"))
    os.makedirs(OUT, exist_ok=True)

    class Power:
        def __init__(s): s.on=False; s.samples=[]
        def _loop(s):
            while s.on:
                try: s.samples.append(pynvml.nvmlDeviceGetPowerUsage(H)/1000.0)
                except Exception: pass
                time.sleep(0.02)
        def __enter__(s): s.on=True; s.t=threading.Thread(target=s._loop); s.t.start(); return s
        def __exit__(s,*a): s.on=False; s.t.join()
        def mean(s): return float(np.mean(s.samples)) if s.samples else 0.0

    base = ["The history of computing spans several distinct eras, each defined by",
            "In a quiet village nestled between two mountains, there lived",
            "The fundamental theorem of calculus connects the concept of",
            "Once the spacecraft cleared the atmosphere, the crew began",
            "Economists have long debated whether monetary policy can",
            "The recipe calls for fresh basil, ripe tomatoes, and a generous",
            "Deep beneath the ocean surface, bioluminescent creatures drift",
            "When the ancient library was finally excavated, archaeologists found"]
    prompts = [base[i % len(base)] for i in range(B)]
    sp = SamplingParams(temperature=0.0, max_tokens=NEW, ignore_eos=True)

    llm = LLM(model=MODEL, enforce_eager=True, dtype="bfloat16",
              max_num_seqs=max(B,8), gpu_memory_utilization=GMU, disable_log_stats=True)
    llm.generate(prompts, sp, use_tqdm=False)   # warmup (lazy-quantize stalls outside timing)

    with Power() as pw:
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t0
    toks = [list(o.outputs[0].token_ids) for o in outs]
    n_out = sum(len(t) for t in toks)
    tok_s = n_out / dt
    meanW = pw.mean()
    res = {"arm": ARM, "model": MODEL, "B": B, "new_tokens": NEW,
           "wall_s": round(dt,4), "out_tokens": n_out, "tok_s": round(tok_s,2),
           "mean_power_W": round(meanW,2),
           "tok_per_s_per_W": round(tok_s/meanW,5) if meanW else 0,
           "first12": [t[:12] for t in toks]}
    json.dump(res, open(f"{OUT}/{ARM}_vllm.json","w"), indent=2)
    np.save(f"{OUT}/{ARM}_tokens.npy", np.array([t[:NEW] for t in toks], dtype=np.int64))
    print(f"[VLLM {ARM}] B={B} tok/s={tok_s:.1f} meanW={meanW:.1f} tok/s/W={res['tok_per_s_per_W']} out_tokens={n_out}", flush=True)

if __name__ == "__main__":
    main()
