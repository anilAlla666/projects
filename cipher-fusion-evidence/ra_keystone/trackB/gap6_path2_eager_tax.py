# Track B Path-2: measure the REAL enforce_eager decode-throughput tax in vLLM 0.20.2 (prior was an estimate).
# This is the SHIPPING periodic-recompute detector's hosting cost (eager co-run disables cudagraph).
# Loads the model ONCE for the given mode (graph|eager), measures decode tok/s at batch 1 and 32.
import sys, time, json, os
mode = sys.argv[1]  # 'graph' or 'eager'
EAGER = (mode=='eager')
MODEL = os.environ.get("GAP6_MODEL","Qwen/Qwen2-7B")
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams

llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=0.85,
          max_model_len=2048, dtype="float16", disable_log_stats=True)

def bench(batch, out_len=256, prompt_len=32, reps=2):
    prompt = "The history of computing began " * 4  # ~ short prompt
    prompts = [prompt]*batch
    sp = SamplingParams(max_tokens=out_len, min_tokens=out_len, ignore_eos=True, temperature=0.0)
    # warmup
    llm.generate(prompts, sp, use_tqdm=False)
    best=None
    for _ in range(reps):
        t0=time.perf_counter()
        outs=llm.generate(prompts, sp, use_tqdm=False)
        dt=time.perf_counter()-t0
        gen_toks=sum(len(o.outputs[0].token_ids) for o in outs)
        tps=gen_toks/dt
        best = tps if (best is None or tps>best) else best
    return dict(batch=batch, out_len=out_len, gen_toks=gen_toks, wall_s=round(dt,3), decode_tok_s=round(best,1))

res = dict(mode=mode, enforce_eager=EAGER, model=MODEL,
           b1=bench(1), b32=bench(32))
print("RESULT", json.dumps(res))
json.dump(res, open(f"/home/ubuntu/cipher-fusion-evidence/ra_keystone/trackB/gap6_path2_{mode}.json","w"), indent=1)
