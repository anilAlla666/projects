import os, time, json, ctypes
ARM=os.environ.get("SP_ARM","fp8"); SPEC=os.environ.get("SP_SPEC","1")=="1"; K=int(os.environ.get("SP_K","7"))
WL=os.environ.get("SP_WL","code"); OUTJ=os.environ["SP_OUTJSON"]
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from vllm import LLM, SamplingParams
kw=dict(model="mistralai/Mistral-7B-v0.1",enforce_eager=False,gpu_memory_utilization=0.85,max_model_len=4096,
        disable_log_stats=True,compilation_config={"cudagraph_capture_sizes":([8,16] if SPEC else [1])})
if ARM=="fp16": kw["dtype"]="float16"
if ARM=="fp8": kw["quantization"]="fp8"
if SPEC: kw["speculative_config"]={"method":"ngram","num_speculative_tokens":K,"prompt_lookup_max":4,"prompt_lookup_min":2}
llm=LLM(**kw)
WLS={"code":"Write a Python class `Matrix` with __init__, __add__, __mul__, transpose, and __repr__ methods, with docstrings and type hints:\n\nclass Matrix:",
     "prose":"Write a creative original short story about a lighthouse keeper who discovers a message in a bottle.",
     "rag":"Context: Paris is the capital of France. The Eiffel Tower is in Paris. The Louvre is in Paris. The Seine flows through Paris.\n\nUsing only the context, list every fact about Paris verbatim:"}
prompt=WLS[WL]
sp=SamplingParams(max_tokens=300,min_tokens=300,ignore_eos=True,temperature=0.0)
llm.generate([prompt],sp,use_tqdm=False)
best=None
for _ in range(3):
    t0=time.perf_counter(); o=llm.generate([prompt],sp,use_tqdm=False); dt=time.perf_counter()-t0
    best=max(best or 0, len(o[0].outputs[0].token_ids)/dt)
al=True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",mode=os.RTLD_NOLOAD)
except OSError: al=False
res=dict(arm=ARM,spec=SPEC,wl=WL,decode_tok_s=round(best,1),anchor_loaded=al)
print("WL",json.dumps(res)); json.dump(res,open(OUTJ,"w"),indent=1)
