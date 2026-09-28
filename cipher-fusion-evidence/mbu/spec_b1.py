import os, time, json, ctypes
ARM=os.environ.get("SP_ARM","fp16"); SPEC=os.environ.get("SP_SPEC","0")=="1"; K=int(os.environ.get("SP_K","7"))
OUTJ=os.environ.get("SP_OUTJSON",f"/home/ubuntu/cipher-fusion-evidence/mbu/specb1_{ARM}_s{int(SPEC)}.json")
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import torch
from vllm import LLM, SamplingParams
kw=dict(model="mistralai/Mistral-7B-v0.1",enforce_eager=False,gpu_memory_utilization=0.85,
        max_model_len=4096,disable_log_stats=True,compilation_config={"cudagraph_capture_sizes":([8,16] if SPEC else [1])})
if ARM=="fp16": kw["dtype"]="float16"
if ARM=="fp8": kw["quantization"]="fp8"
if SPEC: kw["speculative_config"]={"method":"ngram","num_speculative_tokens":K,"prompt_lookup_max":4,"prompt_lookup_min":2}
llm=LLM(**kw)
doc=("The transformer streams its weights from memory every decode step. "
     "Quantization reduces the bytes streamed per token. Speculative decoding reduces the number of forward passes. ")*8
prompt=f"Document:\n{doc}\n\nRepeat the document above exactly, word for word:\n{doc[:200]}"
sp=SamplingParams(max_tokens=300,min_tokens=300,ignore_eos=True,temperature=0.0)
llm.generate([prompt],sp,use_tqdm=False)  # warmup (batch 1)
best=None
for _ in range(3):
    t0=time.perf_counter(); o=llm.generate([prompt],sp,use_tqdm=False); dt=time.perf_counter()-t0
    tps=len(o[0].outputs[0].token_ids)/dt; best=tps if best is None or tps>best else best
al=True
try: ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",mode=os.RTLD_NOLOAD)
except OSError: al=False
res=dict(arm=ARM,spec=SPEC,K=K if SPEC else 0,decode_tok_s=round(best,1),batch=1,anchor_loaded=al)
print("SPECB1",json.dumps(res)); json.dump(res,open(OUTJ,"w"),indent=1)
