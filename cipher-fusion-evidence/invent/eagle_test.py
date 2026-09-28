import os, time, json
os.environ.setdefault("VLLM_LOGGING_LEVEL","INFO")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
BASE="meta-llama/Llama-3.1-8B-Instruct"; EAGLE="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
MODE=os.environ.get("E_MODE","eagle")  # base | eagle
tok=AutoTokenizer.from_pretrained(BASE)
kw=dict(model=BASE,enforce_eager=False,gpu_memory_utilization=0.82,max_model_len=2048,disable_log_stats=False,dtype="bfloat16")
if MODE=="eagle":
    kw["speculative_config"]={"method":"eagle3","model":EAGLE,"num_speculative_tokens":5}
llm=LLM(**kw)
msgs=[[{"role":"user","content":c}] for c in [
  "Explain in detail how the TCP three-way handshake works, step by step.",
  "Write a Python function to compute the nth Fibonacci number with memoization, with comments."]]
prompts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs]
sp=SamplingParams(max_tokens=256,temperature=0.0)
llm.generate(prompts,sp,use_tqdm=False)  # warmup
best=None
for _ in range(3):
    t0=time.perf_counter(); o=llm.generate(prompts,sp,use_tqdm=False); dt=time.perf_counter()-t0
    g=sum(len(x.outputs[0].token_ids) for x in o); tps=g/dt; best=tps if best is None or tps>best else best
print("EAGLE_RESULT", json.dumps({"mode":MODE,"tok_s":round(best,1),"sample_out":o[0].outputs[0].text[:80]}))
