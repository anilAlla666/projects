# Quality stress: perplexity on real text (teacher-forced via prompt_logprobs) for fp16 vs fp8 vs gptq.
# Quantifies the unmeasured "quality cost" caveat the TPW/MBU wins ride on.
import os, json, math
ARM=os.environ["Q_ARM"]; OUTJ=os.environ["Q_OUTJSON"]
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams
kw=dict(model="mistralai/Mistral-7B-v0.1",enforce_eager=True,gpu_memory_utilization=0.85,max_model_len=4096,disable_log_stats=True)
if ARM=="fp16": kw["dtype"]="float16"
if ARM=="fp8": kw["quantization"]="fp8"
if ARM=="gptq": kw["model"]="TheBloke/Mistral-7B-v0.1-GPTQ"
llm=LLM(**kw)
# real-ish held-out text passages (factual + narrative + technical)
texts=[
 "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. It generates most of the cell's supply of adenosine triphosphate, used as a source of chemical energy. Mitochondria contain their own genome, which is separate from the nuclear genome of the cell.",
 "In 1969, the Apollo 11 mission landed the first humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle while Michael Collins remained in lunar orbit aboard the command module.",
 "A transformer is a deep learning architecture that relies on the attention mechanism. It processes the entire input sequence in parallel, computing weighted relationships between all tokens, which allows it to capture long-range dependencies more efficiently than recurrent networks.",
]
sp=SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
outs=llm.generate(texts, sp, use_tqdm=False)
nlls=[]; ntok=0
for o in outs:
    plp=o.prompt_logprobs
    if not plp: continue
    for d in plp[1:]:                       # skip first token (no preceding context)
        if d:
            lp=list(d.values())[0].logprob; nlls.append(-lp); ntok+=1
ppl=math.exp(sum(nlls)/len(nlls)) if nlls else None
res=dict(arm=ARM, ppl=round(ppl,4) if ppl else None, n_tokens=ntok)
print("PPL", json.dumps(res)); json.dump(res, open(OUTJ,"w"), indent=1)
