import os, ctypes
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
import vllm.compilation.cuda_graph as cg
from collections import Counter
modes=Counter()
_orig=cg.CUDAGraphWrapper.__call__
def patched(self,*a,**k):
    if cg.is_forward_context_available():
        fc=cg.get_forward_context()
        modes[(str(self.runtime_mode),str(fc.cudagraph_runtime_mode))]+=1
    return _orig(self,*a,**k)
cg.CUDAGraphWrapper.__call__=patched
from vllm import LLM, SamplingParams
llm=LLM(model="mistralai/Mistral-7B-v0.1",enforce_eager=False,gpu_memory_utilization=0.85,
        max_model_len=2048,dtype="float16",disable_log_stats=True)
# mixed: short + long prompts, varied gen lengths -> prefill + decode + continuous batching
import random
prompts=[("The history of computing began "*random.randint(2,40)) for _ in range(24)]
sps=[SamplingParams(max_tokens=random.choice([16,64,128]),min_tokens=8,ignore_eos=True,temperature=0.0) for _ in prompts]
llm.generate(prompts,sps,use_tqdm=False)
print("RUNTIME_MODES",dict(modes))
