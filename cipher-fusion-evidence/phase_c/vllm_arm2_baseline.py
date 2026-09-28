"""3-arm benchmark, Arm 2 setup — vLLM single-tenant baseline on TinyLlama-1.1B.
Same 5 prompts as the Phase B prototype, gen_len 128, greedy."""
import sys, time, json
from vllm import LLM, SamplingParams

PROMPTS = [
    "The Pacific Ocean is the largest ocean on Earth, covering approximately",
    "Sarah opened the old letter with trembling hands. The handwriting was "
    "her grandmother's, and the date read",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot =",
    "If a train leaves Chicago at 3 PM traveling east at 60 mph and another "
    "leaves New York at 4 PM traveling west at 80 mph,",
    "User: What are the main differences between supervised and unsupervised "
    "learning?\nAssistant: The main differences are",
]
llm = LLM(model="/home/ubuntu/models/TinyLlama-1.1B", dtype="float16",
          gpu_memory_utilization=0.5)
sp = SamplingParams(max_tokens=128, temperature=0.0)
llm.generate(PROMPTS, sp)                       # warm
ITERS = 150
t0 = time.time()
tot = 0
for _ in range(ITERS):
    out = llm.generate(PROMPTS, sp)
    tot += sum(len(o.outputs[0].token_ids) for o in out)
t1 = time.time()
import vllm
print("RESULT vllm=%s tok_s=%.2f total_tok=%d wall=%.2f t0=%.3f t1=%.3f"
      % (vllm.__version__, tot / (t1 - t0), tot, t1 - t0, t0, t1),
      file=sys.stderr, flush=True)
