"""3-arm extended — one vLLM tenant at B requests-in-flight. Submits B streams'
worth of the 5-prompt workload (5*B prompts), vLLM caps concurrency at
max_num_seqs=B (intra-process continuous batching). Barrier-synced."""
import os, sys, time, json
from vllm import LLM, SamplingParams

TENANT = os.environ["TENANT"]
OUT = os.environ["OUT_JSON"]
GMU = float(os.environ.get("GMU", "0.10"))
EAGER = os.environ.get("EAGER", "1") == "1"
B = int(os.environ.get("B", "1"))
READY = "/tmp/arm2_ready_%s" % TENANT
GO = "/tmp/arm2_go"

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
llm = LLM(model=os.environ.get("VLLM_MODEL","/home/ubuntu/models/TinyLlama-1.1B"), dtype="float16",
          gpu_memory_utilization=GMU, enforce_eager=EAGER, max_model_len=512,
          max_num_seqs=B, disable_log_stats=True)
sp = SamplingParams(max_tokens=128, temperature=0.0)
work = PROMPTS * B                                  # B streams of the 5 prompts
llm.generate(work, sp, use_tqdm=False)              # warm
open(READY, "w").write("ready")
while not os.path.exists(GO):
    time.sleep(0.1)
t0 = time.time()
o = llm.generate(work, sp, use_tqdm=False)
t1 = time.time()
outs = [list(x.outputs[0].token_ids) for x in o]
tot = sum(len(x) for x in outs)
json.dump({"tenant": TENANT, "B": B, "t0": t0, "t1": t1, "total_tok": tot,
           "tok_s": tot / (t1 - t0), "outputs": outs,
           "prompt_idx": [i % 5 for i in range(len(work))]}, open(OUT, "w"))
print("ARM2-TENANT %s B=%d tok_s=%.1f total=%d wall=%.2f"
      % (TENANT, B, tot / (t1 - t0), tot, t1 - t0), file=sys.stderr, flush=True)
