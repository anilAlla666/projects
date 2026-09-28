"""WL04 vLLM Serving. Substitution: TinyLlama via vLLM."""
import sys, os, time
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl04")
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vllm not installed; run setup.sh first", file=sys.stderr)
    sys.exit(2)

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6)
sp = SamplingParams(max_tokens=32, temperature=0.0)
prompts = ["Hello, world."]

def step():
    out = llm.generate(prompts, sp)
    return 1, sum(len(o.outputs[0].token_ids) for o in out)

run_for_duration.run(step, DURATION, "WL04", tenant, "decodes")
