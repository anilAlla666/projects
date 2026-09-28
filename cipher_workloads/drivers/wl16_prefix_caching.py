"""WL16 Prefix Caching. vLLM with prefix-cache enabled, repeated prefix."""
import sys, os
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl16")
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vllm not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.6,
          enable_prefix_caching=True)
sp = SamplingParams(max_tokens=16, temperature=0.0)
shared_prefix = "Once upon a time in a faraway land, " * 30
prompts = [shared_prefix + f"unique tail {i}." for i in range(8)]

def step():
    out = llm.generate(prompts, sp)
    return 8, sum(len(o.outputs[0].token_ids) for o in out)

run_for_duration.run(step, DURATION, "WL16", tenant, "prefix_decodes")
