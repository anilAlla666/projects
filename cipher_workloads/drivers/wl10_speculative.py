"""WL10 Speculative Decoding. vLLM with draft model; falls back to vanilla if unsupported."""
import sys, os
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
TARGET = os.environ.get("WL_TARGET", "mistralai/Mistral-7B-v0.1")
DRAFT  = os.environ.get("WL_DRAFT",  "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl10")
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vllm not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

# vLLM speculative decoding API: see vllm.SpeculativeConfig.
# Substitute: if the install doesn't support it, run vanilla and document.
try:
    llm = LLM(model=TARGET, speculative_model=DRAFT, num_speculative_tokens=4,
              dtype="float16", gpu_memory_utilization=0.6)
except (TypeError, ValueError):
    print("WL10: speculative decoding not supported in this vllm build; "
          "running vanilla mode", file=sys.stderr, flush=True)
    llm = LLM(model=TARGET, dtype="float16", gpu_memory_utilization=0.6)

sp = SamplingParams(max_tokens=64, temperature=0.0)
prompts = ["The capital of France is"]

def step():
    out = llm.generate(prompts, sp)
    return 1, sum(len(o.outputs[0].token_ids) for o in out)

run_for_duration.run(step, DURATION, "WL10", tenant, "decodes")
