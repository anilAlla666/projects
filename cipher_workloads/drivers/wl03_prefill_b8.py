"""WL03 LLM Prefill B=8. Substitution: Mistral-7B-v0.1 for Llama-3.1-8B."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "mistralai/Mistral-7B-v0.1")
PROMPT_LEN = 1024

tenant = tenant_register.register("wl03")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

# 8 long-ish prompts to exercise prefill
long_prompt = ("The quick brown fox jumps over the lazy dog. " * 30)[:PROMPT_LEN]
prompts = [long_prompt] * 8
inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=PROMPT_LEN).to("cuda")

def step():
    with torch.no_grad():
        out = m(**inputs)  # prefill-only (logits)
    return 8, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL03", tenant, "prefills")
