"""WL01 LLM Decode B=1.
Substitution: TinyLlama-1.1B for Llama-3.2-1B (gated model)."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl01")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
prompt = tok("Hello, world.", return_tensors="pt").to("cuda")
PROMPT_LEN = int(prompt.input_ids.shape[-1])
MAX_NEW = 32

def step():
    with torch.no_grad():
        out = m.generate(**prompt, max_new_tokens=MAX_NEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    # Count only newly-generated tokens, not prompt echo.
    return 1, int(out.shape[-1]) - PROMPT_LEN

run_for_duration.run(step, DURATION, "WL01", tenant, "decodes")
