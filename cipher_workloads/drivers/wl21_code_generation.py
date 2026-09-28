"""WL21 Code Generation. TinyLlama with code-completion prompts."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl21")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
inputs = tok(["def factorial(n):\n    "], return_tensors="pt").to("cuda")

def step():
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 1, int(out.shape[-1])

run_for_duration.run(step, DURATION, "WL21", tenant, "code_decodes")
