"""WL02 LLM Decode B=8. Substitution: TinyLlama-1.1B."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl02")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
prompts = ["Hello, world."] * 8
inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")

def step():
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=32, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 8, int(out.shape[-1] * out.shape[0])

run_for_duration.run(step, DURATION, "WL02", tenant, "decodes")
