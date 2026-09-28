"""WL13 Long Context 32K. Mistral-7B (native 32K context)."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "mistralai/Mistral-7B-v0.1")
CTX = int(os.environ.get("WL_CTX", "32768"))

tenant = tenant_register.register("wl13")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

# Generate a long context by simple repetition (no .format placeholders;
# tokenizer truncates to CTX tokens regardless of input string length).
long_full = "This is a long context test sentence. " * 8000
inputs = tok(long_full, return_tensors="pt", truncation=True, max_length=CTX).to("cuda")

def step():
    with torch.no_grad():
        out = m(**inputs)
    return 1, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL13", tenant, "long_ctx_passes")
