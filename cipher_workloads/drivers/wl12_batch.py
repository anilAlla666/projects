"""WL12 Batch Processing. TinyLlama with batch=64."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl12")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
prompts = [f"Document {i}: this is a batch test." for i in range(64)]
inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=128).to("cuda")

def step():
    with torch.no_grad():
        out = m(**inputs)
    return 64, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL12", tenant, "batch_passes")
