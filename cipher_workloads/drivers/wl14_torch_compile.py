"""WL14 torch.compile. TinyLlama wrapped in torch.compile."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl14")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
try:
    m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
except Exception as e:
    print(f"torch.compile failed: {e} — running uncompiled", file=sys.stderr)
inputs = tok(["Hello"] * 8, return_tensors="pt", padding=True).to("cuda")

def step():
    with torch.no_grad():
        out = m(**inputs)
    return 8, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL14", tenant, "compile_passes")
