"""WL24 AWQ Quantized Inference."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")

tenant = tenant_register.register("wl24")
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    print("autoawq not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoAWQForCausalLM.from_quantized(MODEL, fuse_layers=False).cuda().eval()
inputs = tok("Hello, world.", return_tensors="pt").to("cuda")

def step():
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=32, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 1, int(out.shape[-1])

run_for_duration.run(step, DURATION, "WL24", tenant, "awq_decodes")
