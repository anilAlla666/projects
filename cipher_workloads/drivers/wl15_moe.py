"""WL15 MoE Models.
NOTE: True MoE workload requires Mixtral-class weights which are not staged.
This driver uses TinyLlama as a proxy. P4.0.9 baseline marks this driver
'proxy_substitution=tinyllama' so downstream MFU comparisons are interpreted
as relative-to-proxy, not absolute MoE figures."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl15")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
inputs = tok(["Question: solve this."] * 4, return_tensors="pt", padding=True).to("cuda")

def step():
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 4, int(out.numel())

run_for_duration.run(step, DURATION, "WL15", tenant, "moe_proxy_decodes")
