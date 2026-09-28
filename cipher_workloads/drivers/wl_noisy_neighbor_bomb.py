"""T4.2.4c noisy-neighbor BOMB driver.

Single-tenant Mistral-7B prefill at B=8, designed to saturate SMs and
starve concurrent decode tenants in the partition-OFF condition.

Same shape as wl03_prefill_b8.py but tenant_id from env.
"""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "120"))
MODEL = os.environ.get("WL_MODEL", "mistralai/Mistral-7B-v0.1")
TID = os.environ.get("CIPHER_TENANT_ID", "nn_bomb")
PROMPT_LEN = 1024

tenant = tenant_register.register(TID)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
m.train(False)

long_prompt = ("The quick brown fox jumps over the lazy dog. " * 30)[:PROMPT_LEN]
prompts = [long_prompt] * 8
inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
             max_length=PROMPT_LEN).to("cuda")

def step():
    with torch.no_grad():
        out = m(**inputs)
    return 8, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WLNN_B", tenant, "prefills")
