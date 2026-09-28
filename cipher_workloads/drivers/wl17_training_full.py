"""WL17 Training Full. TinyLlama full-param training (no LoRA)."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl17")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
m.train()
opt = torch.optim.SGD(m.parameters(), lr=1e-5)
inputs = tok(["training data sample"] * 4, return_tensors="pt", padding=True).to("cuda")

def step():
    out = m(**inputs, labels=inputs.input_ids)
    out.loss.backward(); opt.step(); opt.zero_grad()
    return 1, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL17", tenant, "train_steps")
