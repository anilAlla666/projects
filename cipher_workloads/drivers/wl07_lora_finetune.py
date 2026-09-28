"""WL07 LoRA Fine-tuning. Substitution: TinyLlama-1.1B + LoRA."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl07")
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("peft not installed; run setup.sh first", file=sys.stderr); sys.exit(2)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0)
m = get_peft_model(m, lc)
m.train()
opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-4)
inputs = tok(["the cat sat on the mat"] * 4, return_tensors="pt", padding=True).to("cuda")

def step():
    out = m(**inputs, labels=inputs.input_ids)
    out.loss.backward(); opt.step(); opt.zero_grad()
    return 1, int(inputs.input_ids.numel())

run_for_duration.run(step, DURATION, "WL07", tenant, "lora_steps")
