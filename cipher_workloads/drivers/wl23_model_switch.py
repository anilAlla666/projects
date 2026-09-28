"""WL23 Model Switch Stress. Alternate TinyLlama / Mistral generations."""
import sys, os, torch, gc
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
M_A = os.environ.get("WL_MODEL_A", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
M_B = os.environ.get("WL_MODEL_B", "mistralai/Mistral-7B-v0.1")

tenant = tenant_register.register("wl23")
from transformers import AutoModelForCausalLM, AutoTokenizer

def load(name):
    t = AutoTokenizer.from_pretrained(name)
    if t.pad_token is None: t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    return t, m

# Pre-load both; switch by reference each call.
tA, mA = load(M_A)
tB, mB = load(M_B)
switch = [True]

def step():
    use_A = switch[0]
    switch[0] = not switch[0]
    tok, m = (tA, mA) if use_A else (tB, mB)
    inp = tok(["Hello, how are you?"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return 1, int(out.shape[-1])

run_for_duration.run(step, DURATION, "WL23", tenant, "switches")
