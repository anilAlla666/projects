"""WL11 Agentic multi-turn. 10 turns × 4 simulated tool calls, TinyLlama."""
import sys, os, torch
sys.path.insert(0, "/home/ubuntu/cipher_workloads/launch_lib")
import tenant_register, run_for_duration

DURATION = int(os.environ.get("WL_DURATION", "600"))
MODEL = os.environ.get("WL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tenant = tenant_register.register("wl11")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

context = "User: solve this problem step by step.\nAssistant: "

def step():
    # 10 turns × 4 tool calls = 40 short generations
    total_toks = 0
    for turn in range(10):
        for tool in range(4):
            inp = tok(context, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = m.generate(**inp, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            total_toks += int(out.shape[-1])
    return 40, total_toks

run_for_duration.run(step, DURATION, "WL11", tenant, "subturns")
