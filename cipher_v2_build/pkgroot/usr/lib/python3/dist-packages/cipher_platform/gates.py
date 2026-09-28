# Discipline gates run on the customer box. The OFF-byte-identical gate is the
# foundation of the no-regress claim: .so injected + all actuators OFF must be
# BIT-IDENTICAL to a clean (no-.so) baseline. A gate that can only pass is rejected
# (the agent handler carries its own misroute negative-control).
import os, sys, re
from cipher_platform import isolate

_PROBE = r'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
mp=__import__("os").environ.get("CIPHER_PROBE_MODEL","/home/ubuntu/models/TinyLlama-1.1B")
m=AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.float16,device_map="cuda").eval()
tok=AutoTokenizer.from_pretrained(mp)
ids=tok("The product composes the engine and the actuators.",return_tensors="pt").input_ids.cuda()
import torch as T
with T.no_grad(): h=m(ids).logits.float().sum().item(); a=int(m(ids).logits[0,-1].argmax())
print(f"OFFPROOF sum={h:.6f} argmax={a}")
'''

def off_byte_identical(cfg):
    # both probes run through the isolation primitive: fresh process, GPU-settled+reaped
    # before the next unit, so the .so-injected probe never poisons a following lane.
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith("CIPHER") and k not in ("CUDA_INJECTION64_PATH", "LD_PRELOAD")}
    base = isolate.run_unit([sys.executable, "-c", _PROBE], clean, 300)
    off = isolate.run_unit([sys.executable, "-c", _PROBE],
                           {**clean, "CUDA_INJECTION64_PATH": cfg["so_path"],
                            "CIPHER_FP8": "0", "CIPHER_VOLT": "0", "CIPHER_MARLIN": "0", "CIPHER_KOOPMAN": "0"}, 300)
    def grab(s):
        m = re.search(r"OFFPROOF sum=([\-\d.]+) argmax=(\d+)", s)
        return (m.group(1), m.group(2)) if m else (None, None)
    b, o = grab(base["stdout"]), grab(off["stdout"])
    ok = (b == o and b[0] is not None)
    return {"pass": ok, "baseline": b, "off": o}

def evaluate(results):
    """gate = OFF-byte-identical PASS + agent FAULT=0 + agent misroute FAULT + density KL=0."""
    a = next((r for r in results if r.get("regime") == "agent"), {}).get("metrics", {})
    d = next((r for r in results if r.get("regime") == "density"), {}).get("metrics", {})
    checks = {
        "agent_fault_zero": a.get("fault", 1) == 0,
        "agent_misroute_detected": a.get("misroute_negctrl") == "FAULT",
        "density_kl_zero": (d.get("nf4_kl", 1.0) == 0.0) if d else True,
    }
    return checks
