#!/usr/bin/env python3
# REAL OFF-byte-identical discipline gate (the foundation of the no-regress claim).
# Compares a forward with NO .so (clean baseline) vs the .so INJECTED with all
# CIPHER actuators OFF. They must be BIT-IDENTICAL (logit sum + argmax). Also
# reports whether Koopman/EDMD background calibration fires under the OFF state.
import os, sys, subprocess, re
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
CODE=r'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
m=AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B",torch_dtype=torch.float16,device_map="cuda").eval()
tok=AutoTokenizer.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B")
ids=tok("The product composes the engine and the actuators.",return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    h=m(ids).logits.float().sum().item(); a=int(m(ids).logits[0,-1].argmax())
print(f"OFFPROOF sum={h:.6f} argmax={a}")
'''
clean={k:v for k,v in os.environ.items() if not k.startswith("CIPHER") and k not in("CUDA_INJECTION64_PATH","LD_PRELOAD")}
base=subprocess.run([sys.executable,"-c",CODE],capture_output=True,text=True,env=clean)
offenv={**clean,"CUDA_INJECTION64_PATH":SO,"CIPHER_FP8":"0","CIPHER_VOLT":"0","CIPHER_MARLIN":"0","CIPHER_KOOPMAN":"0"}
off=subprocess.run([sys.executable,"-c",CODE],capture_output=True,text=True,env=offenv)
def grab(s):
    m=re.search(r"OFFPROOF sum=([\-\d.]+) argmax=(\d+)",s); return (m.group(1),m.group(2)) if m else (None,None)
b=grab(base.stdout); o=grab(off.stdout)
koop = "KOOPMAN" in off.stderr or "EDMD-LIVE" in off.stderr
print(f"baseline (no .so):           sum={b[0]} argmax={b[1]}")
print(f"OFF (.so injected, all OFF): sum={o[0]} argmax={o[1]}")
print(f"BYTE-IDENTICAL: {'PASS' if b==o and b[0] is not None else 'FAIL'}")
print(f"Koopman/EDMD fired under OFF (CUDA_INJECTION path): {koop}")
if base.returncode or off.returncode:
    print("STDERR(base tail):", base.stderr[-300:]); print("STDERR(off tail):", off.stderr[-300:])
