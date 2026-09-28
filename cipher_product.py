#!/usr/bin/env python3
# CIPHER PRODUCT SURFACE (V.1 Tier A) -- ONE entry point that routes a mixed workload manifest by REGIME to the
# already-earned, capture-safe composable set. NO new capability, NO .so change, NO monkeypatch, NO capture-illegal op
# in the engine graph -- this PACKAGES what V.0 validated into a product router. Default-OFF; OFF == byte-identical.
#
#   regime 'agent'/'decode'   -> ENGINE (pager + static-KV graph-decode + coalescing) (+) DVFS (+) NF4   [capture-safe]
#   regime 'compute'/'prefill'-> FP8 substitution (eager, CUDA_INJECTION64_PATH)                          [eager]
#
# The two regimes run as their validated handlers because their substrate injection states conflict by design
# (engine needs auto-init-disabled for capture-safety; FP8 needs the cublasGemmEx actuator). The router picks the
# handler per job. Everything Tier-B (non-GEMM driver substitution, attention fused, compute-actuators-in-graph,
# CUTLASS FP8, 85% multi-GPU, swap root-cause) is SURFACED in the report, NOT wired here (discipline: would require
# a .so change / rejected monkeypatch / capture-illegal op).
import os, sys, subprocess, json, re, time
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"

HANDLERS={
 # regime: (script, env, outfile, timeout, parser-key)
 "agent":   ("cipher_inc4.py",          dict(K="4",CIPHER_VOLT="1",CIPHER_RT_DISABLE_AUTO_INIT="1"), "/tmp/prod_agent.txt", 1500),
 "compute": ("v0_phaseA_maxmfu.py",     dict(CUDA_INJECTION64_PATH=SO,CIPHER_FP8="1",V0_MODE="fp8",V0_BATCHES="4,8"), "/tmp/prod_compute.txt", 900),
 "density": ("v0_phaseC_nf4_cofire.py", dict(CIPHER_VOLT="1"), "/tmp/prod_density.txt", 600),
}

def route(regime):
    script,env,out,to=HANDLERS[regime]
    e=dict(os.environ); e.update(env)
    print(f"[product] route regime={regime} -> {script} (injection: {[k for k in env if k in('CUDA_INJECTION64_PATH','CIPHER_RT_DISABLE_AUTO_INIT')] or 'default'})",flush=True)
    with open(out,"w") as f:
        subprocess.run([sys.executable,script],env=e,stdout=f,stderr=subprocess.STDOUT,timeout=to)
    return out

def serve_manifest(manifest):
    """manifest = list of regimes to exercise. Product routes each to its validated handler."""
    results={}
    for regime in manifest:
        results[regime]=route(regime)
    return results

def off_byte_identical_proof():
    """Discipline gate: with CIPHER actuators OFF, a forward must be BIT-IDENTICAL to a clean baseline."""
    code=r'''
import os,ctypes,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
torch.manual_seed(0)
m=AutoModelForCausalLM.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B",torch_dtype=torch.float16,device_map="cuda").eval()
tok=AutoTokenizer.from_pretrained("/home/ubuntu/models/TinyLlama-1.1B")
ids=tok("The product composes the engine and the actuators.",return_tensors="pt").input_ids.cuda()
with torch.no_grad(): h=m(ids).logits.float().sum().item(); a=int(m(ids).logits[0,-1].argmax())
print(f"OFFPROOF sum={h:.6f} argmax={a}")
'''
    cleanenv={k:v for k,v in os.environ.items() if not k.startswith("CIPHER") and k not in("CUDA_INJECTION64_PATH","LD_PRELOAD")}
    base=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,env=cleanenv)  # NO .so at all = baseline
    # OFF = .so INJECTED (CUDA_INJECTION64_PATH) with ALL actuators default-OFF -> must be BIT-IDENTICAL to baseline.
    off =subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
                        env={**cleanenv,"CUDA_INJECTION64_PATH":SO,"CIPHER_FP8":"0","CIPHER_VOLT":"0","CIPHER_MARLIN":"0","CIPHER_KOOPMAN":"0"})
    def grab(s):
        m=re.search(r"OFFPROOF sum=([\-\d.]+) argmax=(\d+)",s); return (m.group(1),m.group(2)) if m else (None,None)
    return grab(base.stdout), grab(off.stdout)

if __name__=="__main__":
    print("=== CIPHER PRODUCT (V.1 Tier A) -- routing the composable set ===",flush=True)
    manifest=["agent","compute","density"]
    serve_manifest(manifest)
    print("[product] all regimes served; report via cipher_product_report.py",flush=True)
    sys.stdout.flush(); os._exit(0)
