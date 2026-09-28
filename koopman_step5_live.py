#!/usr/bin/env python3
# READ-ONLY. Step 5 (bounded) + gate-ladder live confirmation. End-to-end shipped actuator on NARROW fp16 Mistral.
# Arms (each fresh process, set via env): OFF (CIPHER_KOOPMAN=0), ON default gate (=1, beta=0.05),
# ON forced (=1, OOD=0.99 -> bypass gate, expose cache landmine + kernel speed/quality).
# Reads the gate-ladder counters: skip_dtype(bf16) / calls_total(eligible) / handled(substituted) / skipped(registry miss).
# Measures decode tok/s vs OFF baseline (locked 1200MHz). No .so change.
import os, sys, ctypes, time, subprocess
SO="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
MODEL="/home/ubuntu/models/Mistral-7B-v0.1"
ARM=os.environ.get("ARM","off")
LOCK=1200
import torch
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="cuda").eval()
# read koopman counters from the (injected or ctypes-loaded) .so
lib=ctypes.CDLL(SO)
CNT=["cipher_rt_koopman_calls_total","cipher_rt_koopman_calls_handled","cipher_rt_koopman_calls_skipped",
     "cipher_rt_koopman_skip_dtype","cipher_rt_koopman_bf16_observed","cipher_rt_koopman_is_active"]
for s in CNT:
    try: getattr(lib,s).restype=ctypes.c_ulong
    except Exception: pass
def counters():
    d={}
    for s in CNT:
        try: d[s.replace("cipher_rt_koopman_","")]=int(getattr(lib,s)())
        except Exception: d[s]= -1
    return d
def smlock(m): subprocess.run(["sudo","-n","nvidia-smi","-lgc",f"{m},{m}"],capture_output=True); time.sleep(0.3)
def smreset(): subprocess.run(["sudo","-n","nvidia-smi","-rgc"],capture_output=True)

# NARROW repetitive prompt (the working-config calibration/eval condition)
prompt="The quick brown fox jumps over the lazy dog. "*20
ids=tok(prompt,return_tensors="pt").input_ids.cuda()
print(f"[arm={ARM}] koopman_active={counters().get('is_active')} prompt_tok={ids.shape[1]}",flush=True)

c0=counters()
smlock(LOCK)
# warmup + timed greedy decode (manual loop, static-ish), measure tok/s
def decode(nnew):
    out=model(ids,use_cache=True); past=out.past_key_values; nxt=out.logits[:,-1:,:].argmax(-1)
    toks=[nxt]
    for _ in range(nnew-1):
        o=model(nxt,past_key_values=past,use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1:,:].argmax(-1); toks.append(nxt)
    return torch.cat(toks,1)
with torch.no_grad():
    decode(8)  # warmup
    torch.cuda.synchronize(); t0=time.perf_counter()
    gen=decode(128)
    torch.cuda.synchronize(); dt=time.perf_counter()-t0
smreset()
c1=counters()
tps=128/dt
delta={k:c1.get(k,0)-c0.get(k,0) for k in c1}
print(f"[arm={ARM}] tok/s={tps:.2f} wall={dt:.3f}s gen[:40]={tok.decode(gen[0,:40])!r}",flush=True)
print(f"[arm={ARM}] COUNTER delta: total={delta.get('calls_total')} handled={delta.get('calls_handled')} skipped={delta.get('calls_skipped')} skip_dtype={delta.get('skip_dtype')} bf16_observed={delta.get('bf16_observed')}",flush=True)
print(f"STEP5_RESULT arm={ARM} tps={tps:.3f} total={delta.get('calls_total')} handled={delta.get('calls_handled')} skip_dtype={delta.get('skip_dtype')} firstout={tok.decode(gen[0,:12])!r}",flush=True)
sys.stdout.flush(); os._exit(0)
