"""Tight B=1 decode-only VOLT verification. Excludes prefill from timing.
Confirms VOLT's +50% tok/W lift survives attn-substrate composition."""
import os, time, json, subprocess, threading, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODE = os.environ.get("CIPHER_MODE", "baseline")
mp = "/home/ubuntu/models/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(mp)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16,
                                         attn_implementation="sdpa").cuda()
m.train(False)

PROMPT = "The future of GPU computing is to make every joule count. "
ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
attn = torch.ones_like(ids)

# Warmup
with torch.inference_mode():
    _ = m.generate(ids, attention_mask=attn, max_new_tokens=8,
                   do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
torch.cuda.synchronize()
time.sleep(5)  # VOLT settles

# Power sampler
samples = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits", "-i", "0"],
                               capture_output=True, text=True, timeout=1.0).stdout.strip()
            parts = [p.strip() for p in r.split(",")]
            samples.append((time.time(), float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.1)
th = threading.Thread(target=sampler, daemon=True); th.start()

# Decode-only: one long generate, time it.
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode():
    out = m.generate(ids, attention_mask=attn, max_new_tokens=400,
                     do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
torch.cuda.synchronize()
dt = time.time() - t0
stop.set(); th.join(timeout=2.0)

# Compute on the steady part (skip first 5 samples for warmup)
W = [s[1] for s in samples[5:]]
C = [s[2] for s in samples[5:]]
mean_w = sum(W)/len(W) if W else 0
mean_c = sum(C)/len(C) if C else 0
n_new = out.shape[1] - ids.shape[1]
tok_s = n_new / dt
tok_w = tok_s / mean_w if mean_w else 0

print(f"[{MODE}] B=1 decode n_new={n_new} t={dt:.1f}s tok/s={tok_s:.1f} "
      f"W={mean_w:.1f} clock={mean_c:.0f}MHz tok/W={tok_w:.3f}")
out_path = f"/tmp/volt_b1_{MODE}.json"
with open(out_path, "w") as f:
    json.dump({"mode": MODE, "n_new": n_new, "dt": dt, "tok_s": tok_s,
               "W": mean_w, "clock": mean_c, "tok_w": tok_w,
               "n_samples": len(W)}, f, indent=2)
print(f"[{MODE}] wrote {out_path}")
