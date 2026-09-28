"""T4.6.1 S4.3 — VOLT composition smoke.

Runs Mistral-7B B=4 decode for ~30s, samples nvidia-smi power, computes
tok/W. Three modes (selected via env CIPHER_MODE):
  baseline    : no LD_PRELOAD                       — reference watts
  attn_only   : LD_PRELOAD substrate, CIPHER_VOLT off, CIPHER_ATTN_TEST on
  volt_attn   : LD_PRELOAD substrate, CIPHER_VOLT on, CIPHER_ATTN_TEST on

Compares: volt_attn vs baseline tok/W ratio. T4.3.2 reported ~36-50%
watts reduction (≈+57% tok/W). Same lift should survive attn-substrate
load.
"""
import os, sys, time, json, subprocess, threading, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODE = os.environ.get("CIPHER_MODE", "baseline")
WARMUP_S = 5
SAMPLE_S = 30
BATCH = 4
GEN_TOKENS = 64

mp = "/home/ubuntu/models/Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained(mp)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float16,
                                         attn_implementation="sdpa").cuda()
m.train(False)

# Build a B=4 prompt: same 64-token prompt × 4
PROMPT = ("The future of GPU computing is to make every joule count. "
          "Energy efficiency, performance per watt, and total cost of "
          "ownership are increasingly tied to the workload's ability to ")
ids = tok([PROMPT] * BATCH, return_tensors="pt", padding=True).input_ids.cuda()
attn = torch.ones_like(ids)

# Warm up: one decode to JIT cuDNN, allocate KV cache, etc.
with torch.inference_mode():
    _ = m.generate(ids, attention_mask=attn, max_new_tokens=4,
                   do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
torch.cuda.synchronize()
time.sleep(WARMUP_S)

# Power sampler thread (nvidia-smi every 100 ms).
samples = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.gr,clocks.mem",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=1.0).stdout.strip()
            parts = [p.strip() for p in out.split(",")]
            samples.append((time.time(), float(parts[0]),
                            int(parts[1]), int(parts[2])))
        except Exception as e:
            samples.append((time.time(), -1.0, -1, -1))
        time.sleep(0.1)

th = threading.Thread(target=sampler, daemon=True); th.start()

# Loop generate calls for ~SAMPLE_S seconds, count tokens generated.
t0 = time.time()
gen_tokens = 0
gen_calls = 0
with torch.inference_mode():
    while time.time() - t0 < SAMPLE_S:
        out = m.generate(ids, attention_mask=attn, max_new_tokens=GEN_TOKENS,
                         do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
        torch.cuda.synchronize()
        gen_tokens += (out.shape[1] - ids.shape[1]) * BATCH
        gen_calls += 1
elapsed = time.time() - t0
stop.set(); th.join(timeout=2.0)

watts = [s[1] for s in samples if s[1] > 0]
clocks_gr = [s[2] for s in samples if s[2] > 0]
mean_w  = sum(watts) / len(watts) if watts else 0
mean_gr = sum(clocks_gr) / len(clocks_gr) if clocks_gr else 0
tok_s   = gen_tokens / elapsed
tok_w   = tok_s / mean_w if mean_w else 0

result = {
    "mode": MODE, "elapsed_s": elapsed, "gen_tokens": gen_tokens,
    "gen_calls": gen_calls, "batch": BATCH, "tok_s": tok_s, "mean_w": mean_w,
    "mean_clock_gr_mhz": mean_gr, "tok_w": tok_w, "n_power_samples": len(watts),
}
out_path = f"/tmp/volt_comp_{MODE}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[{MODE}] tokens={gen_tokens} t={elapsed:.1f}s "
      f"tok/s={tok_s:.1f} W={mean_w:.1f} clock={mean_gr:.0f}MHz "
      f"tok/W={tok_w:.3f}")
print(f"[{MODE}] wrote {out_path}")
