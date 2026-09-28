#!/usr/bin/env python3
"""Lever 1 hypothesis check: at batch=1 decode, does 700W vs 300W cap matter?

If tok/s collapses at 300W, the lever is weak (workload is power-limited).
If tok/s holds, HBM-bound hypothesis confirmed and full sweep is worth the time.

Crash-safe: atexit + SIGINT/SIGTERM restore original cap.
"""
import os, sys, time, subprocess, threading, atexit, signal
import warnings
warnings.filterwarnings("ignore")

ORIGINAL_CAP_W = 700  # H100 default
SAFE_DEFAULT = 700

def get_cap():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    return float(r.stdout.strip())

def set_cap(w):
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(int(w))],
                       capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        print(f"!! set_cap({w}) failed: {r.stderr.strip()}", file=sys.stderr)
        return False
    return True

def restore():
    print(f"\n[restore] resetting power cap to {SAFE_DEFAULT}W", file=sys.stderr)
    set_cap(SAFE_DEFAULT)

atexit.register(restore)
def sig_handler(signum, frame):
    restore(); sys.exit(128 + signum)
signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

ORIGINAL_CAP_W = int(get_cap())
print(f"[probe] starting at {ORIGINAL_CAP_W}W (will restore on exit)")

# Now load model and prep harness
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count.")
MAX_LEN = 4096   # cover all sweep caps × 1200 tokens + warmups in one capture
TARGET_TOKENS = 1000

print("[probe] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids.shape[1]

def power_sampler(stop_evt, samples):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.2)

def build_graph():
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    nt = out.logits[:, -1:].argmax(-1)
    input_ids = nt.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(1, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
            out_logits.copy_(o.logits)
            input_ids.copy_(out_logits.argmax(-1))
            cache_pos += 1
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids.copy_(out_logits.argmax(-1))
    return g, cache_pos

g, cache_pos = build_graph()
torch.cuda.synchronize()
print("[probe] graph captured")

def measure_one(label, cap_w, n_tokens=TARGET_TOKENS):
    global cache_pos
    if not set_cap(cap_w):
        print(f"  {label}: SKIP (cap-set failed)"); return
    # Reset cache_pos by rebuilding graph would be expensive. Instead, ensure
    # we never exceed MAX_LEN by checking before each measurement.
    if cache_pos.item() + n_tokens >= MAX_LEN - 4:
        print(f"  {label}: SKIP (cache_pos {cache_pos.item()}+{n_tokens} would OOB MAX_LEN={MAX_LEN})")
        return
    # Let DVFS settle under load: do 50 warmup replays
    for _ in range(50):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        cache_pos += 1
        g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[0] for s in samples[3:]] or [0]   # drop first 3 ramp samples
    clk = [s[1] for s in samples[3:]] or [0]
    mean_pw = sum(pw) / len(pw); mean_clk = sum(clk) / len(clk)
    tps = n_tokens / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    print(f"  {label:18s} cap={cap_w:3d}W  tps={tps:7.2f}  elapsed={elapsed:5.2f}s  draw={mean_pw:5.1f}W  clk={mean_clk:.0f}MHz  tok/W={tok_w:6.3f}")

print("=== Lever 1 hypothesis check (batch=1 graph decode) ===")
measure_one("baseline 700W", 700)
measure_one("floor 200W",   200)
print("[probe] done")
