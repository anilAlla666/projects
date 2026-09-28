#!/usr/bin/env python3
"""PILLAR 5 STEPS 5A + 5B — Clock-locked sweep.

Step 5A established (in p1c) that DVFS doesn't engage on natural idle gaps —
clock pinned at boost. Pillar 5's right tool is explicit NVML clock locking.

Step 5B: sweep `sudo nvidia-smi -lgc <freq>` across {600, 800, 1000, 1200,
1400, 1600, 1980} MHz at B={1, 8, 32, 64}. For each (B, clk) measure tps,
watts, tok/W in eager mode. Find the per-batch sweet spot.

Hypothesis: at B=1 (HBM-bound) low clock saves significant power with
minimal tps loss. At B=64 (more compute-bound) the optimal clock is
higher.

Restores boost on exit.
"""
import os, sys, time, threading, subprocess, json, gc, atexit, signal
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


def lock_clock(mhz):
    r = subprocess.run(["sudo","-n","nvidia-smi","-lgc",str(mhz)],
                       capture_output=True, text=True)
    return r.returncode == 0


def restore_clock():
    subprocess.run(["sudo","-n","nvidia-smi","-rgc"],
                   capture_output=True, text=True)


atexit.register(restore_clock)
def _sig(s, f): restore_clock(); sys.exit(128+s)
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)


MODEL = "mistralai/Mistral-7B-v0.1"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 5
MEASURE_SECONDS = 5.0

print("[p5b] loading Mistral-7B-v0.1 (vanilla eager)...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
BASE = ("Energy efficiency means doing more useful work per watt. "
        "The future of GPU computing is to make every joule count. ")
ids_one = tok(BASE, return_tensors="pt").input_ids[0]
N_TILE = (PREFILL_LEN + len(ids_one)-1) // len(ids_one)
big = torch.cat([ids_one for _ in range(N_TILE)])[:PREFILL_LEN]
prompt_ids_1 = big.unsqueeze(0).to("cuda")


def power_sampler(stop, samples):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2: samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.15)


def measure(B):
    MAX_LEN = PREFILL_LEN + 200
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(PREFILL_LEN, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([PREFILL_LEN], device="cuda", dtype=torch.long)
    out_logits = torch.full((B, 1, model.config.vocab_size), float("nan"),
                            dtype=torch.float16, device="cuda")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
    torch.cuda.synchronize()
    if torch.isnan(out_logits).any(): return dict(error="NaN warmup")
    out_logits.fill_(float("nan"))
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter(); n_steps = 0
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
        n_steps += 1
        if cache_pos.item() >= MAX_LEN - 4: break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    if torch.isnan(out_logits).any(): return dict(error="NaN loop")
    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    out = dict(tps=(B*n_steps)/elapsed, draw_w=sum(pw)/len(pw),
               clk_mhz=sum(clk)/len(clk),
               tok_w=((B*n_steps)/elapsed)/(sum(pw)/len(pw) if pw else 1),
               elapsed_s=elapsed, n_steps=n_steps, batch=B)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


CLOCKS = [600, 800, 1000, 1200, 1400, 1600, 1980]
BATCHES = [1, 8, 32, 64]

print(f"\n=== STEP 5B: clock × batch sweep ===")
print(f"{'B':>3} {'clk_set':>7} {'tps':>9} {'watts':>7} {'clk_act':>8} {'tok/W':>8}")
all_results = []
for B in BATCHES:
    print(f"  >> B={B}", flush=True)
    rows = []
    for c in CLOCKS:
        if not lock_clock(c):
            print(f"    clk={c}: lock failed"); continue
        time.sleep(0.5)  # let clock settle
        try:
            r = measure(B)
        except Exception as e:
            restore_clock()
            print(f"    clk={c}: {type(e).__name__}: {e}"); continue
        if "error" in r:
            print(f"    clk={c}: SENTINEL FAIL: {r['error']}"); continue
        r["clk_set"] = c; rows.append(r); all_results.append(r)
        print(f"{B:>3} {c:>7} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
              f"{r['clk_mhz']:>8.0f} {r['tok_w']:>8.4f}")
    # Optimal for this B
    if rows:
        best_tw = max(rows, key=lambda r: r["tok_w"])
        best_tps = max(rows, key=lambda r: r["tps"])
        print(f"   ▸ best tok/W at clk={best_tw['clk_set']} = {best_tw['tok_w']:.4f}  "
              f"(tps {best_tw['tps']:.1f}, {(1-best_tw['tps']/best_tps['tps'])*100:.1f}% off peak tps)")

restore_clock()
with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "p5b_clock_sweep.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n[p5b] wrote p5b_clock_sweep.json")

# Pick optimal clock per batch
print("\n=== optimal clock per batch (best tok/W with ≥95% peak tps) ===")
opt = {}
for B in BATCHES:
    rows = [r for r in all_results if r["batch"] == B]
    if not rows: continue
    peak_tps = max(r["tps"] for r in rows)
    eligible = [r for r in rows if r["tps"] >= 0.95 * peak_tps]
    best = max(eligible, key=lambda r: r["tok_w"]) if eligible else max(rows, key=lambda r: r["tok_w"])
    opt[B] = best["clk_set"]
    print(f"  B={B}: clk={best['clk_set']} tps={best['tps']:.1f} tok/W={best['tok_w']:.4f}")
with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "p5_optimal_clocks.json"), "w") as f:
    json.dump(opt, f, indent=2)
