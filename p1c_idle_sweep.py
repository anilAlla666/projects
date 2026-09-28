#!/usr/bin/env python3
"""PILLAR 1 STEP 1C — Idle injection sweep.

Hypothesis: explicit idle gaps between decode steps cause DVFS engagement
→ clock drop → power drop. Steps 1A+1B showed clock pinned at 1980 MHz on
this H100 with µs-scale natural gaps; sweep larger gaps to see if/when the
clock moves.

Sweeps inter-step CPU sleeps {0, 10, 50, 100, 200, 500, 1000, 5000} µs at
B=1 (most sensitive to per-step overhead) and reports tps, watts, clk.

If clock stays at 1980 MHz across the sweep: Pillar 1's idle-injection
premise is empirically false on this H100, document and move on.
"""
import os, sys, time, threading, subprocess, json, gc
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 5
MEASURE_SECONDS = 6.0

print("[p1c] loading Mistral-7B-v0.1 (vanilla, no CIPHER)...")
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
            r = subprocess.run(["nvidia-smi",
                                "--query-gpu=power.draw,clocks.gr",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 2:
                samples.append((float(parts[0]), int(parts[1])))
        except Exception: pass
        time.sleep(0.15)


def measure_with_idle(B, idle_us, prefill_len=PREFILL_LEN, max_decode=320):
    MAX_LEN = prefill_len + max_decode
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B,
                        max_cache_len=MAX_LEN, device="cuda",
                        dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prefill_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prefill_len], device="cuda", dtype=torch.long)
    out_logits = torch.full((B, 1, model.config.vocab_size), float("nan"),
                            dtype=torch.float16, device="cuda")

    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
    torch.cuda.synchronize()
    if torch.isnan(out_logits).any():
        return dict(error=f"NaN after warmup")

    out_logits.fill_(float("nan"))
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples),
                          daemon=True); th.start()
    sleep_s = idle_us * 1e-6
    t0 = time.perf_counter(); n_steps = 0
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
        n_steps += 1
        if cache_pos.item() >= MAX_LEN - 4: break
        if sleep_s > 0:
            torch.cuda.synchronize()  # ensure GPU truly idle during sleep
            time.sleep(sleep_s)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    if torch.isnan(out_logits).any():
        return dict(error=f"NaN after timed loop")

    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])

    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw  = sum(pw)/len(pw)
    mean_clk = sum(clk)/len(clk)
    min_clk  = min(clk) if clk else 0
    tps = (B * n_steps) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    out = dict(idle_us=idle_us, tps=tps, draw_w=mean_pw,
               clk_mhz_mean=mean_clk, clk_mhz_min=min_clk,
               tok_w=tok_w, elapsed_s=elapsed, n_steps=n_steps,
               last_token_text=sample_text, batch=B)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== IDLE INJECTION SWEEP at B=1 ===")
print(f"{'idle_us':>8} {'tps':>8} {'watts':>7} {'clk_mean':>9} {'clk_min':>8} "
      f"{'tok/W':>7} {'last_tok':>14}")
results_b1 = []
for idle_us in [0, 10, 50, 100, 200, 500, 1000, 5000]:
    r = measure_with_idle(1, idle_us)
    if "error" in r:
        print(f"  idle={idle_us}: SENTINEL FAIL: {r['error']}"); continue
    results_b1.append(r)
    print(f"{idle_us:>8} {r['tps']:>8.2f} {r['draw_w']:>7.1f} "
          f"{r['clk_mhz_mean']:>9.0f} {r['clk_mhz_min']:>8} "
          f"{r['tok_w']:>7.4f} {r['last_token_text']!r:>14}")

with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "p1c_idle_sweep.json"), "w") as f:
    json.dump(results_b1, f, indent=2)
print(f"\n[p1c] wrote p1c_idle_sweep.json")

# Verdict
clocks = [r["clk_mhz_min"] for r in results_b1]
if max(clocks) - min(clocks) < 50:
    print(f"\nVERDICT: clock pinned at ~{int(sum(clocks)/len(clocks))} MHz across "
          f"all idle settings (range: {min(clocks)}-{max(clocks)} MHz). "
          "DVFS does NOT engage on inter-step idle gaps. Pillar 1's "
          "idle-injection premise is empirically false on this H100; "
          "explicit clock locking (Pillar 5) is the correct mechanism.")
else:
    print(f"\nVERDICT: clock varies from {min(clocks)} to {max(clocks)} MHz; "
          "DVFS responds to idle. Find sweet spot.")
