#!/usr/bin/env python3
"""PILLAR 1 STEP 1A — Eager baseline measurement.

Vanilla PyTorch. No LD_PRELOAD. No CIPHER. No graph capture. No INT4. No fused
norms. This is the TRUE per-batch eager baseline that subsequent CIPHER
measurements compare against.

Stack:
  - Mistral-7B-v0.1 fp16
  - Standard nn.Linear, MistralRMSNorm, MistralMLP forwards (untouched)
  - StaticCache (matches CIPHER measurement infrastructure for fairness)
  - Eager decode loop (no torch.cuda.graph)
  - PREFILL_LEN=1024 at every B (uniform → fair CIPHER comparison)
  - ≥10 s sustained decode for stable power sampling

Guards:
  - NaN sentinel on out_logits before warmup AND before timed loop
  - argmax of last logits decodes to a real Mistral vocab token
  - Memory-free print before each measure() to confirm no OOM
"""
import os, sys, time, threading, subprocess, json, gc
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 10
MEASURE_SECONDS = 10.0

print("[p1a] loading Mistral-7B-v0.1 (fp16, vanilla)...")
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
print(f"[p1a] prompt_len={prompt_ids_1.shape[1]}  fp16 weights only, no patches")


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


def measure(B, prefill_len=PREFILL_LEN, max_decode=320):
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

    # Warmup decode steps in eager mode
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True,
                      return_dict=True)
        out_logits.copy_(o.logits)
        input_ids = out_logits.argmax(-1)
        cache_pos += 1
    torch.cuda.synchronize()

    nan_after_warmup = torch.isnan(out_logits).sum().item()
    if nan_after_warmup > 0:
        return dict(error=f"NaN after warmup ({nan_after_warmup})")

    # NaN refill before timed loop
    out_logits.fill_(float("nan"))

    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples),
                          daemon=True); th.start()
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
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    nan_after_loop = torch.isnan(out_logits).sum().item()
    if nan_after_loop > 0:
        return dict(error=f"NaN after timed loop ({nan_after_loop})")

    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])

    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw  = sum(pw)/len(pw)
    mean_clk = sum(clk)/len(clk)
    tps = (B * n_steps) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    out = dict(tps=tps, draw_w=mean_pw, clk_mhz=mean_clk, tok_w=tok_w,
               elapsed_s=elapsed, n_steps=n_steps,
               last_token=last_token, last_token_text=sample_text,
               batch=B, prefill=prefill_len)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


print("\n=== EAGER BASELINE (vanilla PyTorch, no CIPHER, no graph) ===")
print(f"{'B':>4} {'P':>5} {'tps':>9} {'watts':>7} {'clk':>5} {'tok/W':>8} "
      f"{'sec':>5} {'last_token':>22}")
results = []
for B in [1, 8, 32, 64]:
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  >> B={B} P={PREFILL_LEN}  (free mem: {free:.1f} GB)", flush=True)
    try:
        r = measure(B)
    except torch.cuda.OutOfMemoryError:
        print(f"     OOM at PREFILL_LEN={PREFILL_LEN}; retrying with shorter prefill")
        torch.cuda.empty_cache()
        try:
            r = measure(B, prefill_len=512)
        except torch.cuda.OutOfMemoryError:
            print(f"     OOM at prefill=512 too — skipping")
            torch.cuda.empty_cache()
            continue
    except Exception as e:
        print(f"     {type(e).__name__}: {e}"); continue
    if "error" in r:
        print(f"     SENTINEL FAIL: {r['error']}"); continue
    results.append(r)
    print(f"{B:>4} {r['prefill']:>5} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
          f"{r['clk_mhz']:>5.0f} {r['tok_w']:>8.4f} {r['elapsed_s']:>5.1f} "
          f"{r['last_token_text']!r:>22}")

with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "p1a_eager_baseline.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[p1a] wrote p1a_eager_baseline.json")
