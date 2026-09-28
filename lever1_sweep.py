#!/usr/bin/env python3
"""Lever 1 full sweep: batches {1,8,32,64} × caps {200,250,300,400,500,700} W.

Per cap: rebuild graph fresh, do warmup replays, time TARGET_REPLAYS more.
Crash-safe: atexit + SIGINT/SIGTERM restore to 700W.
"""
import os, sys, time, subprocess, threading, atexit, signal, json
import warnings
warnings.filterwarnings("ignore")

SAFE_DEFAULT = 700
def get_cap():
    r = subprocess.run(["nvidia-smi", "--query-gpu=power.limit",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5)
    return float(r.stdout.strip())

def set_cap(w):
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(int(w))],
                       capture_output=True, text=True, timeout=5)
    return r.returncode == 0

def restore():
    print(f"\n[restore] resetting cap to {SAFE_DEFAULT}W", file=sys.stderr)
    set_cap(SAFE_DEFAULT)

atexit.register(restore)
def sig_handler(signum, frame):
    restore(); sys.exit(128 + signum)
signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

print(f"[sweep] starting at {get_cap()}W (will restore on exit)")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = ("The future of artificial intelligence in GPU computing is to make "
          "every joule of energy count. Energy efficiency means doing more useful "
          "work per watt — that is the metric that matters for both training and "
          "inference at scale.")
MAX_LEN = 1024
WARMUP_REPLAYS = 40
TARGET_REPLAYS = 400
BATCHES = [1, 8, 32, 64]
CAPS_W  = [200, 250, 300, 400, 500, 700]

print("[sweep] loading model...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
prompt_ids_1 = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
prompt_len = prompt_ids_1.shape[1]
print(f"[sweep] loaded; prompt_len={prompt_len}")


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
        time.sleep(0.15)


def measure_at_cap(B, cap_w):
    """Build a fresh graph + run warmup + measure. Returns dict or None on failure."""
    if not set_cap(cap_w):
        return None
    prompt_ids = prompt_ids_1.expand(B, -1).contiguous()
    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=MAX_LEN,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(prompt_len, device="cuda", dtype=torch.long)
        out = model(input_ids=prompt_ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_token = out.logits[:, -1:].argmax(-1)
    input_ids = next_token.detach().clone()
    cache_pos = torch.tensor([prompt_len], device="cuda", dtype=torch.long)
    out_logits = torch.empty(B, 1, model.config.vocab_size, device="cuda", dtype=torch.float16)

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
    torch.cuda.synchronize()

    # Warmup with the new cap so DVFS settles
    for _ in range(WARMUP_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=power_sampler, args=(stop, samples), daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(TARGET_REPLAYS):
        cache_pos += 1; g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    mean_pw = sum(pw) / len(pw); mean_clk = sum(clk) / len(clk)
    tps = (B * TARGET_REPLAYS) / elapsed
    tok_w = tps / mean_pw if mean_pw > 0 else 0
    # Free everything before next cap
    del g, cache, cache_pos, out_logits, input_ids, prompt_ids
    torch.cuda.empty_cache()
    return dict(batch=B, cap_w=cap_w, tps=tps, draw_w=mean_pw, clk_mhz=mean_clk,
                tok_w=tok_w, elapsed_s=elapsed, n_samples=len(pw))


results = []
print("\n=== sweep: batches × caps ===")
print(f"{'batch':>5} {'cap_w':>5} {'tps':>9} {'draw_w':>7} {'clk':>6} {'tok/W':>8} {'sec':>5}")
for B in BATCHES:
    for cap_w in CAPS_W:
        try:
            r = measure_at_cap(B, cap_w)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  B={B} cap={cap_w}: OOM — skipping rest of batch")
            torch.cuda.empty_cache()
            r = None
            break
        except Exception as e:
            print(f"  B={B} cap={cap_w}: FAIL {type(e).__name__}: {e}")
            r = None
            continue
        if r is None: continue
        results.append(r)
        print(f"{B:>5} {cap_w:>5} {r['tps']:>9.2f} {r['draw_w']:>7.1f} "
              f"{r['clk_mhz']:>6.0f} {r['tok_w']:>8.4f} {r['elapsed_s']:>5.2f}")
    set_cap(SAFE_DEFAULT)

# Save raw results
out_path = "/home/ubuntu/op31-prod-fix/lever1_sweep_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[sweep] wrote {out_path}")

# Per-batch best cap within tps regression budget
print("\n=== best cap per batch (tps regression <=5% vs 700W) ===")
print(f"{'batch':>5} {'best_cap':>9} {'tps_700':>8} {'tps_best':>9} "
      f"{'tps_loss%':>10} {'tokW_700':>9} {'tokW_best':>10} {'tokW_gain%':>11}")
by_batch = {}
for r in results:
    by_batch.setdefault(r["batch"], {})[r["cap_w"]] = r
for B, caps in sorted(by_batch.items()):
    if 700 not in caps: continue
    base = caps[700]
    best = base
    for cap_w, r in sorted(caps.items()):
        loss = (base["tps"] - r["tps"]) / base["tps"] * 100
        if loss <= 5.0 and r["tok_w"] > best["tok_w"]:
            best = r
    loss = (base["tps"] - best["tps"]) / base["tps"] * 100
    gain = (best["tok_w"] - base["tok_w"]) / base["tok_w"] * 100
    print(f"{B:>5} {best['cap_w']:>9} {base['tps']:>8.1f} {best['tps']:>9.1f} "
          f"{loss:>10.2f} {base['tok_w']:>9.4f} {best['tok_w']:>10.4f} {gain:>11.1f}")

print("[sweep] done")
