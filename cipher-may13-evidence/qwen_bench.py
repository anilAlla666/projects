#!/usr/bin/env python3
"""Cross-model verification: Qwen2.5-1.5B-Instruct.

The five-pillar work was on Mistral-7B. Pillar 5 (clock locking) was the
dominant lever there. This script asks: does it generalize?

Runs:
  - Vanilla eager baseline at B=1, 8, 32, 64 (PREFILL=1024 sustained ≥10s)
  - Clock-lock sweep {600, 800, 1000, 1200, 1400, 1600, 1980} MHz × same batches
  - Reports per-batch optimal clock + Δtok/W vs that batch's eager baseline

NaN sentinel + decoded-token sanity on every measurement.
"""
import os, sys, time, threading, subprocess, json, gc, atexit, signal
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


def lock_clock(mhz):
    return subprocess.run(["sudo","-n","nvidia-smi","-lgc",str(mhz)],
                          capture_output=True, text=True).returncode == 0
def restore_clock():
    subprocess.run(["sudo","-n","nvidia-smi","-rgc"], capture_output=True, text=True)
atexit.register(restore_clock)
def _sig(s,f): restore_clock(); sys.exit(128+s)
signal.signal(signal.SIGINT,_sig); signal.signal(signal.SIGTERM,_sig)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PREFILL_LEN     = 1024
WARMUP_STEPS    = 5
MEASURE_SECONDS = 8.0

print(f"[qwen] loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
print(f"[qwen] model: {model.config.architectures}, "
      f"hidden={model.config.hidden_size}, "
      f"intermediate={model.config.intermediate_size}, "
      f"num_layers={model.config.num_hidden_layers}, "
      f"num_kv_heads={model.config.num_key_value_heads}")

BASE = ("Energy efficiency means doing more useful work per watt. "
        "The future of GPU computing is to make every joule count. ")
ids_one = tok(BASE, return_tensors="pt").input_ids[0]
N_TILE = (PREFILL_LEN + len(ids_one)-1) // len(ids_one)
big = torch.cat([ids_one for _ in range(N_TILE)])[:PREFILL_LEN]
prompt_ids_1 = big.unsqueeze(0).to("cuda")
print(f"[qwen] prompt_len={prompt_ids_1.shape[1]}")


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


def measure(B, max_decode=320):
    MAX_LEN = PREFILL_LEN + max_decode
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
    last_token = int(out_logits[0, 0].argmax().item())
    sample_text = tok.decode([last_token])
    pw  = [s[0] for s in samples[2:]] or [0]
    clk = [s[1] for s in samples[2:]] or [0]
    out = dict(tps=(B*n_steps)/elapsed, draw_w=sum(pw)/len(pw),
               clk_mhz=sum(clk)/len(clk),
               tok_w=((B*n_steps)/elapsed)/(sum(pw)/len(pw) if pw else 1),
               elapsed_s=elapsed, n_steps=n_steps, batch=B,
               last_token=last_token, last_token_text=sample_text)
    del cache, cache_pos, out_logits, input_ids, prompt_ids
    gc.collect(); torch.cuda.empty_cache()
    return out


# Step 1: eager baseline (boost clock)
restore_clock(); time.sleep(0.5)
print("\n=== EAGER BASELINE (Qwen2.5-1.5B, boost 1980 MHz) ===")
print(f"{'B':>3} {'tps':>8} {'watts':>7} {'clk':>5} {'tok/W':>8} {'last_tok':>14}")
eager_base = {}
for B in [1, 8, 32, 64]:
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"  >> B={B} (free mem: {free:.1f} GB)", flush=True)
    try:
        r = measure(B)
    except torch.cuda.OutOfMemoryError:
        print(f"     OOM"); torch.cuda.empty_cache(); continue
    except Exception as e:
        print(f"     {type(e).__name__}: {e}"); continue
    if "error" in r:
        print(f"     SENTINEL FAIL: {r['error']}"); continue
    eager_base[B] = r
    print(f"{B:>3} {r['tps']:>8.2f} {r['draw_w']:>7.1f} {r['clk_mhz']:>5.0f} "
          f"{r['tok_w']:>8.4f} {r['last_token_text']!r:>14}")

# Step 2: clock sweep
print("\n=== CLOCK SWEEP × BATCH (Qwen2.5-1.5B) ===")
print(f"{'B':>3} {'clk':>5} {'tps':>8} {'watts':>6} {'clk_act':>7} {'tok/W':>7}")
all_results = []
for B in [1, 8, 32, 64]:
    if B not in eager_base: continue
    print(f"  >> B={B}", flush=True)
    rows = []
    for c in [600, 800, 1000, 1200, 1400, 1600, 1980]:
        if not lock_clock(c): print(f"    clk={c}: lock fail"); continue
        time.sleep(0.4)
        try: r = measure(B)
        except Exception as e:
            print(f"    clk={c}: {type(e).__name__}: {e}"); continue
        if "error" in r: print(f"    clk={c}: {r['error']}"); continue
        r["clk_set"] = c; rows.append(r); all_results.append(r)
        print(f"{B:>3} {c:>5} {r['tps']:>8.1f} {r['draw_w']:>6.1f} "
              f"{r['clk_mhz']:>7.0f} {r['tok_w']:>7.4f}")
    if rows:
        eb = eager_base[B]
        # Best tok/W among locked clocks
        best_tw = max(rows, key=lambda x: x["tok_w"])
        # Best tok/W with ≥95% peak tps
        peak = max(r["tps"] for r in rows)
        eligible = [r for r in rows if r["tps"] >= 0.95 * peak]
        safe = max(eligible, key=lambda x: x["tok_w"]) if eligible else best_tw
        x_e = eb["tok_w"]; x_s = safe["tok_w"]; x_a = best_tw["tok_w"]
        print(f"   ▸ eager={x_e:.4f}  safe@{safe['clk_set']}MHz={x_s:.4f} "
              f"({x_s/x_e:.2f}×)  aggr@{best_tw['clk_set']}MHz={x_a:.4f} "
              f"({x_a/x_e:.2f}×)")

restore_clock()

with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "qwen15_results.json"), "w") as f:
    json.dump(dict(eager_baseline=eager_base, sweep=all_results), f, indent=2)
print(f"\n[qwen] wrote qwen15_results.json")

print("\n=== CROSS-MODEL SUMMARY ===")
print(f"{'B':>3} {'eager Qwen':>11} {'best Qwen':>10} {'×eager':>7} {'@clk':>5}")
for B in [1, 8, 32, 64]:
    if B not in eager_base: continue
    eb = eager_base[B]
    rows = [r for r in all_results if r["batch"] == B]
    if not rows: continue
    best_tw = max(rows, key=lambda x: x["tok_w"])
    print(f"{B:>3} {eb['tok_w']:>11.4f} {best_tw['tok_w']:>10.4f} "
          f"{best_tw['tok_w']/eb['tok_w']:>6.2f}× {best_tw['clk_set']:>5}")
