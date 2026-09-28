#!/usr/bin/env python3
"""
Phase 4.0 — Continuous-batching baseline harness.

Stock PyTorch only (no CIPHER, no LD_PRELOAD). TinyLlama 1.1B fp16 on H100.
Measures decode-step MFU at batches {32, 64, 128, 192, 256}. The result
either commits us to Instance 4 (graph-captured block) or forces a plan
change. No C++.
"""
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCHES = [32, 64, 128, 192, 256]
WARMUP_STEPS = 30
TIMED_STEPS = 100
PAST_SEQ_LEN = 512  # KV-cache length at decode time (realistic)

H100_FP16_PEAK_TFLOPS = 989.0       # NVIDIA spec
H100_FP16_SUSTAINED_TFLOPS = 700.0  # per feedback_mfu_baseline.md
H100_HBM3_PEAK_GB_S = 3350.0


def build_kv_cache(model, batch, past_len, device, dtype):
    """Allocate a DynamicCache pre-populated with `past_len` tokens of K/V."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = DynamicCache()
    for layer_idx in range(n_layers):
        k = torch.randn(batch, n_kv_heads, past_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, n_kv_heads, past_len, head_dim, device=device, dtype=dtype)
        cache.update(k, v, layer_idx)
    return cache


def time_decode(model, batch, past_len, device, dtype, n_steps):
    input_ids = torch.randint(0, model.config.vocab_size, (batch, 1), device=device)
    pkv = build_kv_cache(model, batch, past_len, device, dtype)
    pos = torch.full((batch, 1), past_len, device=device, dtype=torch.long)
    attn = torch.ones(batch, past_len + 1, device=device, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_steps):
            model(
                input_ids=input_ids,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=pkv,
                use_cache=True,
            )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps


def measure(model, batch, params, device, dtype):
    # Warmup
    time_decode(model, batch, PAST_SEQ_LEN, device, dtype, WARMUP_STEPS)
    # Timed
    step_s = time_decode(model, batch, PAST_SEQ_LEN, device, dtype, TIMED_STEPS)
    flops = 2.0 * params * batch
    hbm_bytes = 2.0 * params  # weight stream lower bound (B << 295 ridge)
    achieved_tflops = flops / step_s / 1e12
    return {
        "batch": batch,
        "step_ms": step_s * 1e3,
        "tokens_per_s": batch / step_s,
        "tflops": achieved_tflops,
        "mfu_spec": achieved_tflops / H100_FP16_PEAK_TFLOPS,
        "mfu_sustained": achieved_tflops / H100_FP16_SUSTAINED_TFLOPS,
        "hbm_gb_s": hbm_bytes / step_s / 1e9,
        "hbm_util": (hbm_bytes / step_s / 1e9) / H100_HBM3_PEAK_GB_S,
    }


def estimate_launch_floor(model, device, dtype):
    """Single-token, no-KV proxy of dispatch overhead (not a gate)."""
    input_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    pkv = build_kv_cache(model, 1, PAST_SEQ_LEN, device, dtype)
    pos = torch.full((1, 1), PAST_SEQ_LEN, device=device, dtype=torch.long)
    attn = torch.ones(1, PAST_SEQ_LEN + 1, device=device, dtype=torch.long)
    with torch.inference_mode():
        for _ in range(20):
            model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                  past_key_values=pkv, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                  past_key_values=pkv, use_cache=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50


def main():
    assert torch.cuda.is_available(), "No CUDA"
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"[load] {MODEL_ID} fp16")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)  # noqa: F841
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa"
    )
    model.train(False)
    params = sum(p.numel() for p in model.parameters())
    print(f"[load] params = {params/1e9:.3f} B  device = {torch.cuda.get_device_name(0)}")

    floor_s = estimate_launch_floor(model, device, dtype)
    print(f"[floor] decode step at B=1 (proxy)  = {floor_s*1e3:.3f} ms")

    rows = []
    for b in BATCHES:
        print(f"[bench] batch={b}  warmup={WARMUP_STEPS}  timed={TIMED_STEPS} ...", flush=True)
        r = measure(model, b, params, device, dtype)
        r["launch_overhead_frac_proxy"] = floor_s / r["step_ms"] * 1e3
        rows.append(r)
        print(f"        step={r['step_ms']:.3f} ms  toks/s={r['tokens_per_s']:.1f}  "
              f"TFLOPS={r['tflops']:.1f}  MFU(spec)={r['mfu_spec']*100:.1f}%  "
              f"MFU(sust)={r['mfu_sustained']*100:.1f}%  HBM={r['hbm_gb_s']:.0f} GB/s "
              f"({r['hbm_util']*100:.1f}%)")

    print("\n=== Phase 4.0 baseline (stock PyTorch, no CIPHER) ===")
    print(f"{'B':>4} | {'step ms':>8} | {'tok/s':>9} | {'TFLOPS':>7} | "
          f"{'MFU spec':>8} | {'MFU sust':>8} | {'HBM GB/s':>8} | {'HBM util':>8} | "
          f"{'launch%':>7}")
    print("-" * 96)
    for r in rows:
        print(f"{r['batch']:>4} | {r['step_ms']:>8.3f} | {r['tokens_per_s']:>9.1f} | "
              f"{r['tflops']:>7.1f} | {r['mfu_spec']*100:>7.1f}% | "
              f"{r['mfu_sustained']*100:>7.1f}% | {r['hbm_gb_s']:>8.0f} | "
              f"{r['hbm_util']*100:>7.1f}% | {r['launch_overhead_frac_proxy']*100:>6.1f}%")

    out = {
        "model": MODEL_ID,
        "params": params,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "past_seq_len": PAST_SEQ_LEN,
        "warmup": WARMUP_STEPS,
        "timed": TIMED_STEPS,
        "launch_floor_ms": floor_s * 1e3,
        "rows": rows,
    }
    out_path = Path(__file__).parent / "bench_continuous_batching_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[json] wrote {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    main()
