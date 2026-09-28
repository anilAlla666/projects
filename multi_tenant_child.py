#!/usr/bin/env python3
"""Single-tenant decode loop with per-token latency tracking.

Spawned by multi_tenant_poc.py. Reads:
    CIPHER_TENANT_ID — int, identifies this tenant
    MT_MODEL_PATH    — path to model
    MT_DURATION_S    — seconds to run
    MT_OUT_PATH      — JSON file to write final stats
    MT_PREFILL       — int, default 128
"""
import argparse, ctypes, gc, json, os, sys, time
import warnings; warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    tid    = int(os.environ.get("CIPHER_TENANT_ID", "0"))
    model_path = os.environ.get("MT_MODEL_PATH",
                                "/home/ubuntu/models/Llama-3.1-8B")
    duration_s = float(os.environ.get("MT_DURATION_S", "30"))
    out_path   = os.environ["MT_OUT_PATH"]
    prefill_len = int(os.environ.get("MT_PREFILL", "128"))
    run_chunk = int(os.environ.get("MT_RUN_CHUNK", "100"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    # Load CIPHER rt so its constructors fire (SENSE, SHIELD, FAIRNESS,
    # PREDICT, etc).  Skip when MT_LOAD_RT=0 so we can run a "no CIPHER"
    # control via the same harness.
    if os.environ.get("MT_LOAD_RT", "1") != "0":
        try:
            rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                             mode=ctypes.RTLD_GLOBAL)
            try:
                rt.cipher_fp8_compute_init.restype = ctypes.c_int
                rt.cipher_fp8_compute_init()
            except Exception:
                pass
            try:
                rt.cipher_fusion_kernels_init.restype = ctypes.c_int
                rt.cipher_fusion_kernels_init()
            except Exception:
                pass
        except OSError:
            pass

    print(f"[tenant {tid}] loading {model_path}...", flush=True)
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    print(f"[tenant {tid}] loaded in {time.perf_counter()-t_load:.1f}s; "
          f"GPU mem={torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)

    # Build prompt.
    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile = (prefill_len + len(ids_one) - 1) // len(ids_one)
    prompt = ids_one.repeat(n_tile)[:prefill_len].unsqueeze(0).to("cuda:0")

    MAX_LEN = prefill_len + run_chunk + 16
    cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)

    # Prefill once.
    with torch.no_grad():
        cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
        o = model(input_ids=prompt, cache_position=cp,
                   past_key_values=cache, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    next_tok = o.logits[:, -1:].argmax(-1)
    input_ids = next_tok.detach().clone()
    cache_pos = torch.tensor([prefill_len], device="cuda:0", dtype=torch.long)

    # Warmup 10 decode steps (untimed).
    for _ in range(10):
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                      past_key_values=cache, use_cache=True, return_dict=True)
        input_ids.copy_(o.logits.argmax(-1))
        cache_pos += 1
        if cache_pos.item() >= MAX_LEN - 4:
            # reset
            cache_pos = torch.tensor([prefill_len], device="cuda:0", dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=prompt, cache_position=torch.arange(prefill_len, device="cuda:0", dtype=torch.long),
                           past_key_values=cache, use_cache=True, return_dict=True)
            input_ids.copy_(o.logits[:, -1:].argmax(-1))
            cache_pos += 1
    torch.cuda.synchronize()

    # Timed loop: continuously generate, track per-token latency.
    print(f"[tenant {tid}] start timed loop (duration={duration_s}s)",
          flush=True)
    latencies_us = []
    n_tokens = 0
    n_resets = 0
    t_start = time.perf_counter()
    while True:
        t_tok = time.perf_counter()
        with torch.no_grad():
            o = model(input_ids=input_ids, cache_position=cache_pos,
                       past_key_values=cache, use_cache=True, return_dict=True)
        input_ids.copy_(o.logits.argmax(-1))
        cache_pos += 1
        torch.cuda.synchronize()  # ensure latency reflects true wall time
        latencies_us.append((time.perf_counter() - t_tok) * 1e6)
        n_tokens += 1
        if cache_pos.item() >= MAX_LEN - 4:
            # Re-prefill.
            cache_pos = torch.tensor([prefill_len], device="cuda:0", dtype=torch.long)
            with torch.no_grad():
                o = model(input_ids=prompt,
                          cache_position=torch.arange(prefill_len, device="cuda:0", dtype=torch.long),
                          past_key_values=cache, use_cache=True, return_dict=True)
            input_ids.copy_(o.logits[:, -1:].argmax(-1))
            cache_pos += 1
            n_resets += 1
        if time.perf_counter() - t_start >= duration_s:
            break
    elapsed = time.perf_counter() - t_start

    # Verify coherence on a fresh small generate.
    coh_text = ""
    try:
        with torch.no_grad():
            v = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                               max_new_tokens=20, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        coh_text = tok.decode(v[0, prompt.shape[1]:],
                              skip_special_tokens=False)[:60]
    except Exception:
        pass

    lat = sorted(latencies_us)
    n = len(lat)
    def pct(p): return lat[min(n-1, int(p * n))] if n else 0.0
    stats = dict(
        tenant_id=tid,
        n_tokens=n_tokens,
        n_resets=n_resets,
        elapsed_s=elapsed,
        tps=n_tokens / elapsed,
        lat_p50_us=pct(0.50),
        lat_p95_us=pct(0.95),
        lat_p99_us=pct(0.99),
        lat_mean_us=sum(lat)/n if n else 0,
        verify_text=coh_text,
        gpu_mem_gb=torch.cuda.memory_allocated(0)/1e9,
    )
    json.dump(stats, open(out_path, "w"), indent=2)
    print(f"[tenant {tid}] done: n={n_tokens} tps={stats['tps']:.1f} "
          f"P50={stats['lat_p50_us']:.0f}us P95={stats['lat_p95_us']:.0f}us "
          f"P99={stats['lat_p99_us']:.0f}us  text={coh_text!r}", flush=True)


if __name__ == "__main__":
    main()
