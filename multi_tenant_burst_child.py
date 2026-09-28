#!/usr/bin/env python3
"""Bursty agent tenant: generate 10 tokens, sleep 1-5s random, repeat.

Tracks per-burst latency (time to complete 10 tokens).
"""
import os, json, random, time, ctypes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings; warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    tid = int(os.environ.get("CIPHER_TENANT_ID", "0"))
    seed = int(os.environ.get("MT_SEED", str(tid * 17 + 3)))
    random.seed(seed)
    duration_s = float(os.environ.get("MT_DURATION_S", "60"))
    out_path = os.environ["MT_OUT_PATH"]
    burst_len = int(os.environ.get("MT_BURST_LEN", "10"))
    sleep_min = float(os.environ.get("MT_SLEEP_MIN", "1.0"))
    sleep_max = float(os.environ.get("MT_SLEEP_MAX", "5.0"))
    model_path = os.environ.get("MT_MODEL_PATH",
                                "/home/ubuntu/models/Llama-3.2-1B")
    prefill_len = int(os.environ.get("MT_PREFILL", "128"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    if os.environ.get("MT_LOAD_RT", "1") != "0":
        try:
            ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"),
                         mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile = (prefill_len + len(ids_one) - 1) // len(ids_one)
    prompt = ids_one.repeat(n_tile)[:prefill_len].unsqueeze(0).to("cuda:0")

    burst_us = []
    n_bursts = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration_s:
        # Stagger initial start so all 15 tenants don't fire at once.
        if n_bursts == 0:
            time.sleep(random.uniform(0, sleep_max))
        # Burst: prefill + decode burst_len tokens.
        MAX_LEN = prefill_len + burst_len + 4
        cache = StaticCache(config=model.config, max_cache_len=MAX_LEN)
        torch.cuda.synchronize()
        t_burst = time.perf_counter()
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
            o = model(input_ids=prompt, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
            input_ids = o.logits[:, -1:].argmax(-1).clone()
            cache_pos = torch.tensor([prefill_len], device="cuda:0", dtype=torch.long)
            for _ in range(burst_len):
                o = model(input_ids=input_ids, cache_position=cache_pos,
                          past_key_values=cache, use_cache=True, return_dict=True)
                input_ids.copy_(o.logits.argmax(-1))
                cache_pos += 1
        torch.cuda.synchronize()
        burst_us.append((time.perf_counter() - t_burst) * 1e6)
        n_bursts += 1
        # Sleep idle.
        time.sleep(random.uniform(sleep_min, sleep_max))

    elapsed = time.perf_counter() - t_start
    s = sorted(burst_us)
    n = len(s)
    def pct(p): return s[min(n-1, int(p*n))] if n else 0.0
    json.dump(dict(
        tenant_id=tid,
        n_bursts=n_bursts,
        elapsed_s=elapsed,
        burst_p50_us=pct(0.50),
        burst_p95_us=pct(0.95),
        burst_p99_us=pct(0.99),
        burst_mean_us=sum(s)/n if n else 0,
        burst_min_us=s[0] if n else 0,
        burst_max_us=s[-1] if n else 0,
    ), open(out_path, "w"), indent=2)
    print(f"[burst tenant {tid}] {n_bursts} bursts P50={pct(0.50):.0f}us "
          f"P95={pct(0.95):.0f}us P99={pct(0.99):.0f}us")


if __name__ == "__main__":
    main()
