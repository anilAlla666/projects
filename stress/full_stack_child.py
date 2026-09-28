#!/usr/bin/env python3
"""Sustained Llama-3.2-1B decode for the full-stack stress test.

Each tenant: load model once, generate burst_len tokens per burst as fast as
possible, no sleep between bursts. Records aggregate token count + per-burst
timings.
"""
import os, json, time, ctypes, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"
import warnings; warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.normpath(os.path.join(ROOT, "..", "libcipher_rt.so"))


def main():
    tid          = int(os.environ.get("CIPHER_TENANT_ID", "0"))
    duration_s   = float(os.environ.get("MT_DURATION_S", "60"))
    out_path     = os.environ["MT_OUT_PATH"]
    burst_len    = int(os.environ.get("MT_BURST_LEN", "10"))
    batch        = int(os.environ.get("MT_BATCH", "1"))
    prefill_len  = int(os.environ.get("MT_PREFILL", "128"))
    model_path   = os.environ.get("MT_MODEL_PATH",
                                  "/home/ubuntu/models/Llama-3.2-1B")
    load_rt      = os.environ.get("MT_LOAD_RT", "1") != "0"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    rt = None
    if load_rt and os.path.exists(RT_PATH):
        try:
            rt = ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)
        except Exception as ex:
            print(f"[t{tid}] CIPHER rt load failed: {ex}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map={"": 0})
    # Set inference mode (no dropout, no autograd state)
    for p in model.parameters():
        p.requires_grad_(False)

    # Wire fusion patches when CIPHER_FUSION_KERNELS / CIPHER_FUSION are on.
    # This lets the C++ cipher_fused_rmsnorm / cipher_fused_silu_mul actually
    # fire (otherwise the Python decoder forward never reaches them).
    fusion_on = (os.environ.get("CIPHER_FUSION_KERNELS", "").lower() in ("on", "1", "true")
                 or os.environ.get("CIPHER_FUSION", "").lower() in ("on", "1", "true"))
    if rt is not None and fusion_on:
        try:
            sys.path.insert(0, ROOT)
            import stress_common as _sc
            _sc.patch_fusion(rt, model)
            print(f"[t{tid}] CIPHER fusion patches applied", file=sys.stderr)
        except Exception as ex:
            print(f"[t{tid}] CIPHER fusion patch failed: {ex}", file=sys.stderr)

    BASE = ("Energy efficiency means doing more useful work per watt. "
            "The future of GPU computing is to make every joule count. ")
    ids_one = tok(BASE, return_tensors="pt").input_ids[0]
    n_tile  = (prefill_len + len(ids_one) - 1) // len(ids_one)
    # Batch-replicate the same prompt across `batch` rows. For B=1 this matches
    # prior behaviour; for B>1 we get genuine batched decode (B sequences
    # advance per step, so n_tokens = burst_len * batch).
    prompt  = ids_one.repeat(n_tile)[:prefill_len].unsqueeze(0).repeat(batch, 1).to("cuda:0")

    burst_us   = []
    n_tokens   = 0
    n_bursts   = 0
    t_start    = time.perf_counter()

    while time.perf_counter() - t_start < duration_s:
        MAX_LEN = prefill_len + burst_len + 4
        cache   = StaticCache(config=model.config, max_cache_len=MAX_LEN,
                              max_batch_size=batch)
        torch.cuda.synchronize()
        t_burst = time.perf_counter()
        with torch.no_grad():
            cp = torch.arange(prefill_len, device="cuda:0", dtype=torch.long)
            o  = model(input_ids=prompt, cache_position=cp,
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
        n_tokens += burst_len * batch
        n_bursts += 1

    elapsed = time.perf_counter() - t_start
    s = sorted(burst_us)
    n = len(s)
    def pct(p): return s[min(n-1, int(p*n))] if n else 0.0

    last_text = ""
    try:
        last_text = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)[:80]
    except Exception:
        pass

    json.dump(dict(
        tenant_id     = tid,
        n_tokens      = n_tokens,
        n_bursts      = n_bursts,
        elapsed_s     = elapsed,
        tokens_per_s  = n_tokens / elapsed if elapsed > 0 else 0.0,
        burst_p50_us  = pct(0.50),
        burst_p95_us  = pct(0.95),
        burst_p99_us  = pct(0.99),
        burst_mean_us = sum(s) / n if n else 0,
        sample_text   = last_text,
    ), open(out_path, "w"), indent=2)
    print(f"[t{tid}] {n_tokens} tokens in {elapsed:.1f}s "
          f"= {n_tokens/elapsed:.1f} tok/s, P50={pct(0.50):.0f}us")


if __name__ == "__main__":
    main()
