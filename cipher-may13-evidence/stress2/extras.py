"""Extra realistic neocloud-relevant tests.

X1 — tail latency under burst:
    100 sequential generate(64 tok) requests on Llama-3.1-8B.
    Report P50, P95, P99, max latency.

X2 — long context (32K and 128K):
    prompt = 32K (and 128K if memory allows) tokens, generate 32 new tokens.
    Verifies the FP8 + fusion path doesn't break at long context.

X3 — KV / prefix reuse simulation:
    Same long prefix (1K tokens) shared across 32 generate() calls each
    extending it with 16 new tokens. CIPHER vs baseline tps.

X4 — power-cap excursion:
    sudo nvidia-smi -pl 250 (force throttle), generate 200 tokens, restore.
    No crash, output coherent.
"""
import os, sys, json, time, subprocess, ctypes, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()
USE_CIPHER = os.environ.get("CIPHER", "1") != "0"


def x1_tail_latency():
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    print("[X1] loading model", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    PROMPT = "The future of GPU computing is"
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    # Warmup x3
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    lat = []
    for i in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(ids, attention_mask=attn, max_new_tokens=64,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    lat_sorted = sorted(lat)
    p50 = lat_sorted[50]
    p95 = lat_sorted[94]
    p99 = lat_sorted[98]
    return dict(n=100, p50_ms=p50, p95_ms=p95, p99_ms=p99,
                max_ms=max(lat), min_ms=min(lat))


def x2_long_context(prefix_tokens):
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[X2-{prefix_tokens}] loading model", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    base_ids = tok("The future of GPU computing is to make every joule count. ",
                   return_tensors="pt").input_ids[0]
    n_tile = (prefix_tokens + len(base_ids) - 1) // len(base_ids)
    big = base_ids.repeat(n_tile)[:prefix_tokens].unsqueeze(0).to("cuda:0")
    attn = torch.ones_like(big)
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model.generate(big, attention_mask=attn, max_new_tokens=32,
                                  do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        text = tok.decode(out[0, big.shape[1]:], skip_special_tokens=False)
        return dict(prefix_tokens=prefix_tokens, ok=True, elapsed_s=elapsed,
                    new_tokens=int(out.shape[1] - big.shape[1]),
                    coherent=("!!!!" not in text and len(text.strip()) > 3),
                    text=text[:120])
    except torch.cuda.OutOfMemoryError as e:
        return dict(prefix_tokens=prefix_tokens, ok=False, error="OOM",
                    detail=str(e)[:200])
    except Exception as e:
        return dict(prefix_tokens=prefix_tokens, ok=False,
                    error=f"{type(e).__name__}: {str(e)[:200]}")


def x3_prefix_reuse():
    """Same 1K prefix, 32 generate() calls each adding 16 tokens.
    Measures wall-time stability across calls."""
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    import torch
    model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    base_ids = tok("Energy efficiency is critical for modern AI systems. " * 64,
                    return_tensors="pt").input_ids
    big = base_ids[:, :1024].to("cuda:0")
    attn = torch.ones_like(big)
    times = []
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(big, attention_mask=attn, max_new_tokens=4,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        for i in range(32):
            t0 = time.perf_counter()
            _ = model.generate(big, attention_mask=attn, max_new_tokens=16,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return dict(n_calls=len(times), mean_s=sum(times)/len(times),
                p50_s=sorted(times)[len(times)//2],
                p99_s=sorted(times)[-1])


def x4_power_cap_excursion():
    """Drop power cap to 250W under load, verify no crash + coherent output."""
    rt, _, _ = (sc.init_cipher() if USE_CIPHER
                else (None, lambda: {}, lambda: {}))
    # Cap power.
    rcap = subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0",
                           "-pl", "250"], capture_output=True, text=True, timeout=10)
    pl_set = (rcap.returncode == 0)
    import torch
    try:
        model, tok, _ = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
        ids = tok("The future of GPU computing is", return_tensors="pt").input_ids.to("cuda:0")
        attn = torch.ones_like(ids)
        with torch.no_grad():
            _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(ids, attention_mask=attn, max_new_tokens=200,
                                  do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        coherent = "!!!!" not in text
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {str(e)[:200]}"
        coherent = None
        elapsed = 0
        text = ""
    finally:
        # Restore default cap.
        subprocess.run(["sudo", "-n", "nvidia-smi", "-i", "0", "-pl", "700"],
                       capture_output=True, text=True, timeout=10)
    return dict(power_cap_set=pl_set, ok=ok, error=err,
                coherent=coherent, elapsed_s=elapsed,
                first_text=text[:120])


def main():
    which = sys.argv[1]
    print(f"[{which}] running", flush=True)
    fn = {"X1": x1_tail_latency, "X2_32K": lambda: x2_long_context(32768),
          "X2_128K": lambda: x2_long_context(131072),
          "X3": x3_prefix_reuse, "X4": x4_power_cap_excursion}[which]
    out = fn()
    out["test"] = which
    out["cipher"] = USE_CIPHER
    suffix = "_baseline" if not USE_CIPHER else ""
    with open(os.path.join(os.path.dirname(__file__),
                            f"{which.lower()}{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{which}] {out}", flush=True)


if __name__ == "__main__":
    main()
