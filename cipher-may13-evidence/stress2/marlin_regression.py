"""GPU 0 — regression of T1, T3, X1 with Marlin + system torch 2.7.

For each test we report PASS / FAIL against the SCORECARD numbers:
   T1  : 200-tok endurance, coherent at tok 1
   T3  : 100 deterministic runs, all match
   X1  : 100-burst tail latency, P99 < baseline P99 (3989 ms)
"""
import os, sys, json, time, ctypes, threading, hashlib, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc
sys.path.insert(0, os.path.dirname(__file__))
from c2_marlin import setup_rt, patch_model, Counters

sc._setup_alloc_env()


def run_t1():
    """200 tokens single-client, coherent + tok/W."""
    rt = setup_rt()
    import torch
    model, tok, _ = sc.load_model("cuda:0", patch=False, rt=None)
    n_c, _ = patch_model(rt, model)
    PROMPT = ("Energy efficiency means doing more useful work per watt. "
              "The future of GPU computing is to make every joule count. ")
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn, max_new_tokens=200,
                              do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    n = out.shape[1] - ids.shape[1]
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    tps = n / elapsed
    coherent = ("!!!!" not in text) and len(text.strip()) > 30
    pass_ = coherent and (n == 200)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return dict(test="T1", pass_=pass_, n=n, tps=tps, mean_w=mean_w,
                tok_w=tps/mean_w if mean_w else 0,
                marlin_calls=Counters.marlin,
                first_text=text[:120])


def run_t3():
    """100 runs of same prompt, same first6 tokens."""
    rt = setup_rt()
    import torch
    model, tok, _ = sc.load_model("cuda:0", patch=False, rt=None)
    n_c, _ = patch_model(rt, model)
    PROMPT = "The future of GPU computing is"
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=8,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    hashes = []
    for i in range(100):
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn, max_new_tokens=64,
                                 do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        seq = out[0, ids.shape[1]:].tolist()
        h = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]
        hashes.append(h)
        del out
    intra_match = sum(1 for h in hashes if h == hashes[0])
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return dict(test="T3", pass_=(intra_match == 100), intra_match=intra_match,
                unique=len(set(hashes)))


def run_x1():
    """100 sequential generate(64) calls; report P50/95/99/max latency."""
    rt = setup_rt()
    import torch
    model, tok, _ = sc.load_model("cuda:0", patch=False, rt=None)
    n_c, _ = patch_model(rt, model)
    PROMPT = "The future of GPU computing is"
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(ids, attention_mask=attn, max_new_tokens=64,
                                do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(lat)
    p50, p95, p99 = s[50], s[94], s[98]
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    pass_ = (p99 < 3989.0)
    return dict(test="X1", pass_=pass_, p50_ms=p50, p95_ms=p95, p99_ms=p99,
                max_ms=max(lat), min_ms=min(lat))


if __name__ == "__main__":
    which = sys.argv[1]
    fn = {"T1": run_t1, "T3": run_t3, "X1": run_x1}[which]
    print(f"[Marlin-regression] running {which}", flush=True)
    out = fn()
    out_path = os.path.join(os.path.dirname(__file__),
                             f"marlin_regression_{which.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Marlin-regression] {which}: {out}", flush=True)
