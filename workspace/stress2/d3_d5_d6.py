"""D3 — FP8 substitution rate.
D5 — MFU (decode B=8, prefill S=2048).
D6 — Persist convergence (PROMOTED count over 5000 tokens).
"""
import os, sys, json, time, ctypes, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress"))
import stress_common as sc

sc._setup_alloc_env()


class _PersistStats(ctypes.Structure):
    _fields_ = [
        ("regions_observed",   ctypes.c_uint64),
        ("regions_promoted",   ctypes.c_uint64),
        ("admit_calls",        ctypes.c_uint64),
        ("evict_calls",        ctypes.c_uint64),
        ("hot_ptr_hits",       ctypes.c_uint64),
        ("hot_ptr_misses",     ctypes.c_uint64),
        ("admitted_count",     ctypes.c_int),
        ("admitted_total_bytes", ctypes.c_size_t),
        ("admitted_window_bytes", ctypes.c_size_t),
        ("admit_fraction",     ctypes.c_float),
        ("hit_ratio",          ctypes.c_float),
        ("l2_persist_max",     ctypes.c_size_t),
    ]


def main():
    rt, fp8_stats, fus_stats = sc.init_cipher()
    rt.cipher_persist_engine_stats.argtypes = [ctypes.POINTER(_PersistStats)]
    rt.cipher_persist_engine_stats.restype = ctypes.c_int

    def persist_stats():
        s = _PersistStats()
        rt.cipher_persist_engine_stats(ctypes.byref(s))
        return dict(observed=s.regions_observed, promoted=s.regions_promoted,
                    admit_calls=s.admit_calls, hot_hits=s.hot_ptr_hits,
                    admitted_count=s.admitted_count,
                    admit_fraction=s.admit_fraction)

    import torch
    print("[D] loading Llama-3.1-8B fp16", flush=True)
    model, tok, _ = sc.load_model("cuda:0", patch=True, rt=rt)
    PROMPT = ("Energy efficiency means doing more useful work per watt. "
              "The future of GPU computing is to make every joule count. ")

    # ---- D3: 200-token decode B=1, count substituted vs passthrough ----
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    fp8_pre = fp8_stats(); pers_pre = persist_stats()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=attn, max_new_tokens=200,
                              do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    fp8_d3 = fp8_stats()
    d3 = dict(pre=fp8_pre, post=fp8_d3,
              delta_calls=fp8_d3["calls"] - fp8_pre["calls"],
              delta_passthroughs=fp8_d3["passthroughs"] - fp8_pre["passthroughs"],
              delta_failures=fp8_d3["failures"] - fp8_pre["failures"],
              delta_weights=fp8_d3["weights"] - fp8_pre["weights"])
    total = d3["delta_calls"] + d3["delta_passthroughs"] + d3["delta_failures"]
    sub_rate = d3["delta_calls"] / total if total > 0 else 0.0
    d3["substitution_rate"] = sub_rate
    print(f"[D3] sub={d3['delta_calls']} pass={d3['delta_passthroughs']} "
          f"fail={d3['delta_failures']} weights={d3['delta_weights']} "
          f"rate={sub_rate*100:.1f}%", flush=True)

    # ---- D5: prefill S=2048 ----
    print("[D5] prefill S=2048", flush=True)
    big_prompt_ids = ids[:, :1].repeat(1, 2048).contiguous()  # 2048 tokens
    big_attn = torch.ones_like(big_prompt_ids)
    # Untimed pass for cuBLAS workspace.
    with torch.no_grad():
        _ = model.generate(big_prompt_ids, attention_mask=big_attn,
                            max_new_tokens=2, do_sample=False,
                            pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    samples = []; stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.05), daemon=True)
    th.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(big_prompt_ids, attention_mask=big_attn,
                            max_new_tokens=4, do_sample=False,
                            pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    pw = [s[1] for s in samples] or [0]
    mean_w = sum(pw)/len(pw)
    # Prefill counts 2048 tokens prefilled per call; max_new_tokens=4 → discount
    # by deducting decode time. For approximate prefill measurement, use ratio.
    prefill_tokens = 2048
    prefill_tps = prefill_tokens / elapsed
    mfu_prefill = (45e9 * prefill_tps) / 989e12 * 100
    d5_prefill = dict(seq=2048, elapsed_s=elapsed, prefill_tps=prefill_tps,
                       mean_w=mean_w, mfu_pct=mfu_prefill)
    print(f"[D5 prefill S=2048] tps={prefill_tps:.1f} W={mean_w:.0f} "
          f"MFU={mfu_prefill:.2f}%", flush=True)

    # ---- D6: 5000-token decode, persist promotions over time ----
    print("[D6] 5000-token decode, sampling persist promotions", flush=True)
    pers_at = {}
    pers_at[0] = persist_stats()
    # Untimed warmup to allocate workspace.
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn, max_new_tokens=4,
                            do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    pers_at[1] = persist_stats()  # post-warmup
    t0 = time.perf_counter()
    # Generate 100 -> snapshot, 1000 -> snapshot, 5000 -> snapshot
    waypoints = [100, 1000, 5000]
    cur = 0
    last_ids = ids
    with torch.no_grad():
        for wp in waypoints:
            n_to_gen = wp - cur
            cur_attn = torch.ones_like(last_ids)
            out = model.generate(last_ids, attention_mask=cur_attn,
                                 max_new_tokens=n_to_gen, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            pers_at[wp] = persist_stats()
            last_ids = out
            cur = wp
            print(f"[D6 @ {wp:>5} tokens] {pers_at[wp]}", flush=True)
    elapsed = time.perf_counter() - t0
    d6 = dict(elapsed_s=elapsed, pers_at=pers_at)

    out_path = os.path.join(os.path.dirname(__file__), "d3_d5_d6.json")
    with open(out_path, "w") as f:
        json.dump(dict(d3=d3, d5_prefill=d5_prefill, d6=d6), f, indent=2,
                  default=str)
    print(f"[D] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
