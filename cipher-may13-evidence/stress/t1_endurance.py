"""Test 1 — 10K-token continuous generation, single client.

Single model.generate() call with max_new_tokens=10_000, do_sample=False.
A background sampler records (t, watts, total_mem_MiB) every ~150 ms;
this also surfaces nvidia-smi every 30 s.

Outputs:
    stress/t1_endurance.json  — full timing/power/memory series and decoded
                                strings at token 1, 5000, 10000.
"""
import os, sys, json, time, threading, gc
sys.path.insert(0, os.path.dirname(__file__))
import stress_common as sc

sc._setup_alloc_env()

USE_CIPHER = os.environ.get("CIPHER", "1") != "0"
USE_FP8    = os.environ.get("STRESS_FP8", "1") != "0"
USE_FUSION = os.environ.get("STRESS_FUSION", "1") != "0"
N_TOKENS = int(os.environ.get("STRESS_T1_TOKENS", "10000"))
PROMPT = ("Energy efficiency means doing more useful work per watt. "
          "The future of GPU computing is to make every joule count. ")


def main():
    rt, fp8_stats, fus_stats = (
        sc.init_cipher(use_fp8=USE_FP8, use_fusion=USE_FUSION)
        if USE_CIPHER else (None, lambda: {}, lambda: {}))
    import torch
    print(f"[t1] torch={torch.__version__} cuda={torch.version.cuda} "
          f"cipher={USE_CIPHER} tokens={N_TOKENS}", flush=True)

    model, tok, load_s = sc.load_model("cuda:0", patch=USE_CIPHER, rt=rt)
    print(f"[t1] loaded in {load_s:.1f}s; "
          f"alloc={torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    attn = torch.ones_like(ids)

    # Quick warmup so cuBLAS workspace is allocated before timing starts.
    with torch.no_grad():
        _ = model.generate(ids, attention_mask=attn,
                           max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()

    fp8_pre = fp8_stats(); fus_pre = fus_stats()
    pre_alloc_gb = torch.cuda.memory_allocated(0) / 1e9
    pre_smi_mib = sc.nvsmi_mem_used_mib(0)
    print(f"[t1] pre-gen alloc={pre_alloc_gb:.2f}GB nvsmi={pre_smi_mib}MiB",
          flush=True)

    samples = []
    stop = threading.Event()
    th = threading.Thread(target=sc.power_sampler,
                          args=(stop, samples, 1, 0.15), daemon=True)
    th.start()

    nvsmi_log = []  # (elapsed_s, alloc_gb, nvsmi_mib, watts)

    def heartbeat():
        t0_hb = time.time()
        next_log = t0_hb + 30.0
        while not stop.is_set():
            now = time.time()
            if now >= next_log:
                try:
                    alloc = torch.cuda.memory_allocated(0) / 1e9
                    smi = sc.nvsmi_mem_used_mib(0)
                    last_w = samples[-1][1] if samples else 0
                    nvsmi_log.append(dict(elapsed_s=now - t0_hb,
                                          alloc_gb=alloc,
                                          nvsmi_mib=smi,
                                          watts=last_w))
                    print(f"[t1 hb {now-t0_hb:6.1f}s] alloc={alloc:.2f}GB "
                          f"nvsmi={smi}MiB W={last_w:.1f}", flush=True)
                except Exception as e:
                    print(f"[t1 hb err] {e}", flush=True)
                next_log += 30.0
            time.sleep(0.5)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn,
                                 max_new_tokens=N_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        crash = None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        crash = f"{type(e).__name__}: {str(e)[:300]}"
        out = None
    stop.set(); th.join(timeout=2); hb.join(timeout=2)

    fp8_post = fp8_stats(); fus_post = fus_stats()
    post_alloc_gb = torch.cuda.memory_allocated(0) / 1e9
    post_smi_mib = sc.nvsmi_mem_used_mib(0)

    n_new = (out.shape[1] - ids.shape[1]) if out is not None else 0
    tps = n_new / elapsed if elapsed > 0 else 0.0
    pw = [s[1] for s in samples[3:]] or [0]
    mean_w = sum(pw)/len(pw)
    tok_w = tps / mean_w if mean_w > 0 else 0.0

    # Coherence checkpoints: tok 1, 5000, 10000
    decoded_at = {}
    if out is not None:
        new_ids = out[0, ids.shape[1]:]
        for label, k in [("tok_1", 1), ("tok_5000", 5000), ("tok_10000", 10000)]:
            if n_new >= k:
                # 30 tokens of context at the checkpoint
                lo = max(0, k - 15); hi = min(n_new, k + 15)
                decoded_at[label] = tok.decode(new_ids[lo:hi],
                                               skip_special_tokens=False)

    # Tok/W per minute window — drift check.
    win_s = 60.0
    if samples:
        t_start = samples[0][0]
        # We don't know per-token timestamps cheaply, but we can split
        # the run into N windows and report mean-watts per window.  The
        # token rate inside each window is tps_overall (since generate
        # is uninterruptible).
        windows = []
        i = 0
        cur_lo = t_start
        cur_hi = t_start + win_s
        while cur_lo < samples[-1][0]:
            ws = [s[1] for s in samples if cur_lo <= s[0] < cur_hi]
            ms = [s[2] for s in samples if cur_lo <= s[0] < cur_hi]
            if ws:
                windows.append(dict(window_lo_s=cur_lo - t_start,
                                    window_hi_s=cur_hi - t_start,
                                    mean_w=sum(ws)/len(ws),
                                    max_mem_mib=max(ms) if ms else 0,
                                    n_samples=len(ws)))
            cur_lo = cur_hi; cur_hi += win_s
    else:
        windows = []

    result = dict(
        cipher=USE_CIPHER,
        target_tokens=N_TOKENS, generated=n_new, elapsed_s=elapsed,
        tps=tps, mean_w=mean_w, tok_w=tok_w,
        crash=crash,
        pre_alloc_gb=pre_alloc_gb, post_alloc_gb=post_alloc_gb,
        pre_nvsmi_mib=pre_smi_mib, post_nvsmi_mib=post_smi_mib,
        mem_growth_mib=post_smi_mib - pre_smi_mib,
        nvsmi_30s=nvsmi_log, windows_60s=windows,
        decoded=decoded_at,
        fp8_delta={k: fp8_post.get(k, 0) - fp8_pre.get(k, 0)
                   for k in fp8_pre} if fp8_pre else {},
        fusion_delta={k: fus_post.get(k, 0) - fus_pre.get(k, 0)
                      for k in fus_pre} if fus_pre else {},
    )
    suffix = os.environ.get("STRESS_SUFFIX",
                            "_baseline" if not USE_CIPHER else "")
    out_path = os.path.join(os.path.dirname(__file__),
                            f"t1_endurance{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[t1] generated={n_new}/{N_TOKENS} in {elapsed:.1f}s "
          f"tps={tps:.1f} W={mean_w:.1f} tok/W={tok_w:.3f} "
          f"mem_growth={post_smi_mib - pre_smi_mib}MiB crash={crash}",
          flush=True)
    print(f"[t1] wrote {out_path}", flush=True)
    for label, txt in decoded_at.items():
        print(f"  {label}: {txt!r}", flush=True)


if __name__ == "__main__":
    main()
