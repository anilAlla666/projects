"""ACTION 2 (continued): time the ACTUAL spec-decode shape: batch=1, seq_len=9, 2K KV cache.

This is what speculative decoding really does: feed K drafted tokens (plus 1 for the
post-draft position) into ONE sequence's forward, with the KV cache loaded once.

Compare to batch=9, seq_len=1 (the parallelism case from spec_batch9_timing.py).
"""
import os, time, json, threading, subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
CTX = 2048
N_ITERS = 100
WARMUP = 20
P_MATCH = 0.59
ACTIVE_W_ASSUMED = 231.6
B1_TOKW = 57.1 / 231.6   # 0.247

def time_forward(model, input_ids, past, n_iters=N_ITERS):
    samples = []
    stop = threading.Event()
    def power_sampler():
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=1)
                parts = r.stdout.strip().split(", ")
                if len(parts) == 2:
                    samples.append((float(parts[0]), int(parts[1])))
            except: pass
            time.sleep(0.1)
    th = threading.Thread(target=power_sampler, daemon=True)
    th.start()

    nxt = input_ids
    p = past
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=p, use_cache=True)
            p = out.past_key_values
        # Reset to original past for next iter (don't keep growing the cache)
        # Actually for timing we want the marginal cost; KV cache grows. So we re-set past from initial each iter.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    powers = [s[0] for s in samples[1:]]
    return (elapsed / n_iters * 1e3,
            sum(powers)/len(powers) if powers else ACTIVE_W_ASSUMED,
            len(powers))


def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    cfg = model.config

    # Configurations to measure
    print(f"\n[measure] each: prefill 2K, then time forward with the given (batch, seq) shape\n", flush=True)
    print(f"  {'shape':<22} {'ms':>7} {'tok/cycle':>10} {'tps_eff':>10} {'power_W':>9} {'tok/W':>7} {'×base':>6}", flush=True)
    print("  " + "-" * 74, flush=True)

    configs = [
        # (batch, seq, label)
        (1, 1, "batch=1 seq=1 (baseline)"),
        (1, 9, "batch=1 seq=9 (REAL spec decode)"),
        (9, 1, "batch=9 seq=1 (parallelism)"),
        (1, 5, "batch=1 seq=5 (K=4 drafts)"),
        (1, 17, "batch=1 seq=17 (K=16 drafts)"),
    ]

    results = []
    for batch, seq, label in configs:
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        # Prefill once, then time forwards of given seq_len
        prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(batch, CTX), device="cuda")
        with torch.no_grad():
            out = model(input_ids=prompt, use_cache=True)
            past = out.past_key_values
        torch.cuda.synchronize()
        # Build the next-token input of shape [batch, seq]
        nxt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(batch, seq), device="cuda")
        # Warmup (each call grows KV cache by `seq` — past must NOT keep growing across all 100 iters,
        # so we re-prefill for each timed iter; but that's expensive. Compromise: take current past
        # and feed nxt N times into a *fresh* past each iter is too slow. Instead, do a SINGLE prefill
        # and time a SHORT bursty loop where the cache grows by seq each call.
        # Cap total iters so cache doesn't blow past max position.
        # For Mistral max=32K we have headroom: 100 * seq ≤ 1700 extra tokens.
        for _ in range(WARMUP):
            with torch.no_grad():
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
        torch.cuda.synchronize()

        # Sample power during timed run
        samples = []
        stop = threading.Event()
        def power_sampler():
            while not stop.is_set():
                try:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=1)
                    parts = r.stdout.strip().split(", ")
                    if len(parts) == 2:
                        samples.append((float(parts[0]), int(parts[1])))
                except: pass
                time.sleep(0.1)
        th = threading.Thread(target=power_sampler, daemon=True)
        th.start()

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            with torch.no_grad():
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)

        ms_per = elapsed / N_ITERS * 1e3
        powers = [s[0] for s in samples[1:]]
        avg_pw = sum(powers)/len(powers) if powers else ACTIVE_W_ASSUMED

        # Effective tokens/second
        # Under spec decode interpretation:
        #   batch=1 seq=K+1: produces K+1 logits per cycle. Markov-accepted prefix
        #     E[accepted] = 1 + sum_{k=1..K} p^k = 1 + p(1-p^K)/(1-p)
        #   batch=B seq=1:  produces B independent sequences' next-token. Treat as B parallel sequences;
        #     effective tokens per second = B / cycle_time
        if seq == 1:
            tok_per_cycle = batch  # parallel sequences
        else:
            K = seq - 1
            p = P_MATCH
            tok_per_cycle = 1 + sum(p**k for k in range(1, K + 1))
        tps_eff = tok_per_cycle / (ms_per / 1000)
        tokw    = tps_eff / avg_pw if avg_pw > 0 else 0
        x_base  = tokw / B1_TOKW

        print(f"  {label:<22} {ms_per:>7.2f} {tok_per_cycle:>10.2f} {tps_eff:>10.1f} {avg_pw:>9.1f} {tokw:>7.3f} {x_base:>5.2f}x", flush=True)
        results.append({
            "batch": batch, "seq": seq, "label": label,
            "ms_per": ms_per, "active_W": avg_pw,
            "tok_per_cycle": tok_per_cycle, "tps_eff": tps_eff,
            "tokw": tokw, "x_baseline": x_base,
        })

        del past
        torch.cuda.empty_cache()

    # Save
    with open("/home/ubuntu/op31-prod-fix/spec_seq9_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] -> spec_seq9_results.json", flush=True)


if __name__ == "__main__":
    main()
