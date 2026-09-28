"""Three measurements for n-gram speculative decoding feasibility on Mistral-7B.

M1: n-gram match rate (zero-cost draft prediction)
M2: batched forward pass overhead (verification cost)
M3: idle vs active power (the tok/W headroom)
"""
import os, sys, time, json, threading, subprocess
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE = 500
NGRAM_RANGE = [2, 3, 4, 5, 6]
PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
]


def m1_ngram_match(model, tok):
    print(f"\n{'='*72}\nM1: n-gram match rate ({len(PROMPTS)} prompts x {N_DECODE} decode tokens)\n{'='*72}", flush=True)

    # Per-prompt: list of decoded token IDs (prompt + 500 decoded)
    all_results = []
    for p_idx, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_len = ids.shape[1]
        seq = ids[0].tolist()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq.append(nxt.item())
            for s in range(N_DECODE - 1):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                seq.append(nxt.item())
        # seq has prompt_len + N_DECODE tokens; the "decoded" tokens are seq[prompt_len..]
        all_results.append({"prompt_len": prompt_len, "tokens": seq})
        print(f"  prompt {p_idx+1}/{len(PROMPTS)}: prompt_len={prompt_len} total_len={len(seq)}", flush=True)

    # n-gram match analysis
    # For each decoded position t (where t >= prompt_len + n), we look at the last n-1 tokens
    # before position t (i.e., seq[t-n+1 : t]), search backwards through seq[0..t-n+1] for that
    # (n-1)-gram, take the LAST match's successor token, and check if it equals seq[t].
    print(f"\n  {'n':>3}  {'positions_evaluated':>21}  {'matches_found':>15}  {'correct_drafts':>16}  {'match_rate':>11}  {'accuracy_when_matched':>22}", flush=True)
    print("  " + "-" * 100, flush=True)

    summary = []
    for n in NGRAM_RANGE:
        total_positions = 0
        n_match_found = 0
        n_match_correct = 0
        per_prompt_stats = []
        for r in all_results:
            seq = r["tokens"]
            # We can analyze every position t >= n where t < len(seq) and t > prompt_len
            # (so we're predicting decoded tokens, and we need n-1 history tokens)
            for t in range(max(n, r["prompt_len"]), len(seq)):
                key = tuple(seq[t - (n - 1): t])  # last n-1 tokens before t
                # Search backwards from t-n+1 down to 0 for the (n-1)-gram followed by anything
                # We want positions i where seq[i:i+n-1] == key, and seq[i+n-1] is the successor.
                # Take the LAST such i (largest i strictly < t-n+1, so i+n-1 < t).
                draft = None
                # Iterate i from t-n down to 0 (so the candidate match ends at <= t-1, successor at <= t-1; wait we want successor index i+n-1 < t, so i < t-n+1 → i <= t-n)
                for i in range(t - n, -1, -1):
                    if tuple(seq[i: i + n - 1]) == key:
                        draft = seq[i + n - 1]
                        break
                total_positions += 1
                if draft is not None:
                    n_match_found += 1
                    if draft == seq[t]:
                        n_match_correct += 1

        match_rate = n_match_correct / total_positions if total_positions > 0 else 0.0
        find_rate  = n_match_found / total_positions if total_positions > 0 else 0.0
        accuracy_given_match = n_match_correct / n_match_found if n_match_found > 0 else 0.0
        print(f"  {n:>3}  {total_positions:>21}  {n_match_found:>15}  {n_match_correct:>16}  "
              f"{match_rate:>10.4f}  {accuracy_given_match:>22.4f}", flush=True)
        summary.append({
            "n": n,
            "positions": total_positions,
            "found": n_match_found,
            "correct": n_match_correct,
            "match_rate": match_rate,
            "find_rate": find_rate,
            "accuracy_given_match": accuracy_given_match,
        })

    # What does the BEST n give us per position (any n that succeeded)?
    print(f"\n  Best-of-n (any n in 2..6 finds a match AND draft is correct):", flush=True)
    best_total = 0
    best_correct = 0
    for r in all_results:
        seq = r["tokens"]
        for t in range(max(NGRAM_RANGE[-1], r["prompt_len"]), len(seq)):
            best_total += 1
            for n in reversed(NGRAM_RANGE):  # prefer longer matches
                if t < n: continue
                key = tuple(seq[t - (n - 1): t])
                for i in range(t - n, -1, -1):
                    if tuple(seq[i: i + n - 1]) == key:
                        if seq[i + n - 1] == seq[t]:
                            best_correct += 1
                        break
                else:
                    continue
                break
    print(f"    {best_correct}/{best_total} = {best_correct/best_total:.4f} (longest-match-first)", flush=True)

    return {"per_n": summary, "best_total": best_total, "best_correct": best_correct,
            "best_rate": best_correct / best_total if best_total else 0.0}


def m2_batched_forward(model):
    print(f"\n{'='*72}\nM2: batched forward overhead at 2K context\n{'='*72}", flush=True)
    cfg = model.config

    BATCH_SIZES = [1, 2, 4, 8, 16]
    CTX_LEN = 2048
    SEQ_LEN = 1
    N_ITERS = 100

    print(f"  context_len={CTX_LEN}  seq_len={SEQ_LEN}  iters/measurement={N_ITERS}", flush=True)
    print(f"\n  {'batch':>6}  {'ms_per_call':>12}  {'us_per_call':>12}  {'ratio_vs_b1':>11}  {'tokens/sec':>11}  {'tokens/sec/W*':>14}", flush=True)
    print("  " + "-" * 80, flush=True)

    results = []
    base_ms = None
    for B in BATCH_SIZES:
        try:
            # Build a batch of B sequences of CTX_LEN random tokens, prefill, then time seq_len=1 calls
            torch.manual_seed(0)
            prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(B, CTX_LEN), device="cuda")
            with torch.no_grad():
                out = model(input_ids=prompt, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            torch.cuda.synchronize()

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(N_ITERS):
                with torch.no_grad():
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            ms_per = elapsed / N_ITERS * 1e3
            us_per = ms_per * 1e3
            tokens_per_sec = B / (elapsed / N_ITERS)
            if base_ms is None:
                base_ms = ms_per
            ratio = ms_per / base_ms
            results.append({"batch": B, "ms_per_call": ms_per, "tokens_per_sec": tokens_per_sec, "ratio_vs_b1": ratio})
            print(f"  {B:>6}  {ms_per:>12.3f}  {us_per:>12.1f}  {ratio:>10.3f}x  {tokens_per_sec:>11.1f}  {'see M3':>14}", flush=True)
            del past, prompt, nxt
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  {B:>6}  OOM ({e})", flush=True)
            results.append({"batch": B, "ms_per_call": None, "ratio_vs_b1": None})
            torch.cuda.empty_cache()

    return results


def m3_power(model):
    print(f"\n{'='*72}\nM3: idle vs active power\n{'='*72}", flush=True)

    def sample_power(label, duration_s):
        samples = []
        stop = threading.Event()
        def loop():
            while not stop.is_set():
                try:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.gr",
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=1)
                    parts = r.stdout.strip().split(", ")
                    if len(parts) == 2:
                        samples.append((float(parts[0]), int(parts[1])))
                except Exception:
                    pass
                time.sleep(0.1)
        th = threading.Thread(target=loop, daemon=True)
        th.start()
        time.sleep(duration_s)
        stop.set()
        th.join(timeout=2)
        return samples

    # Idle: do nothing for 5 seconds (after a sync to drain any pending ops)
    torch.cuda.synchronize()
    time.sleep(2.0)  # let GPU settle
    idle_samples = sample_power("idle", 5.0)
    if idle_samples:
        # Skip first sample (boot/transient)
        ps = [s[0] for s in idle_samples[1:]]
        cs = [s[1] for s in idle_samples[1:]]
        idle_pw = sum(ps) / len(ps)
        idle_clk = sum(cs) / len(cs)
        print(f"  idle:    {idle_pw:>6.1f} W   clock {idle_clk:>5.0f} MHz   ({len(ps)} samples over 5s)", flush=True)
    else:
        idle_pw = 0; idle_clk = 0

    # Active: run continuous forward passes for 5 seconds
    torch.manual_seed(0)
    cfg = model.config
    prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(1, 2048), device="cuda")
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    stop_active = threading.Event()
    def busy_loop():
        nonlocal past, nxt
        while not stop_active.is_set():
            with torch.no_grad():
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    busy_th = threading.Thread(target=busy_loop, daemon=True)
    busy_th.start()
    time.sleep(0.5)  # let it ramp
    active_samples = sample_power("active", 5.0)
    stop_active.set()
    busy_th.join(timeout=2)
    torch.cuda.synchronize()
    if active_samples:
        ps = [s[0] for s in active_samples[1:]]
        cs = [s[1] for s in active_samples[1:]]
        active_pw = sum(ps) / len(ps)
        active_clk = sum(cs) / len(cs)
        print(f"  active:  {active_pw:>6.1f} W   clock {active_clk:>5.0f} MHz   ({len(ps)} samples over 5s)", flush=True)
    else:
        active_pw = 0; active_clk = 0

    delta = active_pw - idle_pw
    print(f"\n  delta:   {delta:>6.1f} W  ({delta/active_pw*100 if active_pw else 0:.1f}% of active draw)", flush=True)
    return {"idle_W": idle_pw, "active_W": active_pw, "idle_MHz": idle_clk, "active_MHz": active_clk}


def main():
    print(f"[load] {MODEL}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    m1 = m1_ngram_match(model, tok)
    m2 = m2_batched_forward(model)
    m3 = m3_power(model)

    # Summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}", flush=True)
    print("M1 best-of-n match rate (longest match first):"
          f"  {m1['best_rate']:.4f} ({m1['best_correct']}/{m1['best_total']})", flush=True)
    print(f"M2 batch=1 ms: {m2[0]['ms_per_call']:.3f}", flush=True)
    if len(m2) >= 5 and m2[4].get('ms_per_call'):
        print(f"M2 batch=16 ms: {m2[4]['ms_per_call']:.3f}  (ratio {m2[4]['ratio_vs_b1']:.2f}x)", flush=True)
    print(f"M3 idle: {m3['idle_W']:.1f} W  active: {m3['active_W']:.1f} W  delta {m3['active_W']-m3['idle_W']:.1f} W", flush=True)

    out_path = "/home/ubuntu/op31-prod-fix/spec_decode_measurements.json"
    with open(out_path, "w") as f:
        json.dump({"M1": m1, "M2": m2, "M3": m3}, f, indent=2)
    print(f"\n[save] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
