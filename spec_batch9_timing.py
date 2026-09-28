"""ACTION 2: time Mistral-7B forward at batch=9 with 2K KV context.

Then compute tok/W under both spec-decode formulas:
  USER (independent acceptance):     1 + 0.59 * 8 = 5.72 tokens/cycle
  STANDARD (sequential, Markov):     1 + sum_{k=1..8} 0.59^k = 2.42 tokens/cycle
"""
import os, sys, time, json, threading, subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
CTX = 2048
N_ITERS = 100
WARMUP = 20

# From M3 / spec_decode_measure.py:
ACTIVE_W = 231.6
IDLE_W   = 115.1
P_MATCH  = 0.59  # n-gram best-of-n match rate

# From M2 / spec_decode_measure.py: batch=1 baseline
B1_MS    = 17.499
B1_TPS   = 57.1   # baseline tok/s
B1_TOKW  = B1_TPS / ACTIVE_W   # 0.247

def main():
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    # Time batch=9 forward at 2K context
    print(f"\n[time] batch=9, ctx_len={CTX}, seq_len=1, iters={N_ITERS}", flush=True)
    cfg = model.config
    torch.manual_seed(0)
    prompt = torch.randint(low=10, high=cfg.vocab_size - 10, size=(9, CTX), device="cuda")
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    for _ in range(WARMUP):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Sample power during the timed run
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
            except Exception: pass
            time.sleep(0.1)
    th = threading.Thread(target=power_sampler, daemon=True)
    th.start()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        with torch.no_grad():
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    b9_ms = elapsed / N_ITERS * 1e3
    powers = [s[0] for s in samples[1:]]
    measured_active_W = sum(powers)/len(powers) if powers else ACTIVE_W

    print(f"[time]   batch=9 forward: {b9_ms:.3f} ms  (vs batch=1 = {B1_MS:.3f} ms, ratio {b9_ms/B1_MS:.3f}x)", flush=True)
    print(f"[time]   active power during run: {measured_active_W:.1f} W  ({len(powers)} samples)", flush=True)

    # ---- Compute tok/W projections ----
    print(f"\n=== Spec-decode tok/W projection (K=8 drafts, p={P_MATCH}) ===", flush=True)
    print(f"  Constants:", flush=True)
    print(f"    batch=1 baseline:        {B1_TPS:.1f} tok/s  /  {ACTIVE_W:.1f} W  =  {B1_TOKW:.3f} tok/W", flush=True)
    print(f"    batch=9 forward time:    {b9_ms:.2f} ms", flush=True)
    print(f"    active power:            {measured_active_W:.1f} W", flush=True)
    print(f"    n-gram match rate p:     {P_MATCH}", flush=True)
    print(f"    K (draft length):        8", flush=True)
    print(flush=True)

    # User's independent-acceptance formula
    accepted_user = 1 + P_MATCH * 8
    tps_user      = accepted_user / (b9_ms / 1000)
    tokw_user     = tps_user / measured_active_W
    print(f"  USER formula (independent acceptance — optimistic upper bound):", flush=True)
    print(f"    expected accepted/cycle: 1 + {P_MATCH} * 8 = {accepted_user:.2f} tokens", flush=True)
    print(f"    throughput:              {tps_user:.1f} tok/s", flush=True)
    print(f"    tok/W:                   {tokw_user:.3f}  ({tokw_user/B1_TOKW:.2f}x baseline)", flush=True)
    print(flush=True)

    # Standard sequential-acceptance Markov formula
    p = P_MATCH
    accepted_markov = 1 + sum(p**k for k in range(1, 9))   # 1 + p(1-p^8)/(1-p)
    tps_markov     = accepted_markov / (b9_ms / 1000)
    tokw_markov    = tps_markov / measured_active_W
    print(f"  STANDARD formula (sequential acceptance — what real spec decoders deliver):", flush=True)
    print(f"    expected accepted/cycle: 1 + sum(p^k, k=1..8) = {accepted_markov:.3f} tokens", flush=True)
    print(f"    throughput:              {tps_markov:.1f} tok/s", flush=True)
    print(f"    tok/W:                   {tokw_markov:.3f}  ({tokw_markov/B1_TOKW:.2f}x baseline)", flush=True)
    print(flush=True)

    # Why the difference matters
    print(f"  Reason for the gap:", flush=True)
    print(f"    Spec decode in production REJECTS draft k as soon as draft k-1 is wrong.", flush=True)
    print(f"    So expected accepted prefix is the geometric series, not K * p.", flush=True)
    print(f"    The user's formula assumes drafts are evaluated independently — true only if", flush=True)
    print(f"    the verifier returns ALL K conditional logits (which it does in batched verification)", flush=True)
    print(f"    AND the draft tree is a tree (Medusa-style) rather than a single chain.", flush=True)
    print(f"    For single-chain n-gram drafts, the Markov formula is what you actually get.", flush=True)
    print(flush=True)

    # Save
    out_path = "/home/ubuntu/op31-prod-fix/spec_batch9_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "batch9_ms": b9_ms,
            "batch1_ms": B1_MS,
            "active_W_measured": measured_active_W,
            "active_W_assumed":  ACTIVE_W,
            "p_match": P_MATCH,
            "K": 8,
            "user_formula": {
                "accepted_per_cycle": accepted_user,
                "tps":  tps_user,
                "tokw": tokw_user,
                "x_vs_baseline": tokw_user / B1_TOKW,
            },
            "markov_formula": {
                "accepted_per_cycle": accepted_markov,
                "tps":  tps_markov,
                "tokw": tokw_markov,
                "x_vs_baseline": tokw_markov / B1_TOKW,
            },
        }, f, indent=2)
    print(f"[save] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
