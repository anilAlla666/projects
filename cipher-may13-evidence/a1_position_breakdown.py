"""A1 follow-up: where in the 200-decode sequence does top1 fail under k=499 decode-only?

If failures are concentrated at positions 100+, the 22.5% gap is "off-manifold drift"
once eval goes beyond what calibration covered (calibration = 100 decode tokens / prompt).

If failures are spread across all 200 positions, it's per-linear chain error.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import projection_quality as pq

def main():
    model, tok = pq.load_model()
    captures = pq.calibrate(model, tok)
    qs = pq.compute_qs(captures, k_max=pq.K_MAX)
    del captures

    eval_ids = tok(pq.EVAL_PROMPT, return_tensors="pt").input_ids.to("cuda")
    print(f"\n[baseline] free run 200 tokens", flush=True)
    baseline = pq.free_run_baseline(model, eval_ids, pq.N_EVAL_DECODE)
    base_text = tok.decode(baseline["decoded_tokens"])
    print(f"[baseline] {base_text[:120]!r}", flush=True)

    # k=499 DECODE-ONLY patching (prefill UNPATCHED, decode PATCHED)
    print(f"\n[projected] k=499 decode-only patching (prefill unpatched, decode patched)", flush=True)
    # Prefill FIRST with full W
    with torch.no_grad():
        out = model(input_ids=eval_ids, use_cache=True)
        prefill_past = out.past_key_values
    # Now patch
    store = {}
    cum_vars = pq.patch_model(model, qs, pq.K_PROFILES["v4_uniform_512"], store)
    try:
        proj = pq.teacher_forced_run(
            model, eval_ids,
            baseline["decoded_tokens"][:pq.N_EVAL_DECODE],
            prefill_past=prefill_past,
        )
    finally:
        pq.unpatch_model(model, store)
        torch.cuda.empty_cache()

    n = min(len(baseline["decode_logits"]), len(proj["logits"]))
    base_argmax = [baseline["decode_logits"][i].argmax().item() for i in range(n)]
    proj_argmax = [proj["logits"][i].argmax().item() for i in range(n)]
    correct = [int(b == p) for b, p in zip(base_argmax, proj_argmax)]

    # Bucket by position
    buckets = [(0, 50), (50, 100), (100, 150), (150, 200)]
    print(f"\n  {'position range':<18} {'top1':>8}", flush=True)
    for lo, hi in buckets:
        sub = correct[lo:hi]
        rate = sum(sub) / len(sub) if sub else 0
        print(f"  {f'[{lo:>3} .. {hi:>3})':<18} {rate:>7.4f}  ({sum(sub)}/{len(sub)})", flush=True)
    overall = sum(correct) / len(correct)
    print(f"  {'overall (0..200)':<18} {overall:>7.4f}  ({sum(correct)}/{len(correct)})", flush=True)

    # Per-position cumulative top1
    print(f"\n  position-by-position top1 (every 10):", flush=True)
    for i in range(0, n, 10):
        cum = sum(correct[:i+1]) / (i+1) if i > 0 else correct[0]
        # Show match status of last few positions in this stretch
        marks = ''.join('1' if c else '0' for c in correct[max(0,i-9):i+1])
        print(f"    pos {i:>3}: cum_top1={cum:.4f}   recent={marks}", flush=True)


if __name__ == "__main__":
    main()
