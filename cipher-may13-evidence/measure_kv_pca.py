"""PCA of Mistral-7B KV-cache contents (per layer, per head, K and V separately).

Hook target: monkey-patch the DynamicCache.update method, which the attention
layer calls with the POST-RoPE keys and POST-projection values right before
they enter the cache. Shapes during decode are [1, num_kv_heads=8, 1, head_dim=128].

Per (layer, head, K|V), run SVD on the 500x128 matrix of captured vectors
and report k needed for 90/95/99% cumulative variance.

Also reports a "joint" PCA on the full 1024-dim concatenated K (or V) per layer
in case heads share structure that per-head PCA would miss.
"""
import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
]
N_DECODE = 100
N = N_DECODE * len(PROMPTS)  # 500


def main():
    print(f"[load] {MODEL} fp16 -> cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)

    # captures[layer][step] -> tensor of shape [num_kv_heads, head_dim] fp32 cpu
    caps_k = [[] for _ in range(N_LAYERS)]
    caps_v = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_patched_update(cache):
        orig = cache.update
        def patched(key_states, value_states, layer_idx, cache_kwargs=None):
            # key_states / value_states: [batch=1, num_kv_heads=8, seq, head_dim=128]
            if capture_enabled[0] and key_states.shape[-2] == 1:  # decode step
                caps_k[layer_idx].append(key_states[0, :, 0, :].detach().to(torch.float32).cpu())
                caps_v[layer_idx].append(value_states[0, :, 0, :].detach().to(torch.float32).cpu())
            return orig(key_states, value_states, layer_idx, cache_kwargs)
        return patched

    # Run 5 prompts x 100 decode tokens, fresh cache per prompt
    print(f"[decode] running {len(PROMPTS)} prompts x {N_DECODE} decode tokens with KV capture")
    t0 = time.time()
    for p_idx, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        cache = DynamicCache()
        cache.update = make_patched_update(cache)
        with torch.no_grad():
            capture_enabled[0] = False
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
            past = out.past_key_values  # may be the same `cache` object
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = True
            for s in range(N_DECODE):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = False
        print(f"[decode]   prompt {p_idx+1}/{len(PROMPTS)} done; layer 0 captures so far: {len(caps_k[0])}")
    print(f"[decode] elapsed {time.time()-t0:.1f}s")

    # Sanity: every layer captured exactly N samples
    counts = [(len(caps_k[i]), len(caps_v[i])) for i in range(N_LAYERS)]
    print(f"[verify] capture counts per layer (K, V): min={min(min(c) for c in counts)}, max={max(max(c) for c in counts)}, expected={N}")
    assert all(ck == N and cv == N for ck, cv in counts), f"capture mismatch: {counts}"

    # Free model
    del model
    torch.cuda.empty_cache()

    # ---- PCA per (layer, head, K|V) ----
    K_GRID = [4, 8, 16, 32, 48, 64, 96]
    THRESHOLDS = [0.90, 0.95, 0.99]

    print(f"\n[svd] running per-head PCA: 32 layers x 8 heads x 2 (K, V) = 512 SVDs on 500x128 matrices")
    per_head_results = {"K": {}, "V": {}}  # [type][layer][head] -> {k_for_thresh, cum_at_k}
    per_layer_joint = {"K": {}, "V": {}}   # [type][layer] -> joint 1024-dim PCA

    for kind, caps in [("K", caps_k), ("V", caps_v)]:
        for layer in range(N_LAYERS):
            stack = torch.stack(caps[layer], dim=0)  # [N, num_kv_heads, head_dim]
            # Per-head PCA
            for head in range(NUM_KV_HEADS):
                X = stack[:, head, :]  # [N, head_dim=128]
                X = X - X.mean(dim=0, keepdim=True)
                Xg = X.to("cuda")
                U, S, Vh = torch.linalg.svd(Xg, full_matrices=False)
                S = S.to("cpu")
                var = S ** 2
                cum = torch.cumsum(var, dim=0) / var.sum()
                rec = {"cum_at_k": {}, "k_for_thresh": {}, "top10_singular": S[:10].tolist(), "max_rank": len(S)}
                for k in K_GRID:
                    rec["cum_at_k"][k] = cum[k-1].item() if k <= len(cum) else 1.0
                for thr in THRESHOLDS:
                    ge = (cum >= thr).nonzero(as_tuple=True)[0]
                    rec["k_for_thresh"][thr] = (int(ge[0].item()) + 1) if len(ge) > 0 else None
                per_head_results[kind].setdefault(layer, {})[head] = rec

            # Joint PCA (all 8 heads concatenated, 1024-dim)
            X = stack.reshape(N, NUM_KV_HEADS * HEAD_DIM)  # [N, 1024]
            X = X - X.mean(dim=0, keepdim=True)
            Xg = X.to("cuda")
            U, S, Vh = torch.linalg.svd(Xg, full_matrices=False)
            S = S.to("cpu")
            var = S ** 2
            cum = torch.cumsum(var, dim=0) / var.sum()
            rec = {"cum_at_k": {}, "k_for_thresh": {}, "top20_singular": S[:20].tolist(), "max_rank": len(S)}
            joint_K_GRID = [8, 16, 32, 64, 128, 256, 384, 499]
            for k in joint_K_GRID:
                rec["cum_at_k"][k] = cum[k-1].item() if k <= len(cum) else 1.0
            for thr in THRESHOLDS:
                ge = (cum >= thr).nonzero(as_tuple=True)[0]
                rec["k_for_thresh"][thr] = (int(ge[0].item()) + 1) if len(ge) > 0 else None
            per_layer_joint[kind][layer] = rec

    # ---- TABLE 1: per-head k for 95% (averaged over 8 heads, per layer) ----
    print()
    print("=" * 88)
    print("TABLE 1: Per-head k needed for 95% / 99% variance (averaged over 8 heads, per layer)")
    print("(head_dim = 128, so k_max per head = 128)")
    print("=" * 88)
    header = f"{'layer':>5}  {'K-95':>6} {'K-99':>6}    {'V-95':>6} {'V-99':>6}    {'K-95-min':>8} {'K-95-max':>8}  {'V-95-min':>8} {'V-95-max':>8}"
    print(header)
    print("-" * len(header))
    for layer in range(N_LAYERS):
        k95s = [per_head_results["K"][layer][h]["k_for_thresh"][0.95] for h in range(NUM_KV_HEADS)]
        k99s = [per_head_results["K"][layer][h]["k_for_thresh"][0.99] for h in range(NUM_KV_HEADS)]
        v95s = [per_head_results["V"][layer][h]["k_for_thresh"][0.95] for h in range(NUM_KV_HEADS)]
        v99s = [per_head_results["V"][layer][h]["k_for_thresh"][0.99] for h in range(NUM_KV_HEADS)]
        k95s = [x if x is not None else 128 for x in k95s]
        k99s = [x if x is not None else 128 for x in k99s]
        v95s = [x if x is not None else 128 for x in v95s]
        v99s = [x if x is not None else 128 for x in v99s]
        print(f"{layer:>5}  {sum(k95s)/8:>6.1f} {sum(k99s)/8:>6.1f}    "
              f"{sum(v95s)/8:>6.1f} {sum(v99s)/8:>6.1f}    "
              f"{min(k95s):>8} {max(k95s):>8}  {min(v95s):>8} {max(v95s):>8}")

    # ---- TABLE 2: per-head cum variance at fixed k (averaged over 8 heads) ----
    print()
    print("=" * 88)
    print("TABLE 2: Per-head cumulative variance at fixed k (averaged over 8 heads, per layer)")
    print("=" * 88)
    K_FIXED = [8, 16, 32, 48, 64]
    header = f"{'layer':>5}  " + "  ".join(f"K@k={k:>3}" for k in K_FIXED) + "    " + "  ".join(f"V@k={k:>3}" for k in K_FIXED)
    print(header)
    print("-" * len(header))
    for layer in range(N_LAYERS):
        kparts, vparts = [], []
        for k in K_FIXED:
            kvals = [per_head_results["K"][layer][h]["cum_at_k"][k] for h in range(NUM_KV_HEADS)]
            vvals = [per_head_results["V"][layer][h]["cum_at_k"][k] for h in range(NUM_KV_HEADS)]
            kparts.append(f"{sum(kvals)/8:>7.4f}")
            vparts.append(f"{sum(vvals)/8:>7.4f}")
        print(f"{layer:>5}  " + "  ".join(kparts) + "    " + "  ".join(vparts))

    # ---- TABLE 3: joint (1024-dim) PCA per layer ----
    print()
    print("=" * 88)
    print("TABLE 3: Joint K/V PCA per layer (heads concatenated, 1024-dim, max rank=499)")
    print("=" * 88)
    header = f"{'layer':>5}  {'K-95':>6} {'K-99':>6}  {'V-95':>6} {'V-99':>6}    {'K@k=64':>8} {'K@k=128':>8} {'V@k=64':>8} {'V@k=128':>8}"
    print(header)
    print("-" * len(header))
    for layer in range(N_LAYERS):
        kr = per_layer_joint["K"][layer]
        vr = per_layer_joint["V"][layer]
        def fmt_k(rec, thr):
            return rec["k_for_thresh"][thr] if rec["k_for_thresh"][thr] is not None else f">{rec['max_rank']}"
        print(f"{layer:>5}  {str(fmt_k(kr,0.95)):>6} {str(fmt_k(kr,0.99)):>6}  "
              f"{str(fmt_k(vr,0.95)):>6} {str(fmt_k(vr,0.99)):>6}    "
              f"{kr['cum_at_k'][64]:>8.4f} {kr['cum_at_k'][128]:>8.4f} "
              f"{vr['cum_at_k'][64]:>8.4f} {vr['cum_at_k'][128]:>8.4f}")

    # ---- Summary ----
    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)

    all_K_95 = []
    all_K_99 = []
    all_V_95 = []
    all_V_99 = []
    for layer in range(N_LAYERS):
        for head in range(NUM_KV_HEADS):
            k95 = per_head_results["K"][layer][head]["k_for_thresh"][0.95]
            k99 = per_head_results["K"][layer][head]["k_for_thresh"][0.99]
            v95 = per_head_results["V"][layer][head]["k_for_thresh"][0.95]
            v99 = per_head_results["V"][layer][head]["k_for_thresh"][0.99]
            if k95 is not None: all_K_95.append(k95)
            if k99 is not None: all_K_99.append(k99)
            if v95 is not None: all_V_95.append(v95)
            if v99 is not None: all_V_99.append(v99)

    def stats(vals, name):
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2]
        p90 = vals[int(0.9 * n)]
        p99 = vals[min(n - 1, int(0.99 * n))]
        print(f"  {name}:  n={n}  min={min(vals)}  median={median}  mean={mean:.1f}  p90={p90}  p99={p99}  max={max(vals)}")

    print("Per-head k for 95% variance:")
    stats(all_K_95, "  K-95")
    stats(all_V_95, "  V-95")
    print("Per-head k for 99% variance:")
    stats(all_K_99, "  K-99")
    stats(all_V_99, "  V-99")

    # CIPHER feasibility headline
    avg_k_95 = sum(all_K_95) / len(all_K_95)
    avg_v_95 = sum(all_V_95) / len(all_V_95)
    compression_95 = HEAD_DIM / max(avg_k_95, avg_v_95)  # bytes saved
    print()
    print(f"Avg per-head k for 95% var:  K={avg_k_95:.1f}  V={avg_v_95:.1f}  (out of head_dim={HEAD_DIM})")
    print(f"Implied per-head bandwidth reduction at 95% var: {compression_95:.2f}x")
    print(f"  -> KV cache traffic compressed from {HEAD_DIM} to ~{max(avg_k_95, avg_v_95):.0f} dims per token per head")
    if compression_95 >= 4:
        print(f"  --> Strong KV-projection win: store {max(avg_k_95, avg_v_95):.0f}/{HEAD_DIM} = {max(avg_k_95, avg_v_95)/HEAD_DIM:.0%} of bytes, reconstruct at attention time.")
    elif compression_95 >= 2:
        print(f"  --> Modest KV-projection win.")
    else:
        print(f"  --> KV per-head intrinsic dim too high for projection to help.")

    # Save raw
    out_path = "/home/ubuntu/op31-prod-fix/kv_pca_results.json"
    serial = {"per_head": {}, "per_layer_joint": {}}
    for kind in ("K", "V"):
        serial["per_head"][kind] = {}
        for layer in range(N_LAYERS):
            serial["per_head"][kind][str(layer)] = {}
            for head in range(NUM_KV_HEADS):
                rec = per_head_results[kind][layer][head]
                serial["per_head"][kind][str(layer)][str(head)] = {
                    "cum_at_k": {str(k): v for k, v in rec["cum_at_k"].items()},
                    "k_for_thresh": {str(t): v for t, v in rec["k_for_thresh"].items()},
                    "top10_singular": rec["top10_singular"],
                }
        serial["per_layer_joint"][kind] = {}
        for layer in range(N_LAYERS):
            rec = per_layer_joint[kind][layer]
            serial["per_layer_joint"][kind][str(layer)] = {
                "cum_at_k": {str(k): v for k, v in rec["cum_at_k"].items()},
                "k_for_thresh": {str(t): v for t, v in rec["k_for_thresh"].items()},
                "top20_singular": rec["top20_singular"],
            }
    with open(out_path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"\n[save] raw -> {out_path}")


if __name__ == "__main__":
    main()
