"""Measure layer-to-layer residual stream similarity in Mistral-7B during decode.

Tests whether middle layers iterate toward a fixed point — a precondition for
Anderson-acceleration-style collapse of multiple layers into a few iterations.

Captures the input residual stream at each of the 32 decoder layers plus the
input of the final RMSNorm (= output of layer 31), giving 33 hidden states per
token and 32 adjacent (L, L+1) pairs.
"""
import time
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
PROMPT = "Explain how a CPU executes instructions step by step"
N_DECODE = 200
N_LAYERS = 32  # captures layer inputs 0..31 plus final-norm input -> 33 capture slots


def main():
    print(f"[load] {MODEL} fp16 -> cuda")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    print(f"[load] done in {time.time()-t0:.1f}s")

    # 33 capture slots: idx 0..31 = input of layer i; idx 32 = input of final norm = output of layer 31
    captures = [[] for _ in range(N_LAYERS + 1)]
    capture_enabled = [False]

    def make_pre(idx):
        def hook(module, args, kwargs):
            if not capture_enabled[0]:
                return None
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hs is None and "input" in kwargs:
                hs = kwargs["input"]
            if not isinstance(hs, torch.Tensor) or hs.dim() != 3 or hs.shape[1] != 1:
                return None
            captures[idx].append(hs[0, 0, :].detach().to(torch.float32).cpu())
            return None
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_pre(i), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(make_pre(N_LAYERS), with_kwargs=True))
    print(f"[hook] registered {len(handles)} pre-hooks (32 layer inputs + 1 final norm input)")

    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        capture_enabled[0] = False
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = True
        print(f"[decode] running {N_DECODE} decode tokens with capture")
        for s in range(N_DECODE):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = False
    for h in handles:
        h.remove()

    counts = [len(c) for c in captures]
    print(f"[verify] capture counts: min={min(counts)}, max={max(counts)}, expected={N_DECODE}")
    assert all(c == N_DECODE for c in counts), f"capture mismatch {counts}"

    del model
    torch.cuda.empty_cache()

    # Stack into [33, N_DECODE, 4096]
    H = torch.stack([torch.stack(c, dim=0) for c in captures], dim=0)  # [33, 200, 4096]
    print(f"[shape] hidden states tensor: {tuple(H.shape)}")

    # Compute per-token cos_sim(L, L+1) for all 32 pairs, then average over tokens
    pairs = []
    for L in range(N_LAYERS):  # L = 0..31, pairs are (L, L+1) including (31, 32 = final norm input)
        a = H[L]      # [N_DECODE, 4096]
        b = H[L + 1]
        cos = F.cosine_similarity(a, b, dim=-1)            # [N_DECODE]
        delta_norm = (b - a).norm(dim=-1) / a.norm(dim=-1)  # [N_DECODE], relative change
        pairs.append({
            "L": L,
            "L_next": L + 1,
            "cos_mean": cos.mean().item(),
            "cos_std": cos.std().item(),
            "cos_min": cos.min().item(),
            "cos_max": cos.max().item(),
            "delta_norm_mean": delta_norm.mean().item(),
            "delta_norm_std": delta_norm.std().item(),
            "norm_a_mean": a.norm(dim=-1).mean().item(),
            "norm_b_mean": b.norm(dim=-1).mean().item(),
        })

    # ---- Report ----
    print()
    print("=" * 100)
    print("Adjacent-layer residual cosine similarity and delta norm (averaged over 200 decode tokens)")
    print("Pair (L, L+1) means: input of layer L vs input of layer L+1")
    print("Pair (31, 32) means: input of layer 31 vs input of final RMSNorm (= output of layer 31)")
    print("=" * 100)
    print(f"{'L':>3}{'L+1':>4}   {'cos_mean':>10} {'cos_std':>9} {'cos_min':>8} {'cos_max':>8}   "
          f"{'||x_a||':>10} {'||x_b||':>10}   {'delta/||x||':>11}")
    print("-" * 100)
    for p in pairs:
        print(f"{p['L']:>3}{p['L_next']:>4}   "
              f"{p['cos_mean']:>10.6f} {p['cos_std']:>9.6f} {p['cos_min']:>8.4f} {p['cos_max']:>8.4f}   "
              f"{p['norm_a_mean']:>10.2f} {p['norm_b_mean']:>10.2f}   "
              f"{p['delta_norm_mean']:>11.4f}")

    # Convergence curve as ASCII
    print()
    print("=" * 100)
    print("Convergence curve (avg cos_sim per layer pair)")
    print("=" * 100)
    cmin = min(p["cos_mean"] for p in pairs)
    cmax = max(p["cos_mean"] for p in pairs)
    width = 60
    for p in pairs:
        bar_len = int((p["cos_mean"] - cmin) / max(1e-9, cmax - cmin) * width)
        print(f"  L={p['L']:>2}->{p['L_next']:>2}  cos={p['cos_mean']:.4f}  |{'#' * bar_len}{' ' * (width - bar_len)}|")

    # Bands
    very_high = [p for p in pairs if p["cos_mean"] > 0.99]
    high = [p for p in pairs if p["cos_mean"] > 0.95]
    print()
    print("=" * 100)
    print(f"Pairs with cos_sim > 0.99 (near-fixed-point):  {len(very_high)}")
    if very_high:
        spans = " ".join(f"({p['L']}->{p['L_next']}: {p['cos_mean']:.4f})" for p in very_high)
        print(f"  pairs: {spans}")
    print(f"Pairs with cos_sim > 0.95:  {len(high)}")
    if high:
        spans = " ".join(f"({p['L']}->{p['L_next']}: {p['cos_mean']:.4f})" for p in high)
        print(f"  pairs: {spans}")

    # Delta-norm analysis
    print()
    print("=" * 100)
    print("Delta norm summary  ||x_{L+1} - x_L|| / ||x_L||  (smaller = layer barely changes residual)")
    print("=" * 100)
    sorted_by_delta = sorted(pairs, key=lambda p: p["delta_norm_mean"])
    print("5 layers with SMALLEST relative change (most fixed-point-like):")
    for p in sorted_by_delta[:5]:
        print(f"  L={p['L']:>2}->{p['L_next']:>2}  cos={p['cos_mean']:.4f}  delta/||x||={p['delta_norm_mean']:.4f}")
    print("5 layers with LARGEST relative change:")
    for p in sorted_by_delta[-5:]:
        print(f"  L={p['L']:>2}->{p['L_next']:>2}  cos={p['cos_mean']:.4f}  delta/||x||={p['delta_norm_mean']:.4f}")

    # CIPHER feasibility verdict
    print()
    print("=" * 100)
    print("ANDERSON-ACCELERATION FEASIBILITY")
    print("=" * 100)

    # Find longest contiguous run of layers with cos > 0.95
    best_start = best_end = -1
    cur_start = -1
    for i, p in enumerate(pairs):
        if p["cos_mean"] > 0.95:
            if cur_start < 0:
                cur_start = p["L"]
            cur_end = p["L_next"]
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
        else:
            cur_start = -1
    if best_start >= 0:
        n_collapsible = best_end - best_start
        if n_collapsible > 0 and best_end > best_start:
            speedup = n_collapsible / 3  # Anderson with ~3 iterations
            print(f"Longest contiguous span with cos > 0.95: layers {best_start}..{best_end} "
                  f"({n_collapsible + 1} layers, {n_collapsible} pairs)")
            print(f"If replaced with 3-step Anderson acceleration: {speedup:.1f}x compute reduction on this span")
            frac = (n_collapsible + 1) / N_LAYERS
            print(f"Span covers {frac:.1%} of the model's depth.")
        else:
            print(f"No contiguous near-fixed-point span found.")
    else:
        print("NO layer pair has cos_sim > 0.95. Layers are NOT iterating toward a fixed point.")
        print("Anderson acceleration would NOT yield speedup on this model.")

    # Save raw
    out_path = "/home/ubuntu/op31-prod-fix/layer_convergence_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "prompt": PROMPT,
            "n_decode": N_DECODE,
            "pairs": pairs,
        }, f, indent=2)
    print(f"\n[save] raw -> {out_path}")


if __name__ == "__main__":
    main()
