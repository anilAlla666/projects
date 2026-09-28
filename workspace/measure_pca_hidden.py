"""PCA measurement of Mistral-7B hidden states during decode.

Captures the residual-stream input to each of the 32 decoder layers across 500
decode tokens (5 prompts x 100 tokens), then per-layer SVD to estimate the
intrinsic dimension k needed for various variance thresholds.
"""
import os
import sys
import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE = 100
N_LAYERS = 32
HIDDEN = 4096

PROMPTS = [
    "Explain how a CPU executes instructions step by step",
    "Write a story about a detective solving a mystery in Tokyo",
    "What are the economic implications of rising interest rates",
    "Describe the process of photosynthesis in detail",
    "Compare and contrast democracy and authoritarianism",
]

K_GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
K_FIXED = [16, 32, 64, 128, 256]
THRESHOLDS = [0.90, 0.95, 0.99, 0.995]


def main():
    print(f"[load] {MODEL} fp16 -> cuda")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda"
    )
    model.train(False)  # inference mode (no dropout / running-stat updates)
    print(f"[load] done in {time.time()-t0:.1f}s")

    # Per-layer capture buffers (will store [4096] cpu fp32 tensors)
    captures = [[] for _ in range(N_LAYERS)]
    capture_enabled = [False]

    def make_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            if not capture_enabled[0]:
                return None
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                hs = args[0]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                hs = kwargs["hidden_states"]
            else:
                return None
            # During decode, hs has shape [batch=1, seq=1, hidden]
            if hs.dim() != 3 or hs.shape[1] != 1:
                return None
            captures[layer_idx].append(hs[0, 0, :].detach().to(torch.float32).cpu())
            return None
        return pre_hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        handles.append(h)
    print(f"[hook] registered {len(handles)} pre-hooks")

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"[decode] prompt {p_idx+1}/{len(PROMPTS)}: {prompt[:50]!r}")
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            # Prefill (capture OFF — single forward over the prompt)
            capture_enabled[0] = False
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Decode (capture ON)
            capture_enabled[0] = True
            for step in range(N_DECODE):
                out = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            capture_enabled[0] = False

        per_prompt = len(captures[0]) - p_idx * N_DECODE
        print(f"[decode]   captured {per_prompt}/layer this prompt; total per layer = {len(captures[0])}")

    for h in handles:
        h.remove()

    counts = [len(c) for c in captures]
    print(f"[verify] capture counts per layer: min={min(counts)} max={max(counts)} expected={N_DECODE*len(PROMPTS)}")
    assert all(c == N_DECODE * len(PROMPTS) for c in counts), "capture count mismatch"

    # Free model memory before SVD work
    del model
    torch.cuda.empty_cache()

    N = N_DECODE * len(PROMPTS)
    print(f"[svd] running per-layer SVD on {N}x{HIDDEN} matrices (32 layers)")
    layer_results = []
    for layer_idx in range(N_LAYERS):
        X = torch.stack(captures[layer_idx], dim=0)  # [N, HIDDEN] float32 cpu
        X = X - X.mean(dim=0, keepdim=True)
        Xg = X.to("cuda")
        U, S, Vh = torch.linalg.svd(Xg, full_matrices=False)
        S = S.to("cpu")
        var = (S ** 2)
        total_var = var.sum().item()
        cum = torch.cumsum(var, dim=0) / total_var

        max_k = len(S)
        cum_at_k = {}
        for k in K_GRID:
            if k <= max_k:
                cum_at_k[k] = cum[k - 1].item()
            else:
                cum_at_k[k] = 1.0  # rank-deficient ceiling

        k_for_thresh = {}
        for thr in THRESHOLDS:
            ge = (cum >= thr).nonzero(as_tuple=True)[0]
            if len(ge) == 0:
                k_for_thresh[thr] = None
            else:
                k_for_thresh[thr] = int(ge[0].item()) + 1

        top20 = S[:20].tolist()
        layer_results.append({
            "layer": layer_idx,
            "cum_at_k": cum_at_k,
            "k_for_thresh": k_for_thresh,
            "top20_singular": top20,
            "total_variance": total_var,
            "max_rank": max_k,
        })
        if layer_idx % 4 == 0:
            print(f"[svd]   layer {layer_idx}: k95={k_for_thresh[0.95]}, k99={k_for_thresh[0.99]}, "
                  f"cum@k64={cum_at_k[64]:.4f}, top1_sigma={top20[0]:.2f}")

    print()
    print("=" * 78)
    print("TABLE 1: k needed for X% variance, per layer")
    print("=" * 78)
    header = f"{'layer':>5}  {'90%':>6}  {'95%':>6}  {'99%':>6}  {'99.5%':>6}"
    print(header)
    print("-" * len(header))
    for r in layer_results:
        def fmt(thr):
            v = r["k_for_thresh"][thr]
            return f">{r['max_rank']}" if v is None else str(v)
        print(f"{r['layer']:>5}  {fmt(0.90):>6}  {fmt(0.95):>6}  {fmt(0.99):>6}  {fmt(0.995):>6}")

    print()
    print("=" * 78)
    print("TABLE 2: Cumulative variance at fixed k, per layer")
    print("=" * 78)
    header2 = f"{'layer':>5}  " + "  ".join(f"{'k='+str(k):>9}" for k in K_FIXED)
    print(header2)
    print("-" * len(header2))
    for r in layer_results:
        row = f"{r['layer']:>5}  " + "  ".join(f"{r['cum_at_k'][k]:>9.4f}" for k in K_FIXED)
        print(row)

    print()
    print("=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)
    k95 = [r["k_for_thresh"][0.95] for r in layer_results if r["k_for_thresh"][0.95] is not None]
    k99 = [r["k_for_thresh"][0.99] for r in layer_results if r["k_for_thresh"][0.99] is not None]
    k95_full = [(r["layer"], r["k_for_thresh"][0.95]) for r in layer_results]
    if k95:
        print(f"Average k for 95% variance: {sum(k95)/len(k95):.1f}  (min={min(k95)}, max={max(k95)})")
    if k99:
        print(f"Average k for 99% variance: {sum(k99)/len(k99):.1f}  (min={min(k99)}, max={max(k99)})")

    sorted_hard = sorted(k95_full, key=lambda x: -(x[1] if x[1] is not None else 9999))
    print("\nHardest layers (highest k for 95% var):")
    for layer, k in sorted_hard[:5]:
        print(f"  layer {layer}: k={k}")

    sorted_easy = sorted(k95_full, key=lambda x: (x[1] if x[1] is not None else 9999))
    print("\nEasiest layers (lowest k for 95% var):")
    for layer, k in sorted_easy[:5]:
        print(f"  layer {layer}: k={k}")

    cum64 = [r["cum_at_k"][64] for r in layer_results]
    avg64 = sum(cum64) / len(cum64)
    min64 = min(cum64)
    max64 = max(cum64)
    print(f"\nAt k=64: avg cumulative variance = {avg64:.4f} (min {min64:.4f}, max {max64:.4f})")
    if avg64 >= 0.95:
        print("  --> CIPHER subspace projection at k=64 captures >=95% on average. 32x weight traffic reduction is supported by the data.")
    elif avg64 >= 0.90:
        print("  --> CIPHER k=64 captures >=90% but <95%. 32x reduction is feasible with some accuracy loss; consider k=128.")
    else:
        print("  --> CIPHER k=64 below 90% on average. Need larger k to hit 95% target.")

    out_path = "/home/ubuntu/op31-prod-fix/pca_hidden_results.json"
    serial = []
    for r in layer_results:
        serial.append({
            "layer": r["layer"],
            "cum_at_k": {str(k): v for k, v in r["cum_at_k"].items()},
            "k_for_thresh": {str(t): v for t, v in r["k_for_thresh"].items()},
            "top20_singular": r["top20_singular"],
            "total_variance": r["total_variance"],
            "max_rank": r["max_rank"],
        })
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "n_decode_per_prompt": N_DECODE,
            "n_prompts": len(PROMPTS),
            "n_total_samples": N_DECODE * len(PROMPTS),
            "hidden_dim": HIDDEN,
            "k_grid": K_GRID,
            "thresholds": THRESHOLDS,
            "layers": serial,
        }, f, indent=2)
    print(f"\n[save] raw per-layer results -> {out_path}")


if __name__ == "__main__":
    main()
