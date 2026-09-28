"""B2: dynamic sparsity exploitation analysis.

Capture MLP intermediate (silu(gate) * up) at each decode step for 100 tokens.
For each layer:
  - Sparsity per token (using threshold |x| > 0.01 * max(|x|))
  - Indices of nonzero entries
  - Jaccard similarity of nonzero positions vs PREVIOUS token

If sparsity > 90% AND Jaccard > 50% → CIPHER could precompute a sparse down_proj
(only read the rows corresponding to non-zero columns).
"""
import os, sys, time, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE = 100
PROMPT = "Explain how a CPU executes instructions step by step"
RELATIVE_THRESHOLD = 0.01  # |x| > 0.01 * max(|x|) defines "nonzero"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.train(False)
    n_layers = len(model.model.layers)

    # Capture down_proj input = silu(gate) * up at each layer per decode step
    # Use forward_pre_hook on down_proj — args[0] is the activation entering down_proj
    captures = [[] for _ in range(n_layers)]
    capture_enabled = [False]

    def make_hook(li):
        def pre(module, args):
            if not capture_enabled[0]:
                return None
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            if x.dim() == 3 and x.shape[1] == 1:
                # detection: shape [1, 1, 14336] for Mistral down_proj input
                captures[li].append(x[0, 0, :].detach().to(torch.float32).cpu())
        return pre

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(i)))

    # Run decode
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = True
        for s in range(N_DECODE):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        capture_enabled[0] = False
    for h in handles: h.remove()

    # Analysis: per layer, per token: sparsity + Jaccard with previous
    print(f"=== B2: MLP intermediate sparsity + Jaccard analysis ===\n", flush=True)
    print(f"Threshold: |x| > {RELATIVE_THRESHOLD} * max(|x|) per-token\n", flush=True)
    print(f"  {'L':>3} {'sparsity_avg':>13} {'sparsity_min':>13} {'sparsity_max':>13} {'jaccard_avg':>12} {'jaccard_min':>12}", flush=True)
    print("  " + "-" * 80, flush=True)

    layer_stats = []
    for li in range(n_layers):
        toks = captures[li]
        prev_idx = None
        sparsities = []
        jaccards = []
        for t, x in enumerate(toks):
            xa = x.abs()
            thresh = RELATIVE_THRESHOLD * xa.max().item()
            mask = (xa > thresh)
            nonzero_idx = mask.nonzero(as_tuple=True)[0]
            n_nonzero = mask.sum().item()
            sparsity = 1.0 - n_nonzero / x.numel()
            sparsities.append(sparsity)
            if prev_idx is not None:
                cur_set = set(nonzero_idx.tolist())
                prev_set = set(prev_idx.tolist())
                inter = len(cur_set & prev_set)
                union = len(cur_set | prev_set)
                jaccard = inter / union if union > 0 else 0.0
                jaccards.append(jaccard)
            prev_idx = nonzero_idx
        s_avg = sum(sparsities)/len(sparsities) if sparsities else 0
        s_min = min(sparsities) if sparsities else 0
        s_max = max(sparsities) if sparsities else 0
        j_avg = sum(jaccards)/len(jaccards) if jaccards else 0
        j_min = min(jaccards) if jaccards else 0
        layer_stats.append({
            "layer": li,
            "sparsity_avg": s_avg,
            "sparsity_min": s_min,
            "sparsity_max": s_max,
            "jaccard_avg": j_avg,
            "jaccard_min": j_min,
        })
        print(f"  {li:>3} {s_avg:>12.4f} {s_min:>13.4f} {s_max:>13.4f} {j_avg:>12.4f} {j_min:>12.4f}", flush=True)

    # Verdict per layer
    print(f"\n  Layers with avg_sparsity > 0.90 AND avg_jaccard > 0.50:", flush=True)
    qual = [s for s in layer_stats if s["sparsity_avg"] > 0.90 and s["jaccard_avg"] > 0.50]
    if qual:
        for s in qual:
            print(f"    layer {s['layer']:>2}: sparsity={s['sparsity_avg']:.4f}  jaccard={s['jaccard_avg']:.4f}", flush=True)
        # Estimated bandwidth savings if we exploit
        # down_proj reads (1 - sparsity) * 14336 rows per token instead of 14336
        # Per-layer down_proj weight: 14336 * 4096 * 2 bytes = 117 MB
        # For qualifying layers, save sparsity * 117 MB per token
        savings_mb_per_token = sum(s["sparsity_avg"] * 117 for s in qual)
        print(f"  Estimated HBM savings on down_proj weight reads: {savings_mb_per_token:.0f} MB/token  ({len(qual)}/32 layers qualify)", flush=True)
    else:
        print(f"    NONE — sparse-down_proj exploitation does not pass the spec criterion", flush=True)

    # Summary
    print(f"\n  Avg across all 32 layers: sparsity={sum(s['sparsity_avg'] for s in layer_stats)/32:.4f}  jaccard={sum(s['jaccard_avg'] for s in layer_stats)/32:.4f}", flush=True)

    with open("/home/ubuntu/op31-prod-fix/b2_sparsity_results.json", "w") as f:
        json.dump(layer_stats, f, indent=2)
    print(f"\n[save] -> b2_sparsity_results.json", flush=True)


if __name__ == "__main__":
    main()
