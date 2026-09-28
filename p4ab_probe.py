#!/usr/bin/env python3
"""PILLAR 4 STEPS 4A + 4B — Probe per-layer importance and per-token confidence.

4A: For each of 32 layers, compute mean ||layer_output - layer_input|| / ||layer_input||
    (residual-stream perturbation). Layers with low delta are skip candidates.

4B: For 500 decode tokens across multiple prompts, compute the entropy of the
    softmax distribution over the vocab. Histogram. High-confidence (low-H)
    tokens are skip-aggressive candidates.
"""
import os, math, json, gc
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE = 200
PROMPTS = [
    "The future of GPU computing is",
    "Energy efficiency in modern systems means",
    "A neural network learns to predict by",
    "When the temperature drops below freezing,",
    "Quantum mechanics describes nature at the",
    "The most efficient algorithm for sorting is",
    "Climate change affects ecosystems through",
    "In a transformer architecture, attention",
    "Python's standard library includes",
    "The Roman empire fell because",
]
print(f"[p4ab] loading Mistral-7B-v0.1...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")

# 4A: per-layer importance probe
layer_deltas = [[] for _ in range(len(model.model.layers))]
hooks = []
def make_hook(idx):
    def hk(module, inp, out):
        x_in  = inp[0]
        x_out = out[0] if isinstance(out, tuple) else out
        diff = (x_out - x_in).float()
        layer_deltas[idx].append(
            (diff.norm().item(), x_in.float().norm().item()))
    return hk
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

entropies = []
for pi, prompt in enumerate(PROMPTS):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=ids.shape[1]+N_DECODE+4,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
        out = model(input_ids=ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    cur = out.logits[:, -1:].argmax(-1)
    cp = torch.tensor([ids.shape[1]], device="cuda", dtype=torch.long)
    for d in layer_deltas: d.clear()
    for _ in range(N_DECODE):
        with torch.no_grad():
            o = model(input_ids=cur, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
        p = F.softmax(o.logits[0, -1].float(), dim=-1)
        H = -(p * (p.clamp_min(1e-30).log())).sum().item()
        entropies.append(H)
        cur = o.logits.argmax(-1)
        cp += 1
    torch.cuda.synchronize()
    print(f"  prompt {pi+1}/{len(PROMPTS)}: {N_DECODE} decode tokens done  "
          f"H_mean={sum(entropies[-N_DECODE:])/N_DECODE:.3f}", flush=True)

for h in hooks: h.remove()

print(f"\nPer-layer mean residual-stream delta (rel) — averaged over "
      f"{len(PROMPTS)*N_DECODE} decode tokens:")
print(f"{'layer':>5} {'mean_rel':>10} {'min_rel':>9} {'max_rel':>9}  {'verdict'}")
per_layer_summary = []
for i, vals in enumerate(layer_deltas):
    if not vals:
        per_layer_summary.append(dict(layer=i, rel_mean=None)); continue
    rels = [d/(n+1e-9) for (d,n) in vals]
    mr = sum(rels)/len(rels); lo, hi = min(rels), max(rels)
    skip = "SKIP-CAND" if mr < 0.05 else ("INT2-CAND" if mr < 0.20 else "FULL")
    per_layer_summary.append(dict(layer=i, rel_mean=mr, rel_min=lo, rel_max=hi,
                                  verdict=skip))
    print(f"{i:>5} {mr:>10.4f} {lo:>9.4f} {hi:>9.4f}  {skip}")

print(f"\n=== entropy histogram over {len(entropies)} decode tokens ===")
bins = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 100.0]
counts = [0]*(len(bins)-1)
for H in entropies:
    for j in range(len(bins)-1):
        if H >= bins[j] and H < bins[j+1]:
            counts[j] += 1; break
total = len(entropies)
print(f"{'range':>14} {'count':>6} {'frac':>6}")
for j in range(len(bins)-1):
    print(f"  [{bins[j]:>4.1f}, {bins[j+1]:>5.1f}) {counts[j]:>6} {counts[j]/total:>6.1%}")
print(f"  mean H = {sum(entropies)/total:.3f}, median = "
      f"{sorted(entropies)[total//2]:.3f}")

with open(os.path.join("/home/ubuntu/op31-prod-fix",
                       "p4ab_probe.json"), "w") as f:
    json.dump(dict(per_layer=per_layer_summary, entropies=entropies,
                   bins=bins, hist_counts=counts,
                   n_tokens=total), f, indent=2)
print(f"\n[p4ab] wrote p4ab_probe.json")

skip_layers = [r["layer"] for r in per_layer_summary
               if r.get("rel_mean") is not None and r["rel_mean"] < 0.05]
int2_layers = [r["layer"] for r in per_layer_summary
               if r.get("rel_mean") is not None
               and 0.05 <= r["rel_mean"] < 0.20]
print(f"\nSkip-candidate layers (rel<0.05): {skip_layers}  ({len(skip_layers)}/32)")
print(f"INT2-candidate layers (0.05-0.20): {int2_layers}  ({len(int2_layers)}/32)")

low_H_frac = sum(1 for H in entropies if H < 1.0) / total
print(f"Fraction of tokens with H<1.0 (skip-eligible): {low_H_frac:.1%}")
