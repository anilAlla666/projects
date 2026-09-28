#!/usr/bin/env python3
"""PILLAR 4 STEPS 4C + 4D — Adaptive compute depth (the 10× lever).

Mechanism:
  - Monkey-patch each MistralDecoderLayer.forward.
  - Global state: skip_mask[32] (bool) + current entropy (set after each step).
  - When a layer's skip bit is true, the layer's forward returns its INPUT
    unchanged → residual flows through, no compute on that layer.
  - After each token, compute entropy of next-token logits. Use THIS entropy
    to set the skip mask for the NEXT token (lag-1 predictor).

Skip policy (heuristic, calibrated to Step 4A's per-layer min_rel ranking):
  - layers with the lowest mean residual-delta in 4A get skipped first
  - aggressiveness scales with entropy:
      H < 0.1   → skip up to 16 layers (very confident)
      H < 0.5   → skip up to 12
      H < 1.0   → skip up to 8
      H < 2.0   → skip up to 4
      else      → skip 0

Quality gate (4D):
  - Run 200 tokens with adaptive skip vs no-skip
  - Compare generated text
  - Compute mean KL(no_skip ‖ skip) on next-token distributions
  - Quality target per spec: <5% degradation (KL < ~0.1)

Then 4E: measure tok/s, watts, tok/W at B=1, 8, 32, 64 in eager mode.
"""
import os, sys, json, time, threading, subprocess, gc
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "mistralai/Mistral-7B-v0.1"
N_DECODE_QUAL = 200
N_PROMPT_QUAL = 5
PREFILL_LEN = 1024
WARMUP_STEPS = 10
MEASURE_SECONDS = 10.0

# Load Step 4A's per-layer min_rel — pick layers with lowest min_rel as
# skip candidates (these layers are sometimes near-identity)
ROOT = "/home/ubuntu/op31-prod-fix"
with open(os.path.join(ROOT, "p4ab_probe.json")) as f:
    probe = json.load(f)
layer_rels = sorted(
    [(r["layer"], r["rel_min"]) for r in probe["per_layer"]
     if r.get("rel_min") is not None],
    key=lambda x: x[1])
SKIP_PRIORITY = [l for (l, _) in layer_rels]   # skip-first ordering by min_rel
# Don't skip layer 0 (input embedding stretching) or layer 31 (logit head feeder)
SKIP_PRIORITY = [l for l in SKIP_PRIORITY if l not in (0, 31)]
print(f"[p4cd] skip priority by min_rel ascending: {SKIP_PRIORITY[:20]}...")

print("[p4cd] loading Mistral-7B...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")

# Global state
class State:
    skip_mask = [False] * len(model.model.layers)
    n_skipped = 0
    n_run = 0


# Monkey-patch DecoderLayer.forward per index
orig_layer_forward = type(model.model.layers[0]).forward
def make_skip_forward(idx, orig):
    def forward(self, hidden_states, *args, **kwargs):
        if State.skip_mask[idx]:
            State.n_skipped += 1
            # Modern transformers returns the tensor directly, not a tuple.
            return hidden_states
        State.n_run += 1
        return orig(self, hidden_states, *args, **kwargs)
    return forward

# Apply per-instance bound forward
for i, layer in enumerate(model.model.layers):
    layer.forward = make_skip_forward(i, orig_layer_forward).__get__(layer, type(layer))


def entropy_to_skip_count(H):
    if H < 0.1:  return 16
    if H < 0.5:  return 12
    if H < 1.0:  return 8
    if H < 2.0:  return 4
    return 0


def update_skip_mask(H):
    n = entropy_to_skip_count(H)
    skip = set(SKIP_PRIORITY[:n])
    for i in range(len(State.skip_mask)):
        State.skip_mask[i] = (i in skip)


# ---- Step 4D: quality gate ----
def run_decode(prompt, n_decode, adaptive=False):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=ids.shape[1]+n_decode+4,
                        device="cuda", dtype=torch.float16)
    with torch.no_grad():
        cp = torch.arange(ids.shape[1], device="cuda", dtype=torch.long)
        out = model(input_ids=ids, cache_position=cp,
                    past_key_values=cache, use_cache=True, return_dict=True)
    cur = out.logits[:, -1:].argmax(-1)
    cp = torch.tensor([ids.shape[1]], device="cuda", dtype=torch.long)
    gen = []; logits_hist = []; entropies = []
    for _ in range(n_decode):
        with torch.no_grad():
            o = model(input_ids=cur, cache_position=cp,
                      past_key_values=cache, use_cache=True, return_dict=True)
        logits = o.logits[0, -1].float()
        p = F.softmax(logits, dim=-1)
        H = -(p * (p.clamp_min(1e-30).log())).sum().item()
        entropies.append(H)
        logits_hist.append(logits.cpu())
        cur = o.logits.argmax(-1)
        gen.append(int(cur.item()))
        cp += 1
        if adaptive:
            update_skip_mask(H)
    torch.cuda.synchronize()
    del cache
    return gen, logits_hist, entropies


print(f"\n=== Step 4D: quality gate ({N_PROMPT_QUAL} prompts × {N_DECODE_QUAL} tokens) ===")
PROMPTS_QUAL = [
    "The future of GPU computing is",
    "Energy efficiency in modern systems means",
    "When training a neural network you should",
    "The Roman empire fell because",
    "Quantum mechanics describes",
]
total_kl = 0.0
total_compare = 0
for prompt in PROMPTS_QUAL:
    State.skip_mask = [False] * len(model.model.layers)
    State.n_skipped = State.n_run = 0
    gen_full, logits_full, _ = run_decode(prompt, N_DECODE_QUAL, adaptive=False)
    n_run_full = State.n_run; n_skipped_full = State.n_skipped
    text_full = tok.decode(gen_full)

    State.skip_mask = [False] * len(model.model.layers)
    State.n_skipped = State.n_run = 0
    gen_adap, logits_adap, ents = run_decode(prompt, N_DECODE_QUAL, adaptive=True)
    n_run_adap = State.n_run; n_skipped_adap = State.n_skipped
    text_adap = tok.decode(gen_adap)

    # KL per token
    for i in range(len(logits_full)):
        p = F.log_softmax(logits_full[i], dim=-1).exp()
        log_q = F.log_softmax(logits_adap[i], dim=-1)
        log_p = F.log_softmax(logits_full[i], dim=-1)
        kl = (p * (log_p - log_q)).sum().item()
        total_kl += kl
        total_compare += 1
    skip_frac = n_skipped_adap / max(1, n_skipped_adap + n_run_adap)
    print(f"  prompt: {prompt!r}")
    print(f"    full:    {text_full[:80]!r}")
    print(f"    adapt:   {text_adap[:80]!r}")
    print(f"    skip-fraction = {skip_frac:.1%}  ({n_skipped_adap} skipped / "
          f"{n_run_adap+n_skipped_adap} layers visited)")

mean_kl = total_kl / total_compare
print(f"\n  mean KL(full || adaptive) = {mean_kl:.4f} nats per token")
print(f"  spec target: < ~0.1 (5% perplexity-degradation proxy)")
print(f"  result: {'PASS' if mean_kl < 0.1 else 'FAIL — quality degraded'}")

with open(os.path.join(ROOT, "p4d_quality.json"), "w") as f:
    json.dump(dict(mean_kl=mean_kl, n_compared=total_compare,
                   skip_priority=SKIP_PRIORITY[:20]), f, indent=2)
print(f"[p4cd] wrote p4d_quality.json")
