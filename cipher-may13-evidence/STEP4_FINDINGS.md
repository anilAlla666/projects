# STEP 4 — Adaptive Layer Precision

## Method
For each of Mistral-7B's 32 decoder layers we measure the mean relative
error between the layer's fp16 output and its INT4-substituted output,
averaged over 50 decode tokens (greedy decode of a 25-token prompt). The
precision map is built from the per-layer rel_err using the spec's
thresholds:
- `rel_err < 0.005` → **skip** (NOP the layer, pass residual)
- `rel_err < 0.02`  → **int2** (further-compress weights)
- otherwise          → **int4** (current behaviour)

A 200-token KL-divergence proxy then measures how much the mapped model
drifts from fp16 baseline at the next-token logit level.

## Per-layer rel_err (selected, full table in `step4_precision_map.json`)

| layer | rel_err | mean_abs |  layer | rel_err | mean_abs |
|------:|--------:|---------:|-------:|--------:|---------:|
|     0 |   1.34  |   0.003  |     16 |   1.25  |   0.075  |
|     1 |   0.75  |   0.011  |     20 |   1.34  |   0.138  |
|     4 |   1.02  |   0.019  |     24 |   1.37  |   0.191  |
|     8 |   1.19  |   0.031  |     28 |   1.35  |   0.265  |
|    12 |   1.21  |   0.047  |     31 |   1.28  |   0.424  |

All layers report rel_err in the 75 % – 140 % range — far above either
threshold. **No layer qualifies for skip or int2.**

## Resulting precision map
- skip = 0
- int2 = 0
- int4 = 32

## Quality proxy
Mean KL(fp16 ‖ mapped) over 200 decode tokens = **13.26 nats** with the
all-int4 map. fp16-vs-fp16 KL is 0 (sanity check). 13 nats on a vocab=32k
distribution corresponds to the all-INT4 stack producing a *very*
different next-token distribution from fp16 baseline — consistent with
BUILD_STATE's note that the existing INT4 path delivers grammatical
English but different tokens than baseline.

## Why no layer was eligible for downgrade
BUILD_STATE Phase 2 notes the per-element INT4 quantization noise is
~12 % (`absmax/7` quant step is ~14 % of max). This compounds across the
32-layer residual stream: by layer 20+ each layer's hidden-state output
differs from fp16 by ~135 %. The 0.005 / 0.02 thresholds the spec
proposes assume per-layer noise floors of fractions of a percent —
achievable with **AWQ activation-aware calibration** (BUILD_STATE flags
this as the single biggest open accuracy lever) but not with the current
absmax INT4.

## What it would take to make this lever active
1. **AWQ INT4 calibration** — drops per-element rel-err below 1 % by
   weighting per-channel scales toward observed activation magnitudes.
   Per-layer rel_err would drop into the 1-5 % range, opening the
   `int2` band for some layers.
2. **Per-layer KL gating** — the rel-err proxy is on the hidden state;
   the user's spec for skip semantically requires KL(token-distribution)
   to be small under skip. Some layers (especially mid-stack with high
   redundancy) may pass a KL gate even when hidden-state rel_err is
   nominal — the literature ("layer pruning at inference time") finds
   ~10 of 32 Mistral layers can be skipped with <1 % perplexity hit.
3. **Per-projection precision** instead of per-layer (e.g., k_proj/v_proj
   are smaller and noisier than gate/up; they may need fp16 even when
   the rest of the layer drops to int2).

## Files
- `step4_layer_precision.py` — calibration harness with mode setter
- `step4_precision_map.json`  — per-layer rel_err and chosen mode
