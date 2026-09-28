# Phase A — Baseline Three-Pillar Matrix — Execution Plan

**Goal:** measure all 24 workloads (WL01-WL24) on substrate `a7ac8e97` across
three pillars — MFU, TPW, multi-tenant — via the CP 5.6 P2 teacher-forced
methodology extended to three passes. Output:
`cipher-fusion-evidence/cp_5_6/PILLAR_MATRIX.md`.

This file is the *engineering input* the operator's spec doesn't fix: the
per-workload config/regime classification, the methodology decisions, and the
execution order. The operator's message is the authoritative spec; this plan
does not restate it.

## Methodology decisions (locked, documented here so they are reviewable)

1. **MFU = analytical FLOPs, not CUPTI per-kernel counting.** `nvprof` is
   removed on H100/CUDA 13; CUPTI per-kernel FLOP attribution is fragile and
   non-reproducible. Use the standard analytical MFU: a transformer forward
   pass ≈ `2 · N_params · N_tokens` FLOPs; training adds backward ≈ `6 · N ·
   tokens` total; diffusion/encoder use their own param×token(or pixel)
   product. MFU = achieved FLOPs/s ÷ **H100 SXM5 dense FP16 tensor peak
   989.4 TFLOP/s** (no sparsity). The denominator is documented per row.
2. **Teacher-forced ≥99% gate applies to the decode regime only.** For
   non-decode regimes the gate is recorded **N/A — <reason>** (per the
   operator's "doesn't fit a pass → document and skip"); PASS 1 still reports
   MFU + throughput + power, and a regime-appropriate lightweight numeric
   spot-check vs vanilla (last-token logit argmax for forward-only; training
   loss delta; output-tensor cosine for encoders) is recorded as
   `numeric_check` — not a gate, just honest substrate-vs-vanilla evidence.
3. **Power = arm-level mean** (CP 5.6 P2 A1 fix), 1 Hz `nvidia-smi`.
4. **Substrate `a7ac8e97`**; `libcipher_v2 86618c30` auto-injected by
   `tenant_register` (this is the CP 5.6 P2 "vanilla" condition — matched).
5. **N=5 matched pairs per arm**, 3 arms (vanilla / marlin / all-on). Token
   budgets per workload below; scoped down + documented if a single arm
   exceeds 30 min.

## Per-workload config + pass applicability

regime key: D=decode (TF gate), F=forward-only, T=training, E=encoder,
X=diffusion/asr, V=vLLM, M=multitenant, DEF=deferred.

| WL | model (substitute) | B | seq/ctx | regime | PASS 1 gate | PASS 2 multi-tenant |
|----|----|----|----|----|----|----|
| 01 | TinyLlama-1.1B | 1 | gen 32 | D | TF ≥99% | yes (4×) |
| 02 | TinyLlama-1.1B | 8 | gen 32 | D | TF ≥99% (per-row) | yes (4×) |
| 03 | Mistral-7B | 8 | prefill 1024 | F | N/A — prefill, no decode | yes (4×) |
| 04 | TinyLlama / vLLM | — | gen 32 | V | TF via vLLM logprobs (adapt) | yes (4×) |
| 05 | 8× TinyLlama | 1 | gen 32 | M | TF ≥99% per tenant | IS the multi-tenant case |
| 06 | all-MiniLM-L6-v2 | 512 | enc | E | N/A — encoder | yes (4×) |
| 07 | TinyLlama + LoRA | 4 | train | T | N/A — training (loss delta) | yes (4×) |
| 08 | SDXL | 1 | 20 steps | X | N/A — diffusion, no tokens | yes (4×) |
| 09 | Whisper | 1 | 60 s audio | X | N/A — ASR (decoder adapt later) | yes (4×) |
| 10 | Mistral+TinyLlama / vLLM | — | gen 64 | V | TF via vLLM logprobs (adapt) | yes (4×) |
| 11 | TinyLlama | 1 | gen 20 ×40 | D | TF ≥99% | yes (4×) |
| 12 | TinyLlama | 64 | prefill 128 | F | N/A — forward-only | yes (4×) |
| 13 | Mistral-7B | 1 | ctx 32768 | F | N/A — forward-only | mem-limited; document |
| 14 | TinyLlama (torch.compile) | 8 | prefill | F | N/A — forward-only | yes (4×) |
| 15 | TinyLlama (MoE proxy) | 4 | gen 16 | D | TF ≥99% | yes (4×) |
| 16 | TinyLlama / vLLM | 8 | gen | V | TF via vLLM logprobs (adapt) | yes (4×) |
| 17 | TinyLlama full-train | 4 | train | T | N/A — training (loss delta) | yes (4×) |
| 18 | multi-GPU TP | — | — | DEF | DEFERRED — single-GPU pod | DEFERRED |
| 19 | CLIP ViT-B/32 | 32 | image | E | N/A — vision encoder | yes (4×) |
| 20 | LLaVA-1.5-7B | 1 | gen 20 | D | TF ≥99% (image-conditioned gold) | yes (4×) |
| 21 | TinyLlama | 1 | gen 64 | D | TF ≥99% | yes (4×) |
| 22 | MiniLM + TinyLlama | 1 | gen 20 | D | TF ≥99% (LLM stage) | yes (4×) |
| 23 | TinyLlama / Mistral | 1 | gen 16 | D | TF ≥99% (per active model) | yes (4×) |
| 24 | TinyLlama-AWQ | 1 | gen 32 | D | TF ≥99% (AWQ vs FP16 gold) | yes (4×) |

Decode/TF-gateable: WL01,02,05,11,15,20,21,22,23,24 (10). Forward-only:
WL03,12,13,14 (4). Training: WL07,17. Encoder: WL06,19. Diffusion/ASR:
WL08,09. vLLM: WL04,10,16. Deferred: WL18.

## Execution order (cleanest / fastest first; report every 4-6)

- **Batch 1** — clean HF decode, base frameworks cached: WL01, WL02, WL11,
  WL15, WL21, WL23. Proves the generalized PASS-1 harness.
- **Batch 2** — forward-only, base frameworks: WL03, WL12, WL13, WL14.
- **Batch 3** — training + encoders: WL07, WL17, WL06, WL19
  (WL06/WL19 need `sentence-transformers` / CLIP weights — install or skip).
- **Batch 4** — decode needing adaptation: WL22, WL20, WL24, WL05
  (WL22 sentence-transformers; WL20 LLaVA weights; WL24 `autoawq`;
  WL05 multi-tenant overlaps PASS 2).
- **Batch 5** — vLLM: WL04, WL10, WL16 (needs `vllm` install).
- **Batch 6** — heavy frameworks: WL08 (diffusers+SDXL), WL09 (whisper).
- WL18 — documented deferred, no run.

A workload whose framework install fails or whose harness needs more than a
bounded adaptation is **flagged and skipped**, revisited at the end.

## Harness

- `pillar_driver.py` — generalized from `tf_gate_driver.py`: workload config
  (model, batch, seq/ctx, max_new, regime) selects gold capture + free-run +
  (decode) teacher-forced gate + analytical FLOPs/MFU. PASS 1.
- `run_pass1_<WL>.sh` per workload (or one parameterized orchestrator).
- `run_pass2_multitenant.sh` — 4 concurrent green-context tenants, aggregate
  tok/s, tok/W, MFU (uses libcipher_v2 GREEN partitioning — the CP 5.3 Step 2
  Axis A path).
- `analyze_pillar.py` — per-workload three-pillar rollup → `PILLAR_MATRIX.md`.
- Evidence under `cipher-fusion-evidence/cp_5_6/phase_a/`.

## Tracking

One row appended to `PILLAR_MATRIX.md` as each workload completes; progress
reported to the operator every 4-6 workloads.
