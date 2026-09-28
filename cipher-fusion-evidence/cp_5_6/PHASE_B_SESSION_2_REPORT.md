# Phase B — Build Session 2 (Steps 1–3): Cross-Tenant Batching Scale N=4/8/16

**Date:** 2026-05-18. Workload WL01 (TinyLlama-1.1B, B=1 greedy decode,
5 prompts × 128 tok). Single H100. Substrate anchor `a7ac8e97` **unchanged**
(prototype is Python-layer, per Session 1).

**Headline: the ≥3.6× substrate-attributable tok/W target is VERIFIED.**
At N=8, cross-tenant batching delivers **3.69× substrate-attributable**
tok/W (batched executor vs naive 8-concurrent), under a teacher-forced
correctness gate (KL_max 5.5e-5 ≪ 0.1). N=16 confirms wide headroom (8.7×).

## Results — Steps 1–3

Substrate-attributable = batched-executor tok/W ÷ naive-N-concurrent tok/W
(the like-for-like denominator). Naive controls measured via
`run_multitenant.sh N=<n> vanilla`. In-process ceiling = the WL01 batch-scan.

| N | batched tok/W | naive N-concurrent | **substrate-attributable** | vs 1-tenant baseline | % of in-process ceiling | TF KL_max |
|---|---|---|---|---|---|---|
| 4 | 1.750 | 0.994 | **1.76×** | 3.59× | 91.1 % | 5.5e-5 |
| 8 | 3.342 | 0.906 | **3.69×** | 6.85× | 88.4 % | 5.5e-5 |
| 16 | 6.758 | 0.775 | **8.72×** | 13.85× | 92.7 % | 5.5e-5 |

Power held ~144–148 W across all N (batching converts idle time to work at
near-constant draw). Cross-process plumbing reaches 88–93 % of the in-process
batched ceiling — the ~5–12 % gap is socket IPC + the static barrier.

**Success criteria (Session 2):**
- N=8 aggregate tok/W ≥ 3.6× single-tenant baseline → **6.85× — PASS**
- substrate-attributable ≥ 3× at N=8 → **3.69× — PASS**
- per-tenant correctness preserved at all N → **KL_max 5.5e-5, gate PASS at N=4/8/16**
- (Steps 4–5 — heterogeneous batch, dynamic scheduler — not yet built.)

The substrate-attributable lift compounds with N: 1.76× → 3.69× → 8.72×. It
rises faster than linear because the naive-concurrent baseline *degrades* with
N (0.994 → 0.906 → 0.775 tok/W — N independent processes contend for the GPU),
while the batched executor scales cleanly. Both effects are real and measured.

## Mid-session correction — the correctness gate (transparent record)

The first N=16 run reported KL_max **10.49** — a hard stop. Diagnosis: **2 of
80 (tenant, prompt) pairs** spiked while 78 were ~6e-6. Root cause was a
**gate-methodology bug in the prototype, not the substrate**: the executor
computed KL on *free-running* logits vs gold logits. When batched FP-noise
flips one argmax at a near-tie, that row's autoregressive decode cascades away
from gold, and position-by-position KL against gold becomes meaningless — the
exact "mechanically broken gate" CP 5.6 P2 identified.

Fix: the executor's correctness pass was rewritten to **teacher-forced** — a
B=N forward over each prompt's clean-FP16 gold sequence, per-step KL with
every position conditioned on the true gold prefix (cascade-free, the CP 5.6
P2 methodology). `cipher_batch_executor.py.pre_tf_fix` preserves the prior
version. Re-run N=4/8/16: KL_max **5.5e-5** at every N — correctness is in
fact preserved; the 10.49 was purely the gate artifact. Throughput numbers
were unaffected by the bug (free-run, separately measured) and are unchanged.

This is logged transparently because it briefly looked like an N=16 failure
and is not: the substrate batches correctly at N=16; the prototype's gate was
unsound and is now sound.

## What is verified, and what is not

**Verified:** cross-tenant decode-step batching, measured end-to-end across
processes, delivers ≥3.6× substrate-attributable tok/W at N=8 on WL01, with a
sound teacher-forced correctness gate passing. This is the engineering target,
met — measured, not projected.

**Not yet verified (honest scope boundary):**
- *One workload.* WL01 = TinyLlama-1.1B B=1 decode. Larger models (7B-class)
  saturate the batch curve at lower N — the N at which ≥3.6× lands is
  regime-dependent and must be measured per workload class.
- *Identical replicated prompts.* All tenants run the same 5 prompts → equal
  lengths, no padding. **Step 4 (heterogeneous batch)** is the real-world test.
- *Static barrier.* Tenants are synchronised per round; **Step 5 (dynamic
  scheduler)** is needed for asynchronous tenants and tail-latency control.
- *Vanilla FP16.* The batching lever earns ≥3.6× **on its own** — Marlin/DVFS
  are separate, not in this path. Composing batching with the quantization
  substrate is later integration.

## Status & next

Steps 1–3 complete; the headline target is verified. Steps 4 (heterogeneous
batch, <15 % padding cost) and 5 (dynamic admission scheduler, tail-latency
preserving) remain — the production-hardening half of Session 2. Per the
plan, Track 2 (weight-sharing) may now run in parallel.

## Artefacts

`phase_b/session1/` — `cipher_batch_executor.py` (+ `.pre_tf_fix`),
`batch_client.py`, `run_batch_n.sh`, `analyze_session2.py`,
`session2_summary.json`, `n{4,8,16}/` (executor results, client JSONs, power
traces, per-N `result.json`). Naive controls: `phase_a/WL01/mt{4,8,16}_vanilla/`.
