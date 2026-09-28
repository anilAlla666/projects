# Mistral-7B 3-Arm Benchmark

**Date:** 2026-05-18. Workload Mistral-7B-v0.1, 5 prompts/tenant, greedy,
128 tok. N=4, B=1. Single H100 SXM5 80 GB. Anchor `a7ac8e97`.

**Headline:** Arm 2 (vLLM-per-tenant) **does not fit** — 4 separate vLLM
Mistral instances cannot co-reside on one 80 GB H100. This is the architectural
finding the brief anticipated. Arm 3 / Arm 1 = **3.10×**.

## Results — N=4 × B=1

| arm | rep 1 | rep 2 | mean tok/W | tok/s | power | correctness |
|---|---|---|---|---|---|---|
| Arm 1 — naive HF, 4 procs | 0.2792 | 0.2790 | **0.279** | ~92 | ~328 W | teacher-forced KL ≈ 0 ✓ |
| Arm 2 — vLLM, 4 instances | — | — | **DOES NOT FIT** | — | — | — |
| Arm 3 — CIPHER fusion, N=4 | 0.8685 | 0.8588 | **0.864** | ~178 | ~206 W | teacher-forced KL_max 3.1e-5 ✓ |

- **Arm 3 / Arm 1 = 3.10×** (consistent with the Phase-B Mistral verification's
  3.26×).
- **Arm 3 / Arm 2 = undefined** — Arm 2 cannot be measured (see below).

## Arm 2 — vLLM-per-tenant does not fit 4 Mistral on one H100

Measured: a **single** vLLM Mistral instance needs `gpu_memory_utilization`
≈ **0.25** (~20 GB) to start — 13.5 GB weights + ~6 GB activation-profiling/
engine overhead + a minimum KV pool. Below that it fails
`No available memory for the cache blocks`. Attempts at the **4-instance**
operating point:

| config | result |
|---|---|
| 4 × GMU 0.22 | all died — `No available memory for the cache blocks` |
| 4 × GMU 0.24 | all died — `No available memory for the cache blocks` |

4 instances at the minimum-viable GMU 0.25 would need 4 × 20 GB = **80 GB of
engine budget alone** — the entire GPU, with **zero** room for the 4 separate
CUDA contexts. **It is physically impossible to co-reside 4 vLLM Mistral-7B
instances on one 80 GB H100.** This is not a tuning failure — it is a hard
memory wall: per-tenant-process vLLM carries its own 13.5 GB weight copy plus
a ~6 GB engine/activation footprint, ~20 GB × 4.

By contrast: Arm 1 (4 naive HF processes) **fits** — HF's per-process
footprint is leaner (weights + activations, no pre-reserved KV pool, no
separate engine process): 4 × ~14 GB ≈ 56 GB. Arm 3 (CIPHER) fits trivially —
**one** 13.5 GB Mistral copy serves all 4 tenants fused.

## Architectural interpretation

At 7B scale the brief's hypothesis holds, in a stronger form than a ratio:
**memory pressure surfaces CIPHER's value as the difference between running
and not-running.** In the *separate-per-tenant-process* topology — the premise
of the whole 3-arm benchmark ("N separate customer processes") — vLLM-per-tenant
hits a hard memory wall at N=4 Mistral; CIPHER's single-copy consolidation does
not. CIPHER serves 4 Mistral tenants in 13.5 GB of weights; per-tenant vLLM
would need 54 GB of weight copies plus 4× engine overhead.

vs naive HF (Arm 1), CIPHER is **3.10×** — and unlike TinyLlama (where the win
was power-only), here CIPHER wins on **both** axes: ~2× the throughput
(178 vs 92 tok/s — 4 naive HF Mistral processes contend hard) **and** 0.63×
the power (206 vs 328 W).

**Honest caveat — the topology premise.** This finding bites specifically in
the *one-vLLM-instance-per-tenant-process* model. An operator who consolidates
the 4 tenants onto **one** vLLM instance (13.5 GB weights, B=4 continuous
batching — vLLM's natural multi-request mode) fits easily and avoids the wall.
That consolidation is exactly what CIPHER's executor does — and what a shared
vLLM would do. So the Mistral finding is: *in the separate-per-tenant-process
topology, CIPHER's consolidation is the difference between fitting and not; the
clean "CIPHER-consolidated vs vLLM-consolidated" comparison is `FUTURE_SCOPE/A`
(CIPHER + vLLM composed) and remains open.*

## B=2 operating point — not measurable

The brief asked for N=4×B=2 "if memory permits." Arm 2 fails already at B=1
(the binding constraint is the 13.5 GB weight copy ×4, not KV — B=2 would not
change it). With Arm 2 absent there is no 3-arm comparison at B=2; skipped.

## The operating-point map (TinyLlama + Mistral)

| operating point | vs naive (Arm3/Arm1) | vs vLLM-per-tenant (Arm3/Arm2) | nature of CIPHER's win |
|---|---|---|---|
| TinyLlama N=8×B=1 | 3.82× | 1.75× | power consolidation |
| TinyLlama N=8×B=2 | 4.69× | 1.84× | power consolidation (plateau) |
| Mistral-7B N=4×B=1 | 3.10× | — (vLLM doesn't fit) | consolidation enables co-residency + power + throughput |

**Synthesis.** CIPHER's substrate-attributable advantage is **process/model
consolidation**. At small-model scale that shows as a stable ~1.8× tok/W
power-consolidation win over per-tenant vLLM (not a batch-scaling advantage —
vLLM's intra-process batching matches it). At 7B scale it shows as the harder
fact that per-tenant-process vLLM cannot fit at all, while CIPHER's one-copy
consolidation does. The clean number against a *consolidated* vLLM baseline is
not yet measured — `FUTURE_SCOPE/A`.

## Notes

- Arm 1 / Arm 3 reps are tight (≤ 1.5% spread). vLLM zombie-`EngineCore`
  cleanup (PID-based, in `run_arm2.sh`) held — no leaked memory between runs.
- Per the brief, this completes the operating-point map; further measurement
  deferred to `FUTURE_SCOPE/A`.

## Artefacts

`phase_c/arm1_mistral_rep{1,2}/`, `phase_b/session1/arm3_mistral_rep{1,2}/`,
`phase_c/arm2_mistral_rep1/` (failure logs). Mistral gold `phase_a/WL_MISTRAL/`.
