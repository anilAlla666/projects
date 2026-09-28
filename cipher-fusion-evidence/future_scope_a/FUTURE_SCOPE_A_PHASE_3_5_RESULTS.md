# FUTURE_SCOPE / A — PHASE 3.5 — CIPHER DVFS ACTUATOR UNDER vLLM — RESULTS

**Date:** 2026-05-19. **Verdict: DVFS composes with vLLM and lifts tok/W —
modestly (+13.9 %, 1.14×) at the sweet spot, NEGATIVE if over-cut.** Below the
1.5× gate → per the gate's own fallback, **DVFS-alone under vLLM is a modest
secondary lever; the primary lift mechanism is multi-tenant density.** Anchors
unchanged (measurement only — CIPHER_VOLT is an existing actuator engaged via
env).

---

## Method

`phase3_5_vllm_probe.py` + `phase3_5_sweep.py`: vLLM 0.21.0, TinyLlama-1.1B,
graph mode (the Phase-3-verified production path), greedy decode, 128 tokens.
5 reps/config, fresh vLLM process, `nvidia-smi -rgc` clock reset between reps.
CIPHER's DVFS engaged via `CIPHER_VOLT=on` + `CIPHER_VOLT_MHZ` — libcipher_rt
locks the clock through the kmod `CIPHER_SET_CLOCK_MHZ` ioctl (NVML path
returns NOT_SUPPORTED on this pod; kmod path used).

## Results — 5 reps each

| config | tok/s | power (W) | SM clock | tok/W | tok/W lift |
|---|---|---|---|---|---|
| **vLLM alone (default)** | 620.7 | 137.7 | 1980 MHz | **4.510** | (baseline) |
| **vLLM + CIPHER DVFS 1200 MHz** | 522.3 | 101.6 | 1200 MHz | **5.139** | **1.139× (+13.9 %)** |
| vLLM + CIPHER DVFS 800 MHz | 356.8 | 90.6 | 810 MHz | 3.938 | **0.873× (−12.7 %)** |

- **token_ids byte-identical** across every config and rep — clock speed does
  not change FP results; the correctness gate holds trivially.
- Clocks locked precisely (1980 / 1200 / 810 MHz, std 0) — the actuator works.

## Reading it honestly

- **DVFS does compose with vLLM and does lift tok/W — but only ~14 %.** At
  1200 MHz: power drops 26 % (137.7→101.6 W), tok/s drops 16 % (620.7→522.3),
  net tok/W **+13.9 %**.
- **Over-cutting is counterproductive.** At 810 MHz: tok/s collapses 42.5 %
  while power drops only 34 % → tok/W goes **negative, −12.7 %**. This is the
  [[cipher-t43-envelope]] "over-aggressive cut" finding, reproduced under vLLM.
  There is a sweet spot near 1200 MHz; aggressive DVFS loses.
- **The +14 % is far below the 1.5× gate** — and that was the predicted
  outcome (stated before the run): the campaign's DVFS envelope showed the
  +55 % figure came from the *eager HF harness* where the GPU idled 70–95 %
  per token. vLLM graph mode is ~9× more memory-efficient — far less idle
  headroom for a clock cut to reclaim. The +14 % is what is genuinely there.
- **The lift is clock-lock physics, not unique to CIPHER.** A `1200 MHz`
  clock lock via plain `nvidia-smi -lgc` would give the same +14 %. CIPHER's
  value-add is doing it *automatically and per-workload-calibrated* — not the
  +14 % itself.

## Actuator-composition scorecard — CIPHER under vLLM, single-instance TPW

| CIPHER actuator | composes with vLLM? | single-instance TPW effect |
|---|---|---|
| **DVFS (VOLT)** | ✅ yes — external clock actuator | **+13.9 % (1.14×)** at 1200 MHz; negative if over-cut |
| **Marlin INT4** | ❌ no — vLLM ships its own fused kernels; no cuBLAS GEMM for CIPHER's `.symver` to intercept | none under vLLM |
| **persistent dispatch** | ❌ no — bypassed by vLLM's CUDA-graph capture | none under vLLM |
| **L2 persistence** | ⚠️ not investigated in Phase 3.5 | DEFERRED — flagged, not claimed |
| SM partitioning | (composes — Phase 5) | isolation, not single-instance TPW |
| weight sharing | (composes via Phase 4 build) | density, not single-instance TPW |

**Net: the only CIPHER actuator that lifts a single vLLM instance's tok/W is
DVFS, at ~1.14×.** Marlin and persistent dispatch — the actuators that *would*
accelerate compute — do not fire under vLLM (vLLM brings its own kernels and
graphs; CIPHER's interception surface for those is not present).

## What this resolves — for the four-metric scorecard

The TPW cell, **CIPHER+vLLM-composed context**, is now honestly fillable:

> **Single-instance TPW under vLLM: ~1.14× (DVFS only, sweet-spot clock-lock).
> Not a multiplier.** CIPHER does not substantially accelerate a single vLLM
> instance — the compute-acceleration actuators (Marlin, persistent dispatch)
> have no interception surface under vLLM. **The real TPW story is
> multi-tenant:** consolidation (Phase C ~1.8× vs vLLM-multi-instance) +
> density-enabled fleet efficiency — measured properly in Phase 4 (weight
> sharing) and CP 5.5.

Per the Phase 3.5 gate's explicit fallback (lift < 1.5×): **the pitch's
primary lift mechanism is multi-tenant density, not single-instance actuator
acceleration.** DVFS is a real but modest (+14 %) secondary lever, and only on
DVFS-favorable workloads.

## Verdict

DVFS composes with vLLM and delivers a genuine but modest **+13.9 % tok/W** at
the 1200 MHz sweet spot — correctness-safe, 5-rep clean. It is **not** a 1.5×
lever and aggressive cuts go negative. The honest conclusion: under vLLM,
CIPHER does not boost a single instance's TPW by a meaningful multiple — the
TPW headline must be the multi-tenant consolidation/density story, with DVFS a
+14 % secondary contribution.

**STOP for adjudication before Phase 4** (weight sharing — the density lever,
the actual headline). Evidence: `phase3_5_vllm_probe.py`, `phase3_5_sweep.py`,
`phase3_5_sweep_result.json`, `p35_*.json/.log`.
