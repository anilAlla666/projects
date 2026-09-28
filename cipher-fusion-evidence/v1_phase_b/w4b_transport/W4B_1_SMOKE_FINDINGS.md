# W.4b.1 — smoke-reproduce baseline — FINDINGS

**Date:** 2026-05-28. **Anchor:** cipher_rt_phase4 `8b5e928` (w4a-pool-eligibility),
libcipher_rt.so md5 `9fe23143`. **Workload:** WL01 TinyLlama-1.1B, N=4, gen_len=128,
5 prompts. **Executor:** `cipher_batch_executor.py` (re-ported to this dir from
`cp_5_6/phase_b/session1/`).

## Result

| Run | agg tok/s | steady-state tok/s (round 2) | wall_s | correctness |
|---|---|---|---|---|
| **Vanilla** (no injection) | 206.7 | 512/1.79 = **286** | 10.2 | KL_max ≤ 5.5e-5 |
| **W.4a substrate injected** (`CUDA_INJECTION64_PATH`) | 78.5 | 512/5.00 = **102** | 26.9 | KL_max ≤ 5.5e-5 |

- **Lever survives.** Vanilla steady-state 286 tok/s matches the CP 5.6 in-process
  B=4 ceiling (285 tok/s, `PHASE_B_CROSS_TENANT_BATCHING_DESIGN.md` §1). The
  re-ported executor + Unix-socket IPC + client are healthy on the current pod
  (4/4 clients, all rounds, no crash, `batch_efficiency=1.0`).
- **Correctness preserved on the substrate.** Teacher-forced per-row logit-KL
  max ≤ 5.5e-5 across all 5 prompts in BOTH runs — passes the CP 5.6 gate.

## Critical finding — injecting the substrate into the executor costs ~2.6×

Injecting the W.4a substrate via `CUDA_INJECTION64_PATH` slows batched decode
286 → ~102 steady-state tok/s (206.7 → ~78.4 aggregate).

**Cold-cache ruled out.** A 3rd, fully-warm injected trial reproduced it exactly
(agg 78.3 tok/s; steady rounds 5.0–5.15s vs vanilla 1.82–1.86s). Consistent
across warm rounds ⇒ real injection overhead, not first-run effects.

**Clocks (sampled during each arm, 0.25s cadence):**

| Arm | steady tok/s | clocks.sm median (max) | clocks.mem | power median |
|---|---|---|---|---|
| Vanilla | 286 | **1980 (1980, pinned max)** | 2619 | 120 W |
| Injected | 102 | **1605 (1980)** | 2619 | 110 W |

**Cause (isolated to a dominant mechanism + a secondary one):**
- The clock drop (1980→1605 median, ~1.23×) accounts for only ~1.23× of the
  2.6× gap. So it is **not** primarily a VOLT/clock-lock effect.
- The dominant cost (~2.1× of the 2.6×) is **CPU-side cuBLAS-shim overhead paid
  by any injected process** — every GEMM traverses the GOT-patched shim
  (observe + actuator-dispatch checks). Math: ~3.2 s extra / 128 steps =
  25 ms/step over ~132 batched GEMMs/step ≈ **~190 µs/GEMM** of host-side
  interception. B=N decode is GEMM-dense (little launch-idle to hide it),
  unlike single-tenant B=1 (the VOLT regime, where injection is cheap).
- The lower injected clock/power (1605 MHz, 110 W < vanilla 1980 MHz, 120 W) is
  consistent with the GPU **starving** while the host does shim work between
  kernels (DVFS drops idle clocks) — i.e. clock-droop is an *effect* of the
  CPU-bound shim, not an independent cause.
- **Not chased:** the exact W.4a-pool-observe-vs-pre-existing-shim split (would
  need a pre-W.4a libcipher_rt build). Immaterial to the design fix below,
  because the fix removes shim cost wholesale.

## Design implication for W.4b.2 — the executor runs VANILLA

The load-bearing decision is **whether the executor injects libcipher_rt at
all**, not where the pool query lives. Since the 2.6× is whole-shim overhead any
injected process pays, only a vanilla executor recovers it. In Form A the
executor *is* a substrate process, so:

- **Executor runs VANILLA** (no `CUDA_INJECTION64_PATH`). Its hot path pays zero
  shim cost by construction. It queries pool eligibility via a **/dev/cipher
  side channel** (ioctl / `cipher_rt_pool_group_get` state read) at
  **batch-formation granularity** (once per round), and runs
  `_correctness_check` / `_mark_blocked` per coalesce-group — never per GEMM.
- **Only the CLIENT tenants inject libcipher_rt**, for model-fingerprint
  registration into the W.6sC cohort (that is what the eligibility partition
  keys on). The clients are thin socket clients — no GEMM hot loop — so their
  injection cost is negligible.
- This matches CP 5.6 design §2.1 (fuse at decode-step granularity, reject
  per-GEMM cross-process fusion) and discipline (j) (barrier substrate-side).

**Open item to verify before committing W.4b.2:** confirm nothing in the
executor's batched hot path *requires* libcipher_rt loaded. For W.4b the lever
is the cross-tenant batching orchestration (CP 5.6 measured it vanilla); the
per-launch actuators (Marlin/VOLT) on the batched GEMMs are actuator-composition
= `FUTURE_SCOPE/E`, explicitly out of W.4b scope. So the expected answer is "no
— executor needs no injection for W.4b." Verify, then commit the design.

## W.4b.1 verdict

**PASS (lever + plumbing + correctness survive; vanilla steady-state 286 tok/s
matches the CP 5.6 B=4 ceiling, KL ≤ 5.5e-5).** One load-bearing design decision
surfaced for W.4b.2: **the executor runs vanilla; only client tenants inject
libcipher_rt (for fingerprint registration); the executor queries pool
eligibility via a /dev/cipher side channel at batch-formation granularity.**
No HARD STOP. Next: verify nothing in the executor hot path needs libcipher_rt
(expected: no, per FUTURE_SCOPE/E), then W.4b.2 wires the side-channel gate.
