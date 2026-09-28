# R-I1 MPS validation — VERDICT: **OUTCOME 1 (PASS) — MPS + green-ctx achieves 1-H100 cross-tenant p99 isolation**

**Date:** 2026-05-29. bf16, clean diag2 harness (`d8_ri1_gateb.py`) re-run **under the CUDA MPS
server** (Default compute mode, no sudo). Measurement only — anchor UNCHANGED. The disjoint
green-ctx partition, **inert without MPS**, **delivers full p99 isolation under MPS**.

## Gate B under MPS — victim wall p99 (ms)

| condition | victim p50 | victim p99 | note |
|---|---|---|---|
| C1 victim SOLO, full GPU | 0.033 | **0.052** | floor |
| C2 victim SOLO, 8-SM partition | 0.098 | **0.114** | the partition floor (target for PASS) |
| C3 victim + aggressor, **NO** partition (MPS) | 1.038 | **1.148** | 22× — MPS concurrency alone does NOT isolate |
| C4 victim + aggressor, **DISJOINT** partition (MPS) | 0.113 | **0.130** | **1.14× C2 floor, 2.5× C1 — ISOLATED** |

- **PASS condition met:** C4 victim p99 (0.130 ms) ≈ C2 8-SM solo floor (0.114 ms), **not** near
  C3 (1.148 ms). Recovered from the non-MPS C4 of **1.959 ms (40×)**.
- **Aggressor kept ~full work:** 507 matmul/s under partition vs 627 unpartitioned (**81%**) — the
  explicit contrast to the throttle's kill-switch (~1%). Productive isolation, not suppression.
- **Disjoint partitions confirmed:** victim mask `0x1000` (8 SMs), aggressor `0xfff` (96 SMs),
  A∩B=∅; both green ctxs `init=1` **under MPS**.

## Both unknowns resolved (measured, not asserted)

1. **Does MPS yield p99 ISOLATION?** Only *with* the partition. MPS **alone** (C3, no partition)
   gives concurrency but the aggressor still oversubscribes all SMs → victim 22× (1.148 ms). MPS
   **+ disjoint green-ctx** (C4) → victim recovers to its 8-SM floor (0.130 ms). The lever is the
   **combination**: MPS provides cross-process *concurrency*; green-ctx CU masks provide the
   *spatial confinement*. Neither alone works (non-MPS partition = 40×; MPS-no-partition = 22×).
2. **Does the built CP54 green-ctx COMPOSE with MPS?** **Yes, with NO new wiring** — the same
   `CIPHER_QOS_CLASS=partition` / `CIPHER_SM_COUNT` env produced engaged green ctxs (`init=1`,
   sm 8 / 96, disjoint masks) under the MPS server. No `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` needed.

**Harness auto-label correction:** the script first printed "OUTCOME 3" because its `confined`
predicate gated on the `on_our_green` CUPTI counter (=0 — a per-launch counting artifact;
`cuCtxSetCurrent` is sticky so launches are confined without incrementing it). The **measured**
gate metric — victim p99 at the partition floor, and the victim running at 8-SM speed (C2/C4 p50
0.10–0.11 ms vs C1 full-GPU 0.033 ms) — proves confinement. Heuristic fixed to judge confinement
empirically (`confined = disjoint and near_solo`); the corrected verdict is **OUTCOME 1 PASS**.

## What this means (and the precise boundary)

- **The 1-H100 cross-tenant p99-isolation ceiling is NOT a hardware wall — it is a DEPLOYMENT-MODE
  requirement: the GPU must run under CUDA MPS.** Bare Default mode time-slices (40×); MPS +
  CIPHER's existing green-ctx partition isolates (near floor). **R-I1's lever is found.**
- **Scope honestly — this is NOT per-tenant isolation for 100 agents.** There are only **15 × 8-SM
  groups**; ≥85 of 100 agents must share groups → contention returns *within* shared groups, and
  MPS active-thread-% can't rescue it (100 clients on 132 SMs ≈ 1.3 SMs each, below the 8-SM
  green-ctx floor). What this run proves: **a bounded number (≤ ~14) of designated latency-sensitive
  tenants can get DEDICATED groups with real p99 isolation; the bulk share.** That reconciles with
  D.8 §4 (at ~2% duty only ~2 agents are active at once) — Goal-1's "each agent thinks it owns the
  GPU" is satisfied by **hardware self-arbitration at realistic duty (D.8) PLUS dedicated-group
  protection for the latency-critical subset (this result)**, not by 100 disjoint partitions.
- **MPS is the only scalable concurrency lever** (MIG ≤ 7 instances; multi-GPU sidesteps): run the
  serving GPU under MPS so CIPHER's green-ctx groups can confine the protected subset.
- Both prior negatives are now explained: D.8 throttle-band (frequency-reducer, can't isolate the
  tail) and bare green-ctx (no cross-process concurrency without MPS) failed; **MPS + green-ctx is
  the composition that works.**

## Verdict + disposition (STOP for Anil)

**OUTCOME 1 — PASS. Cross-tenant p99 isolation on a single H100 is ACHIEVABLE via MPS + the
existing CP54 green-ctx, with the aggressor kept productive (81%).** This was the MPS *validation*
(measurement only); anchor UNCHANGED, no substrate built, KL moot (pure isolation, no output path).

**R-I1 is not yet CLOSED — the lever is validated; the close needs the deployment integration:**
1. **Design the deployment integration** (next substep, design-memo first). The load-bearing
   content is NOT just MPS-under-CDI plumbing — it is the **policy / admission-control problem under
   the 15-group packing constraint**: *who* is designated band ≥1, and *how* are the 15 groups
   allocated when more than ~14 tenants want protection? Plus MPS as a V.1 serving requirement and
   wiring SHIELD band → reserved green-ctx group (band 0 bulk → shared remainder), default-OFF.
2. **Then the R-I1 close gate** (Mem #16): additivity-KL byte-identical OFF + 30-min N=128 soak +
   this Gate-B PASS → rotate + tag.

**Honest residual (carried, not spun):** the 40× was a synthetic continuous FLOP-flood worst case;
D.8 §4 found the H100 self-arbitrates at Goal-1's realistic ~2-active-of-100-at-2%-duty. So whether
the gap *bites* under real V.1 load is still open — but we now know the lever (MPS+green-ctx) if it
does, and the V.1 soak should run under MPS to both test the gap and exercise the isolation.

Artifacts: `ri1_mps.log`, `ri1_gateb_result.json` (this MPS run overwrote the non-MPS one — the
non-MPS C4=1.959ms is in `R_I1_GATEB_VERDICT.md`), `mps_gateb.sh`, `d8_ri1_gateb.py`.
