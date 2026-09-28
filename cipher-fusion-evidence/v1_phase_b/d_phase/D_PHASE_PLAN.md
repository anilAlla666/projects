# D-PHASE PLAN — substrate gap-closure before the V.1 soak

**Date:** 2026-05-29. **Status:** PLAN — STOP for Anil approval (this plan + the D.7 memo)
before any code. **Anchors UNCHANGED until a build is approved** (cipher_rt_phase4
`8b5e928`/`9fe23143`, cipher_kmod `02fc2d1`/0.6.6, cipher_kv_bridge `5a3db034`;
cipher-fusion-evidence HEAD `ca4134f`).

## Why this phase exists

V.1 item-1 readiness (`V1_CP55_READINESS_FINDINGS.md`, `ca4134f`) found the V.1 soak
blocked on **net-new substrate** for the LOCKED Memory-#1 gates. Per Anil: deliver all 5
gates, **no scope-down, no defer-by-adjudication**. The gaps are closed as a **D-series
build phase BEFORE V.1**; V.1 stays measurement-only (Memory #16 — these builds are NOT
folded into V.1). Each D-substep: **design-memo → STOP for approval → build → close-gate →
STOP.** One substep at a time; every new op defaults OFF; worst case cuBLAS passthrough.

## The three substeps (item-1b gaps × LOCKED gates)

| Substep | Gap (audit ID) | LOCKED gate it serves | Current state | Net-new | ED |
|---|---|---|---|---|---|
| **D.7** | R-H1 heterogeneous weight residence (G4) | **Goal 1** — ≥3–5 distinct INT4 model kits co-resident, per-tenant routing, KL=0 | Marlin cache keyed by **`w_ptr`** (process-global); recipes/KV/Koopman model_uuid-keyed (G12/G3); `CIPHER_REGISTER_MODEL` (G10); WA arenas=100 (G2) | Marlin kit residence + per-(model_uuid,layer,K,N) routing at ≥3–5 models; JIT cubin cache scaling; heterogeneous-residence correctness gate | ~5 |
| **D.8** | R-D5 FAIRNESS (G11) + R-I1 SHIELD | **Goal 1 isolation** — 100 bursty agents coexist, no starvation, per-tenant p99 bounded | FAIRNESS `fairness_quota_remaining_pct` **stub** (defaulted 100, no enforcement); SHIELD **unwired**; SM-partition 15-group exists | per-tenant quota ledger + hot-path enforcement (G11 ~4 ED); SHIELD noisy-neighbor p99 wiring (~3–5 ED) | ~7–9 |
| **D.9** | W.8 persistent-kernel (audit M.2) | **Goal 3** — ≥85% weighted-mean MFU | DE-PRIORITIZED by FWD-1 (graphs broke + didn't pay on this stack; static-KV forward win already in vLLM) | resident persistent kernel + ring-buffer dispatch; **REVERSES the FWD-1 decision — memo must justify or surface 85% as a v1.x/Blackwell ceiling** | ~10–15 |

**Dependency order:** **D.7 FIRST** (without heterogeneous residence there is no
100-agent-heterogeneous workload to soak — it unblocks V.1). **D.8 and D.9 are
parallel-OK after D.7** (isolation vs MFU; independent). Goals 2/4/5 need no D-substep
(Goal 2 GPU-aggregate tok/W on VOLT — built/auto-fires; Goal 4 Koopman caller wired;
Goal 5 classify_launch + cipher-platform — the readiness pre-flight item 1c tests their
auto-activation once the CDI stack is restored).

## D.9 carries an explicit reversal (flagged up front)

FWD-1 de-prioritized W.8 on evidence: `torch.compile`/cudagraphs were wrong+slower on
this stack, manual capture hard-aborted, and the real forward win (static KV) is already
in the vLLM product; the residual over the bandwidth floor was ~1.8× and brittle. **D.9
revives W.8 to chase ≥85% MFU — the opposite call.** The D.9 memo MUST justify the
reversal: *what changed* (the V.1 gate now demands 85%, which FWD-1 wasn't evaluating),
*why 85% is reachable on a single H100 at this workload's batch sizes* when FWD-1 found
the decode forward launch/bandwidth-bound (note: 85% MFU is a *prefill/compute-bound*
target; the bursty agentic mix is *decode-heavy/memory-bound*, where 85% may be
physics-unreachable), and *the honest risk it still PARTIALs by physics*. **If the memo
cannot defensibly argue 85%-reachable, it says so — surfacing 85% as a genuine
v1.x-or-Blackwell ceiling, not a faked gate.** (Per Anil's instruction.)

## Discipline (every D-substep)

- **Substrate-line only (Mem #24):** CUDA-dispatch / symbol-intercept; no vLLM/NCCL source
  patch (HARD STOP if required).
- **Default-OFF:** every new op ships gated off; worst case = cuBLAS passthrough == vanilla.
- **CLOSE REQUIREMENT — FULL no-regression regression (Mem #16, HARD GATE; addendum
  2026-05-29):** each D-substep (D.7/D.8/D.9) closes **ONLY** after re-running the **FULL
  prior-substrate regression — W7–W11 microbenches + the 30-min N=128 resolver-coherence
  soak + every prior model's KL — showing ZERO degradation vs the pre-D anchor
  (`8b5e928`/`9fe23143`).** Testing the new op alone is **insufficient**.
- **Correctness (Mem #11, HARD STOP):** the new tenant KL=0 vs its family standalone **AND
  every prior model's KL unchanged**. Any regression → **HARD STOP, revert via tag.**
- **Additive-only preferred:** if modifying W7–W12 substrate is unavoidable, treat it as a
  fresh substrate step with the full regression above — **never a quick-fix.**
- **Fresh substrate (Mem #16):** each substep = the full regression above + 9-cell gate +
  commit tag; anchors rotate only on an approved, fully-regressed build.

## Sequence

1. **(this doc)** D-phase plan → STOP for approval.
2. **D.7 memo** (R-H1) → STOP for approval → build → close-gate (`D7_*_CLOSE.md`) → STOP.
3. **D.8 memo** → approve → build → close → STOP.
4. **D.9 memo** (with the reversal justification) → approve → build/or-surface-ceiling → STOP.
5. Then **V.1 soak** (items 2–6) on the gap-closed substrate + restored CDI stack →
   §4 5-goal verdict.

No code until the plan + D.7 memo are approved.
