# Actuator-Goal map (in-container) — is the cublasLt interception fix V.1-blocking?

**Date:** 2026-05-29. Read-only diagnostic (no fix code, anchor unchanged `9fe23143`).
In-container, D.10 staging `01d4effb` bind-mounted, B=8 Mistral-7B fp16. Verdict for Anil:
**the cublasLt interception gap is NOT a blanket V.1 blocker — it gates only the
matmul-routed actuators (Goals 4, and 3); Goals 1/2/5 rest on SEPARATE paths that engage
in-container.**

## Empirical engagement (one CLASSIFY line, in-container)

`workload=A4_BATCH_INFERENCE confidence=720` · `volt_engage=1` · `marlin_engage=0`
`marlin_bf16_obs=0` · `koop_bf16_obs=0` · `mach_intercepts=0` · `attn_p1..p6=0` · `bw_bound=1`
Plus banners: VOLT "ARMED (auto/no env)", NVML rc=4 NOT_SUPPORTED → **kmod ioctl path
(CIPHER_SET_CLOCK_MHZ) available** (VOLT works in-container via the kmod fallback);
Koopman registered at the matmul-dispatch (but observes 0); attn substrate active (0 obs here).

## Per-goal map

| Goal | Actuator(s) it needs | Path | Engages in-container? | Blocked by cublasLt gap? |
|---|---|---|---|---|
| **1** density/heterogeneity | POOL/coresidence + weight-residence (D.7) + attn (long-ctx) | kmod scheduler / VMM arenas / SDPA — **all SEPARATE from cuBLAS matmul** | scheduler/residence YES (kmod); attn 0-obs on this decode (separate path) | **NO** — separate paths (D.7 has its own status) |
| **2** tok/W ≥2× | cross-tenant batching (the 3.06×-N=4 lever) + VOLT | POOL/density (separate) + **VOLT kmod clock (separate)** | **VOLT engage=1 ✓**; batching = POOL path (separate) | **NO** — primary levers separate; Marlin-INT4 is a *bonus*, not the 2× driver |
| **3** MFU ≥85% | Marlin (compute-reduction) **or** persistent-kernel (D.9) | Marlin = matmul-routed (BLIND); D.9 not built | **NO** — Marlin blind; D.9 absent | **YES** (Marlin blind) — and/or needs D.9 |
| **4** Koopman O(1) | Koopman | matmul-routed (registered, `koop_obs=0`) | **NO — BLIND** | **YES** (cublasLt gap) |
| **5** auto-profile | classifier + cipher-platform | CUPTI/classify (separate) | **YES** (`A4_BATCH_INFERENCE`, conf 720) | **NO** |

## Verdict (scopes the next substep — STOP for Anil)

- **NOT cublasLt-blocked (separate paths engage in-container):** Goal 5 (classifier ✓),
  Goal 2's clock lever (VOLT engage=1 ✓), Goal 1's density/residence (kmod scheduler/arenas —
  D.7's own track). The cublasLt interception is **not a prerequisite** for these.
- **cublasLt-blocked (matmul-routed actuators blind):** **Goal 4 (Koopman)** outright, and
  **Goal 3 (≥85% MFU)** if it relies on Marlin (it also independently needs D.9 persistent-
  kernel, which is unbuilt/de-prioritized — so Goal 3 ≥85% is doubly gated regardless of the
  cublasLt fix).
- **Therefore the cublasLt interception fix (the next-level "intercept torch's cublasLt before
  it resolves" work) is a Goal-3/4 prerequisite, NOT a V.1-wide blocker.**

**Honest magnitude caveats (not the engage question — the gate-value question):**
- Goal 2 ≥2×: VOLT *engages* but VOLT-alone on Mistral is weak (memory `cipher-t43`: Mistral
  B=1 VOLT was −14%); the ≥2× rests on the **cross-tenant batching density** lever (3.06× N=4),
  which is the POOL/multi-tenant path and must be *measured multi-tenant* (not this single-
  tenant B=8 run) to confirm. Separate from cublasLt either way.
- Goal 3 ≥85%: the hardest — blocked by Marlin-blind AND needs D.9 (de-prioritized by FWD-1
  on physics: 85% is prefill/compute-bound, the agentic mix is decode/memory-bound).

## Recommendation

Pursue V.1 on the **non-cublasLt-blocked goals first** (1 density, 2 via batching+VOLT, 5
profile) — these are measurable in-container now without the interception fix. Treat the
**cublasLt interception fix as the Goal-4 (Koopman) prerequisite** (and a Goal-3 contributor),
sequenced only if/when Goal 4 is on the v1 critical path. This avoids sinking more depth into
intercepting an actuator (Marlin/Koopman) that Goals 1/2/5 do not depend on. Anchor unchanged.
