# FUTURE_SCOPE/D — Multi-Workload Extension (WL02–WL24)

## Goal

Extend the verified cross-tenant batching result beyond WL01 (TinyLlama-1.1B)
and Mistral-7B to the rest of the 24-workload matrix at static N, with
per-workload regime classification — so the substrate-attributable lift is
characterised across the workload space, not asserted from two points.

## Architectural approach

Reuse the Phase A pillar harness (`pillar_driver.py`, `run_multitenant.sh`,
`analyze_*`) and the Session-2 batched executor. For each workload, run the
3-arm-style static measurement (vanilla / naive N-concurrent / CIPHER batched)
with the teacher-forced correctness gate. Crucially — most of WL02–WL24 are
**not** B=1 causal-LM decode (the regime classification from `PHASE_A_PLAN.md`):
~10 are prefill/training/encoder/diffusion workloads where the teacher-forced
decode gate does not apply, 3 are vLLM, 1 is deferred multi-GPU. The extension
must:

- run the cross-tenant batching measurement only on the **decode-regime**
  workloads where it is defined (WL01/02/11/15/21/23/24 and decode-bearing
  20/22);
- for non-decode workloads, classify the regime and document why cross-tenant
  decode-step batching does not apply (a regime-appropriate primitive, if any,
  is separate scope);
- report per-workload: regime, substrate-attributable lift (where defined),
  actuator engagement, correctness.

## Build scope + dependencies

- **Depends on:** Track 2 weight-sharing (`B`) is desirable for the larger
  models at higher N (memory); framework installs (`setup.sh` — vLLM,
  diffusers, whisper, etc.) for the non-base workloads.
- **Scope:** per-workload harness adaptation (model, batch, regime); GPU runs;
  the per-workload regime table.

## Time estimate

3–5 days of GPU + harness-adaptation time.

## Success criteria

- Every decode-regime workload measured: substrate-attributable tok/W at
  static N, correctness-gated.
- Every non-decode workload classified with a documented reason the
  decode-batching primitive does not apply.
- A complete per-workload regime + lift table — the honest matrix the Phase A
  brief originally sought, scoped to where the methodology is sound.
