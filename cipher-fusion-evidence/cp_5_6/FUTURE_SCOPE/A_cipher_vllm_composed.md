# FUTURE_SCOPE/A — CIPHER + vLLM Composed Architecture

**Recommended next-phase priority.** Unlocks the 3-arm benchmark (C) and is
the production architecture; CP 5.5 depends on it.

## Goal

Make CIPHER's substrate-layer **cross-process** decode fusion operate *above*
production-grade vLLM serving instances — composing CIPHER's distinctive
primitive (fusion across separate tenant processes + driver-level actuators)
with vLLM's distinctive strength (intra-process paged-attention continuous
batching). The composed stack is CIPHER's product; neither layer reimplements
the other.

## Architectural approach

The architectural distinction (per `INDUSTRY_METHODOLOGY_ALIGNMENT.md`): vLLM
fuses requests *within* one serving process; CIPHER fuses *across* separate
tenant processes, which vLLM structurally cannot. Composition:

- Each tenant remains a separate process. Within a process, requests are
  served by a vLLM instance (paged KV, intra-process continuous batching) —
  unchanged, production-grade.
- CIPHER's substrate sits **above**: it transparently intercepts decode work
  and routes it so that decode steps from *different tenant processes* sharing
  a model are co-scheduled into one batched execution — the cross-process
  fusion that no single vLLM instance can do.
- CIPHER's driver-level actuators (green-context partitioning, DVFS, the
  teacher-forced correctness gate) attach at the substrate layer, orthogonal
  to vLLM's request scheduler.
- vLLM's paged KV is preserved untouched — CIPHER does not manage KV pages; it
  manages cross-process *scheduling/fusion* and actuation.

Integration points to design: (1) the interception boundary — where CIPHER
hooks (the decode-step / engine-step boundary, transparent to tenant code,
`cipher_spec_decode` monkeypatch pattern); (2) the cross-process batch-formation
contract — how N vLLM instances' ready decode steps are gathered; (3) how
vLLM's paged KV stays per-instance while the weight-bound GEMMs fuse;
(4) actuator attachment (green ctx, DVFS, gate).

## Build scope + dependencies

- **Depends on:** the cross-tenant batching primitive (built, static-verified
  this CP); a vLLM integration/fork-point study.
- **Scope:** vLLM integration layer; the transparent cross-process interception
  shim; the cross-process scheduler; actuator wiring; correctness gating.
- **Open design question:** whether cross-process GEMM fusion is feasible
  without forking vLLM's engine, or whether a thin vLLM fork is needed.

## Time estimate

~2 weeks design + prototype.

## Success criteria

- 3-arm benchmark (`FUTURE_SCOPE/C`): Arm 3 (CIPHER over vLLM) shows a
  measurable substrate-attributable lift over Arm 2 (vLLM intra-process alone)
  on the N-separate-tenant-processes workload.
- Transparent to tenant code (tenants run unmodified).
- Per-tenant correctness gate (teacher-forced KL ≤ 0.1) preserved.
- vLLM's paged KV / intra-process batching unmodified and intact.
