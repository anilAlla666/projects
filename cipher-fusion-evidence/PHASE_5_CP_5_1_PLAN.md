# Phase 5 — CP 5.1: vLLM / TGI live-decode integration — PLAN

**Date:** 2026-05-17. Planning memo (written during the CP 4.8 soak).
**Status:** Phase 5 first CP. Not started — scoped here for sequencing.

## Why this is CP 5.1 (first in Phase 5)

CP 4.8's descriptive MFU run surfaced it directly: **WL04, WL10, WL16
(vLLM serving / speculative decode / prefix caching) failed to run** — a
`xgrammar`/`tvm_ffi` environment-dependency break in the test harness. vLLM
never imported; **substrate interception was never reached.** This is
integration-stack scope, not a substrate defect — but it means CIPHER's
production story is currently unverified against the serving stack operators
actually run (vLLM, TGI). Phase 5 cannot credibly target an operator pilot
(Phase 6) without closing that gap first. Hence CP 5.1 leads Phase 5.

It also composes with the standing Phase-4.6 finding: the KV-dedup substrate
primitive is validated (100-proc) but **not wired into a live decode engine's
KV cache**. "Live-decode integration" is the umbrella — CP 5.1 makes the
substrate real against vLLM/TGI.

## Scope

- **vLLM compatibility** — verified against vLLM **0.5 / 0.6 / 0.7** (the
  versions in operator fleets), substrate active via `CUDA_INJECTION64_PATH`.
- **TGI 2.x compatibility** — same, for Hugging Face TGI.
- **Operator deployment recipe** — Docker image + `cipher-platform.deb`
  (the CP 2.5 deliverable), reproducible install.
- **Measurement** — MFU + tok/W on **real vLLM workloads** (continuous
  batching, paged-attention) — not the synthetic single-stream drivers; this
  is where the substrate's value composition is actually exercised.
- **Pass criterion** — the composed stack runs, transparent and correct,
  under operator deployment patterns: **CoreWeave / Lambda / Nebius reference
  stacks**.

## Calendar

**2–3 weeks.** Prerequisite for the Phase 6 operator pilot.

## Notes carried in

- The CP 4.8 vLLM failure is **not** to be fixed inside CP 4.8 (wrong scope) —
  it is the seed of this CP.
- CP 5.1 is where the §3 PMU cross-check tooling gap (`cipher_flopd` not
  deployed, CP 4.8 finding) should also be closed — a real operator
  deployment wants the live FLOP-telemetry daemon. Fold the `cipher_flopd`
  build into CP 5.1's operator-deployment recipe.
- Anchors at CP 5.1 start: kmod `e2f50452`, libcipher_rt `c2c5d313`,
  libcipher_v2 `86618c30`.
