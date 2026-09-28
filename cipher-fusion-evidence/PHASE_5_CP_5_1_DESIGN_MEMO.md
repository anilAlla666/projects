# Phase 5 — CP 5.1: vLLM / TGI live-decode integration — DESIGN MEMO

**Date:** 2026-05-17. **Status:** Phase 5 first CP. Not started — scoped for
adjudication. Supersedes `PHASE_5_CP_5_1_PLAN.md` (consolidated here into the
§1–§7 per-CP structure; the original is preserved as history).

---

## §1 — Scope / what CP 5.1 ships

CP 5.1 makes the CIPHER substrate **real against the live serving stacks
operators run**, replacing the synthetic single-stream drivers used through
Phase 4. It ships:

- **vLLM compatibility** — substrate transparent and correct against vLLM
  **0.20.2** (the version in operator fleets), injected via
  `CUDA_INJECTION64_PATH`, exercised under continuous batching and
  paged-attention.
- **TGI 2.x compatibility** — same, for Hugging Face TGI.
- **Operator deployment recipe** — a reproducible Docker image plus the
  CP 2.5 `cipher-platform.deb`, installable on operator reference stacks.
- **`cipher_flopd` deployment** — the live FLOP-telemetry daemon (a CP 4.8
  §5.4 carried-forward gap: not deployed on the soak pod, so the PMU FLOP
  cross-check fell back to PyTorch `FlopCounterMode`). A real operator
  deployment wants live FLOP telemetry; folding `cipher_flopd` into the
  deployment recipe closes the gap where it belongs.
- **Live-workload measurement** — MFU and tok/W measured on real vLLM/TGI
  decode (continuous batching), not synthetic drivers.

## §2 — Why this CP / dependencies

**Dependency in.** CP 4.8's descriptive-MFU run surfaced the gap directly:
WL04 / WL10 / WL16 (vLLM serving / speculative decode / prefix caching)
**failed at vLLM engine init** on an `xgrammar`/`tvm_ffi` environment break —
substrate interception was never reached (`CP_4_8_REPORT.md` §5.2). That is
integration-stack scope, not a substrate defect, and it means CIPHER's
production story is currently **unverified against the stacks operators run**.
CP 5.1 also composes with the Phase 4.6 finding that the KV-dedup primitive is
validated (100-proc) but not yet wired into a live decode engine.

**Dependency out.** CP 5.1 is the stated **prerequisite for the Phase 6
operator pilot** (see `PHASE_5_PLAN.md` §4) and feeds the CP 5.5 integration
measurement.

## §3 — Approach

Repair and pin the pod's vLLM / `xgrammar` / `tvm_ffi` environment, then
verify substrate transparency by differential testing: each target workload
run substrate-off vs substrate-on, asserting byte-identical or
numerically-equivalent output (the established CIPHER correctness bar — cf.
Mistral byte-identical, CP 4.6.1). Build the deployment recipe against
**CoreWeave / Lambda / Nebius reference stacks** so the install is validated
on operator-representative infrastructure, not just this pod.

## §4 — Gate criteria

CP 5.1 PASSES iff:

1. The CP 4.8 env-fail workloads (WL04 / WL10 / WL16) run clean to a
   measurement window — the `xgrammar`/`tvm_ffi` break is closed.
2. Substrate-on vs substrate-off output is correct (byte-identical or
   numerically-equivalent) on vLLM 0.20.2 and TGI 2.x.
3. The operator deployment recipe installs and runs reproducibly on at least
   one operator reference stack.
4. MFU / tok-W are measured on live continuous-batching decode.

Honest bar: CP 5.1 is a **correctness + integration** CP — the pass is "the
composed stack runs transparent and correct under operator deployment
patterns," not a new performance headline.

## §5 — Calendar

**2–3 weeks.** No load-bearing unknown — the environment break is diagnosed
(`CP_4_8_REPORT.md` §5.2) and the substrate is unmodified. Spread is staffing
and reference-stack access, not technical risk.

## §6 — Risks / known-unknowns

- vLLM v1 eagerly imports the `xgrammar` structured-output backend at engine
  startup; if `tvm_ffi` cannot be cleanly pinned, a vLLM-version pin or a
  backend-disable path may be needed — to be resolved early.
- Operator reference stacks may differ from this pod's driver/CUDA
  (580.105.08 / cu13); the recipe must not hard-depend on pod-specific
  versions.
- `cipher_flopd` has never run on this pod — its deployment is itself
  unvalidated and should be smoke-tested as a sub-task, not assumed.

## §7 — Anchors at CP start

kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`.
CP 5.1 ships **no substrate change** — the anchors are held; only the
integration/deployment layer is built.
