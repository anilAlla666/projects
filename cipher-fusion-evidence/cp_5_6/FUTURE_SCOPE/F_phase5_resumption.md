# FUTURE_SCOPE/F — Original Phase 5 Sequence Resumption

## Goal

Resume the original Phase 5 sequence, deferred during the CP 5.6 F1 detour.
Three CPs remain: CP 5.3 STEP 2B, CP 5.4, CP 5.5.

## The three pending CPs

### CP 5.3 STEP 2B — two-model speculative-decode acceptance collapse
- **Correction (2026-05-18):** an earlier draft of this line said "on
  Mistral-7B INT4" — that is wrong. The collapse is the **Llama model-draft
  arm** (target Llama-3.1-8B, draft Llama-3.2-1B-Instruct); Mistral's CP 2.4
  spec arm used an n-gram draft and ran clean. The measured axis is
  substrate-OFF (0.490) → substrate-ON (0.036), not FP16→INT4. See
  `cp_5_3/CP_5_3_STEP_2B_SCOPE.md` §0 and `cp_2_4/CUDNN_ATTN_MARLIN_HANG.md`.
- **Goal:** resolve Axis B of CP 5.3 STEP 2 — the model-draft speculative-decode
  acceptance collapse (observed 0.490 → 0.036 with the substrate on).
- **Approach:** design-memo-first — scope memo `CP_5_3_STEP_2B_SCOPE.md` written
  2026-05-18, awaiting adjudication. Leading post-CP-5.6 hypothesis: the
  collapse is F1 (measured before F1 was fixed in `a7ac8e97`). Step 1 is a
  three-arm discriminator; decide whether spec decode is viable on the INT4
  substrate path or should be scoped out for that regime.
- **Dependencies:** none new — CP 5.3 STEP 2 substrate (`dc804eb3`/`a7ac8e97`).
- **Time:** ~3–5 days.
- **Success criteria:** Axis B root-caused; CP 5.3 closes or the INT4-spec
  path is explicitly scoped out with evidence.

### CP 5.4 — per-tenant arbitration
- **Goal:** per-tenant resource arbitration (the fair-share / QoS layer across
  concurrent tenants).
- **Approach:** not started — needs a design memo.
- **Dependencies:** composes with the cross-tenant batching scheduler; best
  designed alongside `FUTURE_SCOPE/A` (the composed architecture's scheduler
  is the natural home for arbitration).
- **Time:** ~1–2 weeks.
- **Success criteria:** per-tenant QoS (latency/throughput share) enforced and
  measured under multi-tenant load.

### CP 5.5 — 100-tenant integration soak
- **Goal:** a 100-tenant integration soak — the scale-validation CP.
- **Approach:** not started.
- **Dependencies — hard:** requires `FUTURE_SCOPE/A` (CIPHER + vLLM composed
  architecture — 100 tenants need production-grade serving) **and**
  `FUTURE_SCOPE/B` (weight-sharing — 100 same-model tenants cannot each hold a
  weight copy). CP 5.5 cannot start until A and B land.
- **Time:** ~1–2 weeks after dependencies land.
- **Success criteria:** 100 concurrent tenants, stable soak, aggregate
  metrics + per-tenant correctness gate, no degradation over the soak window.

## Sequencing

CP 5.3 STEP 2B is independent and can run any time. CP 5.4 is best co-designed
with `FUTURE_SCOPE/A`. CP 5.5 is gated on A + B. Recommended order:
A (composed architecture) → B (weight-sharing) + C (3-arm benchmark) →
CP 5.4 (arbitration, alongside A) → CP 5.5 (soak). CP 5.3 STEP 2B can be
slotted independently.

## Time estimate

CP 5.3 STEP 2B ~3–5 days; CP 5.4 ~1–2 weeks; CP 5.5 ~1–2 weeks after A+B.
