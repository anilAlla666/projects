# CP 5.6 — Close-Out Report

**Date:** 2026-05-18. Substrate anchors at close: libcipher_rt **`a7ac8e97`**,
kmod `e2f50452`, libcipher_v2 `86618c30`, cipher_kv_bridge `8d6ffe3f`.

## Executive summary

CP 5.6 began as the F1 root-cause + re-measurement CP and became the campaign's
honesty correction. It closes with:

- **Closed / fixed:** finding F1 (Marlin full-GPU cross-context race) — root-caused
  and fixed in `a7ac8e97`.
- **Retracted:** the composed **≈3.6× tok/W** headline, in all three places it
  was recorded (CP 2.4 3.617×, CP 2.5 gate-b 3.602×, CP 4.8 Task B 3.6166×) —
  an F1-degenerate-loop artifact measured by a timing-only gate.
- **Verified:** F1-fixed, correctness-gated replacements — single-tenant
  Mistral-7B 1.54× tok/W; **static cross-tenant batching 3.69× (N=8 TinyLlama)
  / 3.26× (N=4 Mistral) substrate-attributable**.
- **Bounded:** dynamic continuous-batching throughput is paged-attention
  serving-engine territory — out of CIPHER's substrate scope.
- **Architectural decision:** Option 1 + Option 3 — close on the verified
  static result; compose with vLLM for the dynamic regime.
- **Deferred:** the original Phase 5 sequence (CP 5.3 STEP 2B, CP 5.4, CP 5.5)
  and six scoped future-work items (`FUTURE_SCOPE/A–F`).

## 1. F1 root cause + fix — Priority 1

**Root cause:** `cipher_rt_marlin_engine_dispatch` wrapped the Marlin full-GPU
GEMM in `PrimaryCtxGuard` → the GEMM and its output write ran on the **primary**
context while the caller/consumer ran on a **green** context — two contexts,
two NULL streams, no cross-context ordering → the consumer raced the producer →
degenerate decode (all-zero logits → argmax(0)). Bisected by isolation; Variant F
(a real second context) proved it. **Fix (P1):** the full-GPU branch records a
completion event inside the guard, then `cuStreamWaitEvent` on the caller's
stream after the guard restores context — a non-blocking GPU-side dependency.
Verified: full-GPU path 99/100 degenerate → 0/100; green path unchanged.
Anchor rotated `dc804eb3 → a7ac8e97`; `dc804eb3` preserved as `.pre_cp56`.
Evidence: `cp_5_6/f1_p1_verify.{cpp,runlog}`, `cp56_p1_build.log`,
`f1_audit/FINDINGS.md`.

## 2. TPW re-measurement + retraction — Priority 2

A teacher-forced top-1 / logit-KL correctness gate (cascade-free — FP-tie- and
cascade-immune, unlike a free-running gate) was built and the CP 2.4 composed
gate re-measured on `a7ac8e97`. **Verdict C:** the 3.617× does not survive.
On F1-fixed decode the all-on arm's `accept_rate` drops from a degenerate
1.000 to 0.19–0.77 — the loop is gone, and with it the 1.795× tok/s "speedup"
(it was spec-decode hitting 100% acceptance on a looping sequence). The honest
single-tenant Mistral-7B B=1 composed lift is **1.54× tok/W** (0.67× tok/s ×
2.3× DVFS power-cut — the DVFS half is real; the throughput half was the
artifact). Marlin INT4 fails the 99% teacher-forced gate at ~90–93% — honest
quantization drift, not a bug (prompt-2/code passes at 100%). Detail:
`CP_5_6_P2_REPORT.md`.

## 3. Phase 4 audit synthesis — Priority 3

`PHASE_4_AUDIT_P3.md` classified every fusion-campaign CP for F1 exposure.
**F1's blast radius is exactly one measurement** — the composed ≈3.6× tok/W —
recorded three times, all retracted. Every other CP claim stands: pre-Fix-A CPs
have no `PrimaryCtxGuard`; CPs 4.4/4.7 closed analytically at memo; CPs
4.6/5.1/5.2 are fp16/KV-cache paths that never engage Marlin INT4; CP 5.3
STEP 2 is the CP that *caught* F1. No cascade.

## 4. Phase B engineering work

A re-measurement that retracts a headline needs a real replacement. Phase B
built one.

- **Diagnostic 1** — the weight-sharing premise was falsified: B=1 decode is
  *not* HBM-bandwidth-bound (measured HBM utilisation 4–6%); it is
  launch/overhead-bound. Shared weight pages give *capacity*, not bandwidth.
  The real lever is **cross-tenant batching** (amortise launch overhead across
  N tenants). `PHASE_B_DIAG1_WEIGHT_SHARING.md`.
- **Cross-tenant batching primitive** — a substrate-orchestrated batched-decode
  executor (Form A): one process holds the model, tenant clients submit decode
  requests, the substrate fuses them into one B=N kernel sequence — transparent
  to tenant code. In-process batch-scan ceiling: B=8 → 7.70× tok/W vs B=1.
- **Session 1 (N=2 prototype)** — 0.872 tok/W = 1.79× baseline, **94.7% of the
  in-process B=2 ceiling** (cross-process plumbing overhead ~5%), KL 6.6e-5.
- **Session 2 Steps 1–3 (N=4/8/16 static scaling)** — substrate-attributable
  (vs naive N-concurrent): N=4 1.76×, **N=8 3.69×**, N=16 8.72×; 88–93% of the
  in-process ceiling; teacher-forced KL 5.5e-5. A mid-session gate-methodology
  bug (free-running KL cascading on an FP-tie) was caught at N=16 and fixed to
  teacher-forced — transparently logged. `PHASE_B_SESSION_2_REPORT.md`.
- **Mistral-7B verification** — N=4 batched 0.861 tok/W vs naive 4-concurrent
  0.264 → **3.26× substrate-attributable**, KL 3.2e-5. The lever generalises to
  the 7B model, and *better* — naive concurrency degrades worse at 7B, so the
  substrate's value grows with model size. `PHASE_B_MISTRAL_VERIFY.md`.
- **Step 4 (heterogeneous batch)** — eviction-only continuous batching has a
  **57.7% drain cost** (vs the <15% target): the round is gated by the longest
  tenant while the batch drains. Architectural-ceiling finding; the fix is
  admission. `PHASE_B_STEP_4_FINDING.md`.
- **Sub-component 1 (admission control)** — FIFO queue + admit-on-free;
  re-prefill-on-rebatch. Verified correct (12/12 complete, batch held full,
  KL 5.5e-5).
- **Sub-component 2 (KV-splice)** — survivors' KV kept, only the admitted row
  prefilled; left-padding invariant. Verified **correct** (KL 5.5e-5) — but
  throughput unchanged (~52 tok/s, ~25% of static): the bottleneck is the
  per-step cost of the manual continuous-batch loop, not the rebatch strategy.
  `PHASE_B_SUBCOMPONENT_2_FINDING.md`.

## 5. Methodology — industry alignment

`INDUSTRY_METHODOLOGY_ALIGNMENT.md` audited CIPHER's methodology against
industry serving/power-benchmark practice. Key correction: the comparison to
vLLM is **architectural, not a tok/W metric race** — vLLM fuses *within* one
process; CIPHER fuses *across* separate tenant processes, which vLLM
structurally cannot. The 3-arm cross-process benchmark (naive / vLLM
intra-process / CIPHER substrate) is the hard gate before external claims.
Citation provenance for the named external standards (TokenPowerBench AAAI
2026, MLPerf Power 2025, ML.Energy, 1/W Law) is pending verification.

## 6. Architectural decision — Option 1 + Option 3

- **Option 1** — CP 5.6 closes on the verified *static* cross-tenant batching
  result. Dynamic admission throughput is correctly serving-engine territory,
  not CIPHER scope.
- **Option 3** — compose with vLLM: vLLM provides production-grade
  intra-process paged batching; CIPHER's substrate provides cross-process
  fusion and driver-level actuators vLLM structurally cannot. The composed
  architecture is CIPHER's product. (Option 2 — rebuilding a paged-attention
  engine — was rejected: it reinvents vLLM, adds no magnitude, and weakens the
  diligence story.)

## 7. Verified claims

- F1 cross-context race **fixed** in libcipher_rt `a7ac8e97`.
- Cross-tenant decode-step batching primitive at the substrate layer,
  **transparent to tenant code**, fusing across **separate tenant processes**.
- At **static batch sizes, Flash-attention path**:
  - **N=8 TinyLlama-1.1B: 3.69× substrate-attributable tok/W** vs naive
    8-concurrent.
  - **N=4 Mistral-7B-v0.1: 3.26× substrate-attributable tok/W** vs naive
    4-concurrent.
  - Both **correctness-gated** — teacher-forced logit-KL ≤ 0.1 per tenant.
- Cross-process plumbing efficiency: **88–95% of the in-process ceiling**.
- Methodology industry-aligned (citation provenance pending).

## 8. Verified boundaries

- **Dynamic admission throughput** — the mechanism (admission, eviction,
  KV-splice) is verified *correct*; throughput is limited by the manual
  continuous-batch loop — paged-attention territory. The CIPHER + vLLM composed
  architecture is the next-phase path (`FUTURE_SCOPE/A`).
- **Multi-workload** — only WL01 (TinyLlama) and Mistral-7B verified at static
  N. WL02–WL24 pending (`FUTURE_SCOPE/D`).
- **Actuator composition** — cross-tenant batching has not been measured
  composed with Marlin INT4 / DVFS / KV optimisation (`FUTURE_SCOPE/E`).
- **Architectural alternative** — CIPHER (cross-process fusion) and vLLM
  (intra-process fusion) are structurally distinct and composable, not
  competitors.

## 9. CP 5.6 status + Phase 5 impact

**CP 5.6 is CLOSED** on this evidence. Deferred during the F1 detour and still
pending: **CP 5.3 STEP 2B** (INT4 acceptance collapse on Mistral), **CP 5.4**
(per-tenant arbitration), **CP 5.5** (100-tenant integration soak). CP 5.5 now
depends on the CIPHER + vLLM composed architecture (`FUTURE_SCOPE/A`) and
Track 2 weight-sharing (`FUTURE_SCOPE/B`). Future work scoped in
`FUTURE_SCOPE/A–F`; recommended next-phase priority is **A**.

## 10. Evidence index

`cp_5_6/` — `f1_p1_verify.*`, `cp56_p1_build.log`, `f1_audit/FINDINGS.md`
(F1 fix); `CP_5_6_P2_REPORT.md` + `p2/` (re-measurement); `PHASE_4_AUDIT_P3.md`
(audit); `PHASE_B_DIAG1_WEIGHT_SHARING.md`, `PHASE_B_*_DESIGN.md`,
`PHASE_B_SESSION_{1,2}_REPORT.md`, `PHASE_B_MISTRAL_VERIFY.md`,
`PHASE_B_STEP_4_FINDING.md`, `PHASE_B_SUBCOMPONENT_2_FINDING.md`,
`phase_b/` (Phase B); `INDUSTRY_METHODOLOGY_ALIGNMENT.md` (methodology);
`FUTURE_SCOPE/A–F` (next phase).
