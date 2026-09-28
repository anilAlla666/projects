# CP 5.4 — PER-TENANT ARBITRATION — CLOSEOUT (PARTIAL)

**Date:** 2026-05-19. **Status: PARTIAL CLOSE** — substrate-arbitration
primitive validated through Step 1.6B-3; Steps 1.6B-4 / 1.6P / 1.7 deliberately
deferred. Documentation only — no source modified, no anchor rotation, no GPU.

---

## 1 — HEADLINE

CP 5.4 (per-tenant arbitration) closes substrate-arbitration primitive
validation through Step 1.6B-3. Remaining sub-steps (1.6B-4 CIPHER comparison,
1.6P POOL-scaling, 1.7 failure modes) are **deferred** — the synthetic stress
conditions they measure (B=1 TinyLlama continuous decode at N=7–10) do not
reflect real neocloud customer workloads (agent loops, mixed inference,
batched serving), and so do not inform customer-facing claims. The
substrate-value measurements that **do** inform the product story — TPW lift,
MFU, memory density, aggregate throughput, dynamic SM migration, cross-tenant
weight sharing — are established in Phase B, Phase 4, Track 2, Track 3, and
today's TPW re-test, and culminate in **CP 5.5's 100-tenant production-scale
measurement**. CP 5.4 leaves the SM-arbitration substrate **built, correct, and
ready for CP 5.5 to build on**.

## 2 — MEASUREMENT PROVENANCE

**Claims SUPPORTED by closed CP 5.4 sub-steps:**

| claim | substantiating step | evidence |
|---|---|---|
| kmod 15-group SM ledger, two-clause disjointness invariant | Steps 1.1–1.3 | `CP_5_4_STEP_1_1_BUILD_LOG.md`, `..._1_3_DESIGN_MEMO.md` |
| QoS classes — PARTITION (isolated green-ctx) vs POOL (batched) | Steps 1.3a–1.3b′ | `step1_3/`, `step1_3b/` |
| lift curve flat across 120→40 SM (K≤7 "failure" was a fixed cross-stream race) | Step 1.4 | `step1_4/`, [[cipher-cp54-step1-4]] |
| green-ctx churn ~1.7 ms/resize, K-independent; Marlin cubin one-shot | Step 1.5 | `step1_5/CP_5_4_STEP_1_5_REPORT.md` |
| mixed deployment — 2×16-SM PARTITION + POOL co-resident, disjoint under load | Steps 1.6B-1/1.6B-2/1.6B-2A | `step1_6/CP_5_4_STEP_1_6B_2_CHECKPOINT.md` |
| substrate composes correctly through Track 2 + Track 3 anchor rotations | §0 1.6B-2A re-validation gate (this session) | `step1_6/CP_5_4_STEP_1_6B_2A_REVALIDATION.md` |
| naive multi-tenant baseline characterised, OP-2/5/asym, 5 reps | Step 1.6B-3 (today) | `step1_6/CP_5_4_STEP_1_6B_3_RESULTS.md` |

**Partial data from the stopped Step 1.6B-4** (recorded for honesty, not a
closed claim — `step1_6/CP_5_4_STEP_1_6B_4_STOPPED.md`):

| observation | data | status |
|---|---|---|
| OP-2 (low contention) Arm-B CIPHER CV | 0.256 [95% CI 0.252–0.260] vs naive 0.160 | complete (5 reps) — CIPHER **does not** isolate at low contention |
| OP-5 (high contention) Arm-B CIPHER CV | 0.172 vs naive 0.183 (3 of 5 reps) | partial — CIPHER **modestly tighter**, ~6 %; NOT CI-confirmed |
| D4 — Marlin × partition | `marlin_engine_initialised=False` | resolved — Marlin out of the B=1 measured path |

The honest read of the partial 1.6B-4 data: **SM-partition isolation is
contention-gated** — it hurts at low contention (green-ctx overhead with little
to isolate) and appears to help at high contention. A *definitive*,
CI-confirmed statement was not produced and is deferred (§4).

**Claims explicitly NOT covered by CP 5.4 (deferred):**

| not covered | deferred to |
|---|---|
| per-tenant latency under realistic agent workloads | CP 5.5 |
| POOL-scaling at B=20/50/100 with weight sharing | CP 5.5 |
| failure-mode behaviour (D5 resize churn, D9 crash recovery) | Phase 6 |
| long-running soak (24 h+) | Phase 6 |
| per-tenant power metering (D7 — NVML is whole-GPU) | Phase 6 |

## 3 — WHAT CP 5.4 ESTABLISHED

- An **SM-arbitration ledger** in the kmod — 15 × 8-SM groups (120 SM;
  measured H100 green-ctx split), additive ioctls, do_exit reaper.
- A **two-clause disjointness invariant** — structurally (build-time
  self-verify) and at runtime (per-round `%smid` probe, orchestrator-centralised
  clause-1 ⊆-allocation and clause-2 pairwise-disjoint checks).
- Two tenant classes — **PARTITION** (isolated green-context SM slice on
  `libcipher_rt`) and **POOL** (batched executor on `libcipher_v2`), arbitrated
  by the single kmod ledger, disjoint by construction.
- A **naive multi-tenant baseline** characterised (Step 1.6B-3, OP-2/5/asym,
  5 reps, correctness-gated).
- **Substrate composition verified** — the 1.6B-2A re-validation gate confirmed
  the PARTITION green-ctx → `CIPHER_CP54_ALLOCATE` path survived the Track 2
  and Track 3 anchor rotations.
- The SM-arbitration substrate primitive is **built, correct, and ready** for
  CP 5.5's 100-tenant scenarios.

## 4 — WHAT CP 5.4 DELIBERATELY DEFERRED

- **1.6B-4 — synthetic Arm-B-vs-Arm-A CV comparison.** Partially run (OP-2
  complete, OP-5 3/5 reps — §2). Not completed: the synthetic B=1-decode CV
  contest is not load-bearing for the product pitch — it would not have
  informed a customer SLA claim, which depends on *real-workload* latency. The
  contention-gated finding above is the useful residue; the CI-confirmed
  version is folded into the CP 5.5 real-workload latency measurement.
- **1.6P — POOL-scaling stress test (B=20/50/100).** CP 5.5 measures POOL
  scaling at production scale with real workloads — measuring it twice, once
  synthetically, adds a number nobody pitches.
- **1.7 — failure modes (D5 resize churn, D9 crash recovery).** Operational /
  production-hardening concerns, not architectural-proof concerns → Phase 6.
- **Bootstrap-CI separability on synthetic CV.** Not a pitchable artefact;
  the CI that matters is on production-workload latency at CP 5.5.

Rationale, in one line: CP 5.4's job was to **prove the arbitration substrate
correct**. It did. Measuring that substrate's behaviour under *synthetic*
stress is not the same as measuring the *product* — the product is measured at
CP 5.5 under the workload customers actually run.

## 5 — V1 BOUNDARIES (carried to CP 5.5)

- Single-GPU H100 substrate.
- 15-group ledger — max 15 × 8-SM concurrent partition groups (120 of 132 SM
  usable; 12-SM remainder unallocatable, accepted/documented).
- **B=1 PARTITION tenants validated**; B>1 partition × Marlin needs
  verification (deferred — FUTURE_SCOPE/A or CP 5.5).
- Synthetic-workload arbitration validated; **production-workload arbitration
  is CP 5.5**.
- SM-partition latency isolation is **contention-gated** (§2) — not a
  universal win; this boundary is carried into the CP 5.5 latency design.

## 6 — V2 / PHASE 6 DEFERRALS

- **D7** — per-tenant power metering (NVML is whole-GPU only).
- **D8** — KV cache at production scale, B ≈ 95–100.
- **D12** — asymmetric-model matrix.
- **D13** — DVFS × mixed deployment.
- **1.6P** — POOL-scaling stress test (folded into CP 5.5 production scale).
- **1.7** — failure modes (D5 resize-churn rate, D9 concurrent-crash recovery).
- Bootstrap-CI separability on real-workload latency.

## 7 — PRODUCTION DEPLOYMENT GUIDANCE

- CP 5.4 closes **substrate-arbitration primitive correctness** — the
  foundation CP 5.5 builds on. The ledger, the disjointness invariant, the two
  tenant classes, and DSM (Track 3) are production-ready primitives.
- **Real-workload per-tenant latency, SLA characterisation, fairness, and
  noisy-neighbour isolation are CP 5.5** (substrate under realistic
  agent-shaped load — bursty, persistent context, realistic duty cycle) **and
  Phase 6** (production hardening). The contention-gated isolation finding
  (§2) says this measurement must be done under real load — a synthetic
  contest would mis-state the customer SLA either way.
- The substrate primitives compose with vLLM via **FUTURE_SCOPE/A** — the next
  workstream, which begins with a design memo (paperwork only).

## 8 — ANCHOR LINEAGE AT CP 5.4 PARTIAL CLOSE

| artifact | md5 | state |
|---|---|---|
| `cipher_kmod.ko` | `008b3c66` | unchanged — Track 2 SC5 anchor, still current |
| `libcipher_rt.so` | `83afd1ca` | unchanged — Track 3 close anchor, still current |
| `libcipher_v2.so` | `cc0479b8` | unchanged through CP 5.4 |
| `cipher_kv_bridge.so` | `c04b0c39` | unchanged — Track 2 SC3 anchor |

CP 5.4 rotated **no anchors** — all sub-steps were verification + throwaway
`cp54_s16_*` harnesses. All fallbacks preserved with md5 verification
([[cipher-kbuild-clean-wipes-ko]]). Verified unchanged at this partial close
(`md5sum`, 2026-05-19); `/proc/cipher/{arenas,migrations}` operational; dmesg
clean.

---

## VERDICT — CP 5.4 PARTIAL CLOSE

The SM-arbitration substrate is **built, correct, composition-verified, and
ready for CP 5.5**. Steps 1.6B-4 / 1.6P / 1.7 are deferred — deliberately, with
the reasoning recorded — to where the measurements actually inform the product:
CP 5.5 (real-workload scale) and Phase 6 (production hardening). The partial
1.6B-4 data is preserved and its contention-gated finding recorded honestly.

**Next workstream: FUTURE_SCOPE/A** — CIPHER + vLLM composed production
architecture — starting with a design memo (paperwork only, no source, no GPU).

**Awaiting final adjudication of CP 5.4 partial closure.**
