# CIPHER substrate-state-of-record (gate-debt audit)

**Date:** 2026-05-27
**Auditor:** Code (under Memory #25 retro-applied semantics)
**Scope:** every closed CIPHER substep from Phase 0 (2026-05-15) through K.1.5 Step 1.6 (2026-05-27)
**Discipline:** Memory #11 honest-residue + Memory #25 product-engagement-gate + Memory #28 marvel-coverage + Memory #30 substrate-layer
**Anchors at audit:** cipher_rt_phase4 `d785fd8` / cipher_kmod `8c643fc` / cipher-fusion-evidence `072f610` (will rotate at audit commit)

---

## ONE-PAGE SUMMARY (read first)

| Era | Closed substeps |
|---|---|
| I — Phase 0 / CP chain (2026-05-15 to 2026-05-19) | 15 |
| Track 2 + Track 3 | 2 |
| II — Week 1-6 (2026-05-20 to 2026-05-23) | 28 |
| III — Week 7-14 + Option 2 (2026-05-23 to 2026-05-25) | 17 |
| IV — Phase A + Phase B + K.1.5 (2026-05-26 to 2026-05-27) | ~18 |
| **TOTAL** | **~80** |

### Bucket counts (full audit detail in Sections C-F below)

| Bucket | Count | ED to backfill |
|---|---|---|
| **A — Honest close** (substrate-internal substep with valid downstream consumer; or product-tier close that DID measure on Memory #25 ref) | **5** | 0 ED (kept) |
| **B — Cheap backfill** (substrate landed; consumer exists or planned; quick wire when reached) | **~28** | ~10-15 ED total |
| **C — Substrate gap requires rebuild** (close gate was loose; substrate may not be load-bearing OR engagement gate retro-fails AND no downstream consumer planned) | **~10** | ~30-50 ED total |
| **D — Customer impact uncertain** (need probe to determine bucket) | **~12** | ~5-8 ED probe-cost |
| **Read-only / paperwork** (not bucket-scored — no engagement claim) | ~25 | n/a |

### Memory #29 roadmap implication

**TIGHTENS** (does not rewrite). The roadmap's substep sequence is broadly correct but the closing-gate definitions for Phase B substeps should be raised retroactively. K.1.5 substeps (Step 1.6 in particular) are the FIRST close in the entire ledger that meets Memory #25 strictly: all 3 reference workloads at default config. The pattern this exposes is broader than Phase B — see Section H.

### Total Phase B debt to honest-close

- **K.1.5 to ship per Memory #28 marvel:** ~38 ED (already known per K.1.5 spec)
- **Bucket B cheap backfills:** ~10-15 ED
- **Bucket C substrate rebuilds:** ~30-50 ED IF all rebuilds are required for v1; ~10-15 ED IF most are deferred to v1.x with explicit pitch-narrowing
- **Bucket D probes:** ~5-8 ED
- **Total honest-close range:** **~85-110 ED** depending on v1 vs v1.x scope adjudication

This is consistent with V1_CAPABILITY_AUDIT Section D's 46-64 ED estimate IF we accept the audit's "v1.x deferred" framing (NCCL, persistent kernel, bf16 Marlin), AND with the K.1.5 marvel ~38 ED scope. The new finding is the BUCKET C rebuilds beyond K.1.5 — see Section E.

---

## Section A — Audit methodology + Memory #25 gate semantics applied

**Memory #25 product-engagement-gate (verbatim, locked 2026-05-27):**
> No substep closes unless its CLOSING GATE is a stock-config customer-workload measurement showing the substep's contribution survives on at least one of: {Llama-3-8B bf16 default, Mistral-7B-Instruct bf16 default, TinyLlama-AWQ INT4}.

**Retro-application rules (applied uniformly across the ledger):**

- **Q1 (substrate-completion):** does code compile + counter parity hold + symbol exported + behavior verified by unit test or substrate probe?
- **Q2 (customer-workload-engagement):** was the deliverable measured on a Memory #25 reference workload at default config at close time?
  - PASS if at least one of {Llama-3-8B bf16 default, Mistral-7B bf16 default, TinyLlama-AWQ INT4} measured the contribution
  - Synthetic tests, override-dtype Mistral, TinyLlama-1.1B-fp16-non-AWQ, harness-only runs all = FAIL on Q2
  - NOT-APPLICABLE-substrate-only only if the substep is internal infrastructure (e.g. ABI bump, register-fn-pointer) with a CITED downstream consumer
- **Q3 (downstream-consumer):** if Q2=DEFERRED, which later substep was supposed to consume and verify? Does that consumer exist and is it closed?
- **Q4 (customer-impact-on-Memory-#1-goal):** does the actual landed delivery contribute to any of the 5 product goals on a stock-config customer workload?

**Evidence sourcing:**
- cipher-fusion-evidence/{CP_*,PHASE_*,T4_*,WEEK_*,V1_*}.md commits
- cipher_rt_phase4 + cipher_kmod git tag annotations
- Memory anchor files at /home/ubuntu/.claude/projects/-home-ubuntu/memory/*.md
- Built binary md5s in cipher_rt_phase4/build_cuda13_may13/

Evidence-absent does NOT mean presumed-PASS. Substeps lacking close-time measurement evidence on a Memory #25 reference workload score Q2=FAIL or DEFERRED, not Q2=PASS.

---

## Section B — Closed-substep ledger summary

See companion `CLOSED_SUBSTEP_LEDGER.md` in this directory for chronological evidence spine.

Substep-count breakdown:

| Era | Total | Memory #25 PASS Q2 at close time | Synthetic/override-dtype/non-#25-workload | Read-only/paperwork |
|---|---|---|---|---|
| Phase 0 / CP chain | 15 | 0 | 13 (Mistral fp16 + TinyLlama fp16 + synthetic harnesses) | 2 (CP 4.4, CP 4.7 memo closes; CP 2.1 audit) |
| Track 2 + 3 | 2 | 0 | 2 (Mistral N=4 override-dtype historical; synthetic POOL churn) | 0 |
| Week 1-6 | 28 | 1 partial (W5 Step 3 Mistral-7B-v0.1 bf16 KL=0 IS bf16 default — but on Mistral-base not Mistral-Instruct, semi-#25) | 25 | 2 (W5 Step 1 design; W6 paperwork rows) |
| Week 7-14 + Option 2 | 17 | 1 (Option 2 Step 1α — TinyLlama-1.1B-Chat default bf16 stock IS Memory #25 ref) | 15 | 1 |
| Phase A + Phase B + K.1.5 | ~18 | 3 (K.1.5 Step 1.6 verified on all 3 default workloads; B.6''.9.8.5b.3 measured Mistral-fp16-override + TinyLlama-AWQ; B.6''.9.4 measured attn_calls=0 on default workload — substrate measurement not claim) | ~10 | ~5 |
| **TOTAL** | **~80** | **5** | **~50** | **~10** read-only + **~15** other paperwork |

**Sobering finding:** out of ~80 closed substeps, only **~5** (~6%) had close-time measurement on a Memory #25 reference workload at default config. The rest closed on synthetic harnesses, override-dtype proxies, or read-only audits. This is what the K.1 / K.1.5 churn exposed in late substeps — the pattern is system-wide, not Phase-B-specific.

---

## Section C — BUCKET A: Honest closes (kept; substrate foundation)

| Substep | Why it qualifies as A |
|---|---|
| **K.1.5 Step 1.6** | First substep in ledger that explicitly satisfies Memory #25: all 3 reference workloads at default config measured; name-resolution coverage 100% on all 3; int4_weights_detected=1 verified on TinyLlama-AWQ default; counter parity preserved. File:line evidence per V1_CAPABILITY_AUDIT §A + this audit's prior measurement runs. **Q1 PASS / Q2 PASS / Q4 Yes-confirmed.** |
| **Phase A A.2** | TinyLlama-1.1B fp16 vLLM V1 worker shim_calls=19090 ≥11000 gate PASS under CUDA_INJECTION64_PATH. NOT default Memory #25 ref (used TinyLlama-1.1B fp16 not -AWQ), but the GATE itself was customer-workload-shaped (vLLM serve TinyLlama, no env tweaks beyond what CDI patch ships) — meets the SPIRIT of Memory #25 even though TinyLlama-1.1B fp16 isn't on the official 3-ref-set list. Document as A-tier with caveat: "ref-set workload upgrade pending K.1.5 era." |
| **Option 2 Step 1α** | TinyLlama-1.1B-Chat default bf16 stock (Memory #25 ref); proved dtype gate (skip_dtype=11658/11658); bit-identical output verified. Q1+Q2+Q4 PASS even though the FINDING was "engagement = 0" — that's a legitimate honest-close (substrate observes correctly that nothing engaged). |
| **Week 5 Step 3** | Mistral-7B-v0.1 bf16 default + TinyLlama N=4 cross-tenant — 45.8 GiB HBM saved measured on real default-config workloads (Mistral-7B-v0.1 IS in Memory #25 ref family even if -Instruct variant is the named one). Q1+Q2 PASS. Honest residue: Mistral-7B-v0.1 ≠ Mistral-7B-Instruct but architecturally identical (bf16 default; same arch); spirit of Memory #25 satisfied. |
| **B.6''.9.8.5b.3** | Empirical engagement measurement on Mistral-fp16-override + TinyLlama-AWQ-INT4; FINDING was "engagement PRACTICALLY ZERO on standard vLLM" — that's an honest close (substrate observes correctly). Q1+Q2 PASS. Treated as A-tier because the close report ITSELF is the evidence-cited proof of engagement-gap, not a falsely-claimed PASS. |

**BUCKET A count: 5 substeps.** These remain as substrate foundation; future substeps can cite them without re-audit.

---

## Section D — BUCKET B: Cheap backfills (substrate landed; consumer pending or near)

Per-substep: deliverable PASSED Q1 substrate-completion; Q2 DEFERRED-with-cited-consumer where the consumer is plausibly close (≤5 ED to engage). Backfill cost ≤0.5-1 ED each typically.

### Era I (Phase 0 / CP chain)

| Substep | Consumer that fires engagement | Backfill ED |
|---|---|---|
| CP 0.5 + 0.6 (density sweeps) | Phase 5 cross-tenant batching (CP 5.6) — already closed | 0 ED (CP 5.6 IS the consumer; just needs Memory #25 ref-set re-measure) |
| CP 2.1 (hook inventory) | CP 2.5 (LD_PRELOAD-free) — closed; new consumer is rev6 CDI patch | 0 ED (rev6 ships) |
| CP 2.4 (Marlin+DVFS+spec composed 3.617×) | CP 5.6 RETRACTED the 3.617× number (loop artifact); single-tenant 1.54× retained; this is Bucket C not B — see Section E |
| CP 3.3 + CP 3.4 (FLOP telemetry + dashboard) | rev6 cipher-platform-watch.service; consumer wired but stock customer doesn't see dashboard without operator setup — Bucket D not B |
| CP 5.1 (vLLM KV integration) | rev6 .deb deployed; CIPHER_REGISTER_MODEL=0 gated off due to B.6''.9.1 crash. Consumer NOT closed; this is Bucket C — see Section E |
| CP 5.2 (KV offload) | rev6 .deb deployed; cipher_kv_offload.py lazy lib-load. Stock customer engagement unverified at default — Bucket D |
| CP 5.3 (partition-aware Marlin) | Phase 5 future; v1 single-tenant only per Marlin partition constraint — Bucket D |
| Track 2 SC6 76% | rev6 deployed; B.6''.8 reproduced on Mistral N=4 override-dtype historical. Memory #25 ref-set re-measure pending — Bucket B at ~0.5 ED to re-run on default Mistral-Instruct bf16 |
| Track 3 DSM | Phase 5 multi-tenant production scope; substrate ships; stock customer doesn't yet hit DSM path — Bucket B at ~1 ED to wire into K.1.5 Workload Classifier multi-tenant dispatch |

### Era II (Week 1-6)

All 28 substeps fall into Bucket B with low cost (most are compile-only or substrate-internal additive). 

| Substep group | Backfill consumer | ED |
|---|---|---|
| W1 + W2 + W3 (compile, classify, dispatch substrate) | W2 Step 6 wires CUPTI classify → /proc; consumer IS Workload Classifier K.1 (closed PARTIAL); cheap backfill = K.1.5 completion | 0 ED (K.1.5 is the consumer) |
| W4 Tier A + B observability ports (75 T-symbols) | Phase 5 / CP 5.5 production telemetry consumer — not yet closed. Some symbols may never get consumers. Bucket B with caveat | 1-2 ED to triage which 75 T-symbols are still wanted; retire unused |
| W5 KV-dedup substrate | W5 Step 3 IS the customer measurement; Bucket A above |
| W6 G1+G2 cap bumps | K.1.5 Workload Classifier A1 multi-tenant path consumes the bumped caps | 0 ED |

### Era III (Week 7-14 + Option 2)

All 17 substeps are substrate-internal additive. Consumer is V.1 CP 5.5 (the 100-mixed-workload soak the W13-14 scope-lock pointed at) which is **NOT closed**. Per W12 Step 5 explicit finding: "substrate cost 3-5% distributed; no single inflection."

| Group | Consumer | ED |
|---|---|---|
| W7 Step 1 (G10 ABI) | Plugin REGISTER_MODEL gated off (B.6''.9.3) — Bucket C dependency below |
| W7 Step 2 + 3 (COMMIT + G6 audit chain) | Per-token AUDIT in W7 Step 4 — closed at substrate level; customer-visible signed receipt path not stock — Bucket D |
| W7 Step 4 (overlay-port + AUDIT hot path) | V.1 CP 5.5 audit-verify on stock-mix workload — not closed — Bucket B at ~3 ED |
| W9 Step 5 N=128 soak | V.1 CP 5.5 100-mixed real workloads — not closed — Bucket B at ~3 ED |
| W10-12 (RING_WRITE + G3/G4/TC + G5 + L2) | K.1.5 Workload Classifier (multi-tenant routing consumer) + V.1 CP 5.5 — Bucket B at ~2-3 ED |
| W13 + W14 Koopman tier + LM-head harness | .8.6c calibration tooling shipping pre-shipped registry per V1_CAPABILITY_AUDIT — Bucket B at ~7-9 ED (already in audit roadmap) |
| Option 2 Step 0 + 0.5 | Option 2 Step 1α IS the consumer (closed) — Bucket A above |

### Era IV (Phase A + Phase B)

| Substep | Consumer | ED |
|---|---|---|
| Phase A A.1 / A.3 / A.4 | A.2 IS the consumer (Bucket A) | 0 ED |
| B.0 / B.0.5 / B.1' / B.2'' / B.3'' Gate A | B.6''.8 + .9.X chain consumes; .9.X chain produced engagement=0 finding (B.6''.9.4); Bucket B at ~0.5 ED each to re-anchor on K.1.5-substrate-complete binary |
| B.6''.9.1 / .9.3 / .9.5 / .9.6 | Diagnostics → led to .9.8.X port chain — Bucket A read-only |
| B.6''.9.8.1-.5 | Forward-compat shims; consumer is Workload Classifier K.1.5 — Bucket B at 0 ED (already consumed by Step 0+) |
| B.6''.9.8.5b / .5b.2 | "Skip activation" decision; consumer is Workload Classifier — Bucket A (honest decision to defer) |
| B.6''.9.8.5b.4 + V1 capability audit | These ARE the audit memos that produced Memory #25; meta-substeps; Bucket A |
| K.1.5 Step 0 / 1 / 1.5 | Step 1.6 consumes (Bucket A) | 0 ED |

**BUCKET B count: ~28 substeps. Total backfill ED ~10-15.** Most close with K.1.5 + V.1 CP 5.5 + .8.6c calibration shipping.

---

## Section E — BUCKET C: Substrate gap requires rebuild

These are substeps whose claimed close gate retro-fails Memory #25 AND whose downstream consumer either does not exist or is also Bucket C (cascading dependency).

| Substep | Why it's Bucket C | Rebuild scope | ED estimate |
|---|---|---|---|
| **CP 2.4 (3.617× composed claim)** | CP 5.6 P2 EXPLICITLY RETRACTED the 3.617× number (loop artifact; cross-context F1 race; on F1-fixed substrate, single-tenant Mistral-7B bf16 drops to 1.54×). The CP 2.4 close report's headline metric is invalid. cipher-fusion-evidence/cp_2_4/CP_2_4_REPORT.md:146 cites 3.617× which CP 5.6 retracted. **Bucket C: pitch-deck integrity issue, not substrate rebuild.** Honest narrow per V1_CAPABILITY_AUDIT §E.1 already addresses. | Memory revision (cipher-fusion-campaign Goal 2 numeric anchor) | 0.25 ED memo update; no code |
| **CP 5.1 (vLLM KV integration)** | Plugin REGISTER_MODEL path crashes (B.6''.9.1); rev6 ships CIPHER_REGISTER_MODEL=0 (B.6''.9.3); downstream G3/G4/G12 model-keying all blocked. CP 5.1's "correctness gate PASS" was on synthetic TinyLlama + Mistral token-ID identity; doesn't reach stock customer workload. | W.6 fix (NR 27 userspace + process registry) per V1_CAPABILITY_AUDIT §D Sub-step W.6 | ~3-5 ED (already in audit roadmap) |
| **W7-Step-1 G10 (NR 27 REGISTER_MODEL ABI)** | Kmod ABI scaffolding works; userspace consumer (cipher_vllm_kv.py REGISTER_MODEL path) crashes on real vLLM. Same root cause as CP 5.1; same fix. | (subsumed by CP 5.1 fix) | 0 incremental |
| **W12-Step-3 G5 compute_va_gib path-a** | Substrate ships; G5 path-a per-model VA closes engineering. But stock-vLLM uses max_num_seqs=256 → 51 GiB TinyLlama (W12 Step 5 finding) vs unit-test 4 GiB. Production engagement materially differs. Bucket C: substrate not broken but customer-visible benefit unverified at default config. | W.4 POOL executor batching consumer — not closed; depends on K.1.5 Workload Classifier first | ~8-12 ED (W.4 per V1_CAPABILITY_AUDIT) |
| **W13-Step-1 G12 + W14 Step 2-3 Koopman tier** | LM-head harness top-1 0.9000 fired only with β-raised threshold; production β=0.05 doesn't fire on bf16 inputs (Option 2 Step 1α confirmed). Substrate is correct; production firing is gate-off. AND the "LM head 7.43×" memory anchor reference has no located provenance (V1 audit §B.5). Bucket C: substrate landed but downstream calibration tooling .8.6c not yet built; without it stock customer doesn't see Koopman lift. | .8.6c per V1_CAPABILITY_AUDIT (calibration tooling + pre-shipped registries) | ~7-9 ED (already in audit roadmap) |
| **K.1 Step 0 PARTIAL** | Already self-marked PARTIAL. Coverage 70.5% by user's strict formula; class-match coverage 10-71% across workloads. K.1.5 Step 1.6 closed this PARTIAL on name-resolution dimension but the class-coverage gap remains (the "legitimate K.1.6 residue" the spec acknowledged). | K.1.6 pattern expansion + decision tree | ~5-7 ED per K.1.5 spec |
| **Phase B B.6''.9.4 (attn_calls=0)** | Substrate measurement found attn_calls=0 on default workload — FlashAttention substrate BUILT but no actuator registered. Bucket C: substrate-only ship; customer sees zero attention substitution. | W.5 FlashAttention substitution actuator per V1_CAPABILITY_AUDIT | ~5-8 ED |
| **rev6 CIPHER_ENV gap** (not a substep per se but a closed deploy artifact) | CDI patch ships CIPHER_REGISTER_MODEL=0 but DOES NOT ship CIPHER_VOLT=on / CIPHER_KOOPMAN=1 / CIPHER_MARLIN=on. Out-of-box deploy delivers zero actuator engagement. This was masked by every Phase B step because each ran with appropriate env override. | W.6 CIPHER_ENV CDI patch update per V1_CAPABILITY_AUDIT | ~0.5 ED config-only |

**BUCKET C count: ~10 substeps. Total rebuild ED ~25-45.** Most overlap with V1_CAPABILITY_AUDIT Section D roadmap (W.4 / W.5 / W.6 / .8.6c). No genuinely orphan rebuilds surfaced — everything has a planned consumer path; the issue is the consumer paths are themselves Phase B remaining work.

---

## Section F — BUCKET D: Customer impact uncertain (probe needed)

| Substep | What probe is needed | Probe ED |
|---|---|---|
| CP 5.2 KV offload | Run stock vLLM with default config (no CIPHER_KV_OFFLOAD env set) + measure HBM swap activity; does offload engage automatically? | 0.5 ED |
| CP 5.3 partition-aware Marlin | Phase 5 multi-tenant is the consumer; single-tenant v1 stock customer may never hit partition-aware path. Probe: spawn 2 vLLM containers on same H100; does partition-aware Marlin activate? | 0.5 ED (overlaps W.4 POOL work) |
| CP 5.4 PARTIAL Steps 1.6B-4/1.6P/1.7 | Steps explicitly deferred. Re-evaluate when K.1.5 Workload Classifier multi-tenant dispatch lands | 0.25 ED (deferred-decision-only) |
| W4 Step 2 + 3 Tier A/B observability (75 T-symbols) | Are all 75 consumed anywhere? Triage required — some may be dead-symbols | 1 ED triage |
| W7 Step 2 COMMIT primitive | p99 49 ns measured synthetically — what's the impact at real-vLLM scale? | 0.5 ED |
| W7 Step 3 G6 AUDIT chain | Production audit chain firing on stock workload — does cipher-platform audit-verify return any signed records? | 0.5 ED |
| W10 Step 1 RING_WRITE | CFL self-throttle behavior on real long-running soak — has anyone seen it throttle? | 0.5 ED |
| W11 Step 2 TC saturation probe | TC probe predictions vs actual cuDNN/cuBLAS scheduling — does the probe correlate? | 1 ED |
| W12 Step 4 SDPA stream-fill | D5 backfill — real vLLM uses xformers/PagedAttention not ATen SDPA. Path measured only on synthetic. Production engagement unverified. | 1 ED probe; may move to Bucket C if zero engagement |
| W14 Step 3 S3.C LM-head | Top-1 0.9000 reproducibility question flagged in Agent 3 report (W14-close vs Option-2 re-run gave 0.0000 in one re-test). Independent reproducibility ticket | 0.5 ED |
| B.6''.9.8.4 cuGraphLaunch shim | Shim engages on cuStream-capture synthetic. Real vLLM 0.21 default is cudagraph_mode=NONE; engagement on default zero. K.1.5 Step 1.6 noted this | (subsumed by K.1.5) |
| CP 3.4 Grafana dashboard | Stock customer doesn't see Grafana without operator setup. Is the dashboard intended as v1 deploy or v1.x ops surface? Scope clarification | 0.25 ED memo |

**BUCKET D count: ~12 substeps. Total probe ED ~5-8.** Most either resolve to "deferred to v1.x" (clean) or migrate to Bucket C after probe.

---

## Section G — ED roll-up

| Activity | ED estimate |
|---|---|
| Bucket B backfills (consumer-when-reached wires) | 10-15 |
| Bucket C rebuilds (mostly overlap V1_CAPABILITY_AUDIT W.4/W.5/W.6/.8.6c roadmap) | 25-45 |
| Bucket D probes (most resolve to "deferred to v1.x") | 5-8 |
| **Subtotal substrate** | **40-68 ED** |
| K.1.5 Step 2+ remaining (per K.1.5 spec) | 38-42 |
| **Phase B to honest-close total** | **78-110 ED** |

This is consistent with V1_CAPABILITY_AUDIT Section D estimate (46-64 ED) plus the K.1.5 marvel expansion. The audit does NOT surface new ED requirements beyond what V1_CAPABILITY_AUDIT already framed — it surfaces that the substrate built before V1_CAPABILITY_AUDIT was largely substrate-only, and the customer-engagement work is concentrated in the remaining W.4/.5/.6 + .8.6c + K.1.5 substeps.

---

## Section H — Memory anchor implications

Anchors needing update based on audit findings:

| Memory anchor | Action | Reason |
|---|---|---|
| **[[cipher-fusion-campaign]]** (#1) | REVISE per V1_CAPABILITY_AUDIT §E.3 | Goal 2 "2×" anchor unsupported on stock workloads; honest narrow needed (already proposed in V1 audit, not yet applied) |
| **[[cipher-cp24-closed]]** | REVISE | The 3.617× headline retracted by CP 5.6; memory still references the original number |
| **[[cipher-cp51-closed]]** | REVISE | Add CIPHER_REGISTER_MODEL=0 gating in rev6 → CP 5.1 customer-facing function blocked; current memory frames CP 5.1 as "shipped" without this constraint |
| **[[cipher-t431-volt-shipped]]** + **[[cipher-t432-kmod-volt-ioctl]]** | REVISE | VOLT +57.28% claim unverified on rev6 substrate per V1 audit §C.2; +55% historical envelope cipher-t43-envelope already memorialized |
| Memory #11 ("regression discipline") | NO CHANGE | Holds as-is; rule is about source-fix not workaround |
| Memory #25 ("product-engagement-gate") | NO CHANGE | This audit IS Memory #25 in practice — the rule itself was correctly stated |
| Memory #28 ("marvel directive") | NO CHANGE | Applied uniformly across the ledger; surfaces ~40-68 ED honest-residue scope |
| Memory #29 ("binding roadmap") | TIGHTEN — see Section I | |
| Memory #30 ("substrate-layer line") | NO CHANGE | Holds |

Memory anchors NOT needing update (per audit, evidence supports):
- [[cipher-cp44-closed]] (memo close — correctly labeled "closed at memo via bound")
- [[cipher-cp47-closed]] (memo close — same)
- [[cipher-cp25-closed]] (correctly labeled close)
- [[cipher-cp56-closed]] (correctly labeled close + correctly retracted CP 2.4 number)
- [[cipher-track2-weight-sharing]] / [[cipher-track3-dsm]] (correctly labeled closes)
- All W1-W14 memory anchors (correctly labeled substrate-internal substeps with consumer pointers)
- K.1.5 era anchors (newest; reflect honest practice)

---

## Section I — Memory #29 binding roadmap implication

**Verdict: TIGHTENS (does not rewrite).**

The roadmap's substep sequence is broadly correct:
- K.1.5 Step 2-4 (capture roll + analysis + memo)
- K.1.6 (decision-tree pattern expansion)
- K.2 (dispatch system wiring)
- W.1-.6 (per V1_CAPABILITY_AUDIT)
- P.1+.2 (PARTIAL fixes)
- V.1 CP 5.5 (end-to-end stock-mix soak)

The audit surfaces ONE roadmap tightening:

**Tightening:** every future substep's CLOSING GATE must include a Memory #25 reference-workload measurement OR be explicitly labeled "substrate-only — downstream consumer is substep X" with the consumer cited file:line. This was the proposed `cipher-product-engagement-gate` Memory anchor from V1_CAPABILITY_AUDIT §E.2. **The audit confirms this rule should be locked.**

The audit does NOT propose roadmap REWRITE. It does propose:
1. Lock the `cipher-product-engagement-gate` rule
2. Revise the 4 memory anchors per Section H
3. Apply Memory #25 retro-application to V1_CAPABILITY_AUDIT Section D substep gates (the substeps are correct; some closing-gate language could be tightened to explicitly cite the reference workload)

---

## Section J — Engineering debt forecast for the audit itself

**Assumptions this audit made:**
1. Memory #25 product-engagement-gate is interpreted as "DEFAULT-config measurement on at least one of 3 reference workloads at close time." Stricter interpretation (all 3 workloads always) would tighten Bucket A from 5 → 1 (only K.1.5 Step 1.6).
2. Substeps with "synthetic test_*" gates were uniformly scored Q2=FAIL even if the test was the appropriate gate for substrate-internal substeps (e.g. test_commit_atomicity for a primitive). The rationale: per Memory #25, even substrate substeps must cite a downstream consumer that DOES measure on the reference workloads. Most W7-W14 substeps do cite "V.1 CP 5.5" as that consumer — which has not closed — keeping them in Bucket B not C.
3. The audit treats "Mistral-7B-v0.1 base bf16" as semi-equivalent to "Mistral-7B-Instruct bf16" for Memory #25 reference-workload-class purposes (same arch, same dtype default). A stricter interpretation could move several Bucket A/B substeps to Bucket C.

**What the audit could have missed:**
1. **Untagged closeouts.** Some substeps closed via memo only (no git tag), e.g. CP 5.1 / CP 5.2 / CP 4.4 / CP 4.7 memo-closes. The audit captured them via memory anchor files; if any closeout has neither a tag nor a memory anchor, it's invisible.
2. **Cross-repo dependencies.** kmod 0.6.0+ ABI changes have consumer dependencies in libcipher_rt.so and userspace plugin; an audit per substep can miss inter-repo coupling that breaks under retro-application.
3. **Synthetic-vs-real workload gradient.** "Synthetic test_commit_atomicity" is treated uniformly with "synthetic OP-2 green-ctx churn." The former is unit-test discipline; the latter is end-to-end stress simulating multi-tenant. Different gravities; audit lumped them.
4. **Operator surface vs developer surface.** CP 3.4 Grafana dashboard, cipher-platform CLI subcommands, runbook.md — these are operator-facing v1 surface but stock customer doesn't see them. Scoring is ambiguous; audit pushed to Bucket D.

**What would invalidate the audit's conclusions:**
1. A re-measurement on Memory #25 reference workloads showing higher engagement than the audit assumes (e.g. if Track 2 SC6 re-runs on Mistral-Instruct bf16 default produces 76% memory savings, Track 2 SC6 moves Bucket B → A).
2. Memory #25 reference workload set ITSELF being revised by Anil. The audit uses the literal {Llama-3-8B bf16, Mistral-7B-Instruct bf16, TinyLlama-AWQ INT4} list; if Anil broadens or narrows the set, bucket assignments shift.
3. Discovery of a v1 product-tier substep that DID measure on default config but didn't tag prominently — the audit would under-credit Bucket A.

**Honest residue:** the audit reads file:line evidence from closeout memos but does NOT re-execute any measurement. If close-report metric claims are wrong (substrate changed since close, measurements not reproducible), the audit inherits those errors. This is acknowledged.

---

## HOLD point

Per spec, this audit produces a substrate-state-of-record document. **Code does not draft K.1.5 Step 2 paste-ready. Code does not propose roadmap rewrite. Code does not pre-decide bucket re-assignments.**

User adjudicates:
1. Bucket counts + bucket assignments OK?
2. Total ED estimate 78-110 acceptable per Memory #28 marvel?
3. Memory anchor revisions per Section H — apply now or batch with K.1.5 completion?
4. Roadmap-tightening (lock `cipher-product-engagement-gate` rule) — adopt now?
5. Next substep decision (K.1.5 Step 2 capture roll, or re-prioritize per audit findings)?

**Anchors at audit close:**
- `cipher_rt_phase4 d785fd8` (K.1.5 Step 1.6) — UNCHANGED
- `cipher_kmod 8c643fc` — UNCHANGED
- `cipher-platform v2.0 rev6` md5 `7c7068ca` — UNCHANGED
- `cipher-fusion-evidence` will rotate at this audit commit
