---
name: cipher-phase5-scoped
description: "Phase 5 scoped 2026-05-17 — 5 CPs, paper-work done, CP 5.1 vLLM/TGI is next GPU work"
metadata: 
  node_type: memory
  type: project
  originSessionId: 927085f0-ed71-4d9a-8a3c-286f09e5e63c
---

Phase 4 CLOSED via CP 4.8 (2026-05-17); the 24h sustained-load soak was DEFERRED from CP 4.8 to CP 5.5 (so no soak is running — supersedes the old cp48-inflight memory). Phase 5 goal: 100 concurrent real-decode tenants on one H100, integrated end-to-end, measured honestly; target close Q4 2026.

Phase 5 = 5 CPs (**2/5 CLOSED**), each its own design memo in `cipher-fusion-evidence/`:
- CP 5.1 vLLM/TGI live-decode integration — **CLOSED 2026-05-17** (memo §4 #2 correctness only; see [[cipher-cp51-closed]])
- CP 5.2 KV offload hierarchy HBM→DRAM→NVMe — **CLOSED 2026-05-17** (memo §4 #1 correctness only, Option A snapshot-on-preempt; see [[cipher-cp52-closed]])
- CP 5.3 partition-aware Marlin — **CRITICAL PATH**, IN FLIGHT. STEP 1 (split-K generalisation diagnostic) PASS + **adjudicated** → 4-wk bounded-grid-rework branch. **STEP 2 (Axis A) executed + PASS (2026-05-18)**, awaiting adjudication: `grid←green-ctx SM count` + `PrimaryCtxGuard` scoped to quant; gates A-num/B/C all PASS re-verified (no KU1 deadlock, 8-SM no spill, 2 partitions disjoint). A-tok 3% was a **broken full-GPU reference** — triangulated vs gold FP16: partition build produces coherent decode on TinyLlama+Mistral; the shipped libcipher_rt `c2c5d313` full-GPU Marlin is degenerate on real decode = **finding F1**, separate audit (do NOT fold into CP 5.3). STEP 2 ships libcipher_rt `c2c5d313`→`dc804eb3`. Report `cp_5_3/CP_5_3_STEP_2_REPORT.md`; tarball `cp_5_3_step2_evidence.tar.gz` md5 `c94473a4`. **STEP 2B = Axis B** (two-model INT4 acceptance collapse 0.490→0.036) next; CP 5.3 closes only when 2A+2B both land
- CP 5.4 per-tenant arbitration extension (2-3 wk); prereq CP 5.2 — dependency now satisfied
- CP 5.5 100-tenant integration measurement + the deferred 24h soak (1-2 wk); prereq 5.1-5.4

Paper-work phase COMPLETE 2026-05-17: `PHASE_5_PLAN.md`, CP 5.1/5.2/5.3 design memos, and `PHASE_5_CP_5_3_SONG_HAN_ENGAGEMENT_SCOPE.md` (Song Han advisory: unmodified YC FAST template, 0.5-1% equity, 2yr vest / 6mo cliff, no cash; milestones M1-M4 gate via mutual-termination right, not custom vest tranches). Pod suspended after; GPU resumes when CP 5.1 ready (~2-3 wk prep).

Campaign snapshot at suspend: `cipher_campaign_state_20260517_phase5_paperwork.tar.gz` (md5 08a0499e..., 1137 files) — copied to persistent NFS `/lambda/nfs/Anil/` (local `/home/ubuntu` is ephemeral; NFS was previously empty).

**Why:** Phase 5 is the real-serving-stack + multi-tenant phase of the 23-CP campaign ([[cipher-fusion-campaign]]).
**How to apply:** CP 5.1 + CP 5.2 closed; next GPU work is CP 5.3 (partition-aware Marlin, critical path); follow design-memo→approve→build discipline per [[cipher-fusion-campaign]] and [[cipher-phase-discipline]].

Anchors: kmod `e2f50452`, libcipher_v2 `86618c30` held; libcipher_rt `c2c5d313`→`dc804eb3` (CP 5.3 STEP 2 fix shipped, anchor move pending STEP 2 adjudication; prior preserved as `libcipher_rt.so.pre_cp5_3_step2`). Finding F1's audit may move it again.
