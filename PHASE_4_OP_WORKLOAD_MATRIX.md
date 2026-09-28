# Phase 4 — Op-to-Workload Applicability Matrix (T4.0.8)

**Date:** 2026-05-13
**Status:** Binding source of truth for which ops fire on which workloads. Used by P4.2–P4.7 to determine declared-workloads-set per cluster checkpoint.

## Scope

33 canonical CIPHER ops × 24 binding workloads (WL01–WL24 per `PHASE_4_ARCHITECTURE.md`).

The "33 ops" canonical list = 12 core Stage 0/1/2 + 21 overlay (numbered 13–31 + STRAGGLER + NCCL P2P).

Audit-surfaced extensions (PARTITION_ROUTER, SM_PACKER, GREEN_CTX, L2_PERSIST, PERSIST_ENGINE, KV_COMPRESS, KV_REDIRECT, FUSION, FUSION_KERNELS, FLOW_PATTERNS, FLOW_RECORDER, FLOW_SUBSTITUTE, WEIGHT_SHARE, WEIGHT_COMPRESS, ATTN_KOOPMAN, GRAPH, GRAPH_INSPECT, OVERLAP, MEM_LAYOUT, RECIPES, MARLIN, SUBSTITUTE_V2, THERMAL_FEEDBACK, POWER_CAP, PARAM_RECOVERY, VMM, FP8_COMPUTE) are **named aspects of the 33 canonical ops** rather than separate rows:

| Extension | Subsumed by canonical op |
|---|---|
| PARTITION_ROUTER, SM_PACKER, GREEN_CTX | ARBITRATE (Stage 2 partition decisions) |
| L2_PERSIST, PERSIST_ENGINE, MEM_LAYOUT | PREDICT (Op 17) + SUBSTITUTE (Stage 0) |
| KV_COMPRESS, KV_REDIRECT | SUSTAIN (Op 15) + SUBSTITUTE (Stage 0) |
| FUSION, FUSION_KERNELS, FLOW_* | SUBSTITUTE (Stage 0) + ORACLE (Stage 0) |
| WEIGHT_SHARE, WEIGHT_COMPRESS | SUBSTITUTE (Stage 0) — recipe selection |
| ATTN_KOOPMAN | ADAPT (Stage 2) + SUBSTITUTE |
| GRAPH, GRAPH_INSPECT | ORACLE + CLASSIFY |
| OVERLAP | (NCCL family, deferred Class D) |
| MARLIN, RECIPES, SUBSTITUTE_V2 | SUBSTITUTE (Stage 0) |
| THERMAL_FEEDBACK | THERMOSTAT (Op 20) |
| POWER_CAP | VOLT (Op 30) + HIBERNATE (Op 31) |
| PARAM_RECOVERY, VMM | INFRA (substrate, not workload-facing) |
| FP8_COMPUTE | SUBSTITUTE (recipe path, env-gated) |
| GENERATE (audit term) | SUBSTITUTE (NVRTC compile path) |
| ROUTE (audit term) | ARBITRATE + ORACLE |
| PARTITION (audit term) | ARBITRATE |

## Applicability scheme

- **A** = applies strongly; expected to lift MFU on this workload
- **s** = applies somewhat; may lift, secondary contributor
- **.** = does not apply (no-op or zero impact)

## The 33 canonical ops

### Stage 0 — Critical path (~12 ns total budget; runs every kernel launch)
1. **CLASSIFY** — kernel signature → kernel_class enum
2. **ORACLE** — recipe lookup by kernel_class + params_hash
3. **SUBSTITUTE** — kernel substitution (Marlin INT4 / FP8 / fused / cuBLAS passthrough); subsumes FUSION, FUSION_KERNELS, RECIPES, SUBSTITUTE_V2, FP8_COMPUTE, WEIGHT_COMPRESS, MARLIN, FLOW_SUBSTITUTE
4. **COMMIT** — final dispatch to the chosen kernel
5. **SAMPLE** — emit ring entry (timestamp, params_hash, kernel_class)
6. **RING_WRITE** — SPMC ring publication via `release` store

### Stage 1 — Shadow thread (observational, off critical path)
7. **REMEMBER** — CfC hidden-state update from ring observations
8. **VALIDATE** — Welford online stats; 3-σ anomaly detection
9. **AUDIT** — HMAC-SHA256 tamper-evident chain
10. **SPECULATE** — CfC prediction → look-aside buffer for next kernel class

### Stage 2 — Background thread (Koopman + arbitration)
11. **ADAPT** — EDMD snapshots + Koopman matrix solve + LNN weight swap; subsumes ATTN_KOOPMAN
12. **ARBITRATE** — Green Context SM partition rebalance; subsumes PARTITION_ROUTER, SM_PACKER, GREEN_CTX

### 21 overlay ops (numbered 13–31 + STRAGGLER + NCCL_P2P)
13. **SENSE** (Op 13) — session classification HUMAN/AGENT/BATCH
14. **SHIELD** (Op 14) — latency-budget protection + band priority hints
15. **SUSTAIN** (Op 15) — KV pressure slope detection; subsumes KV_COMPRESS, KV_REDIRECT
16. **GUARD** (Op 16) — KV cache privacy / leak detection
17. **PREDICT** (Op 17) — proactive L2 preload candidates; subsumes L2_PERSIST, PERSIST_ENGINE, MEM_LAYOUT
18. **RECEIPT** (Op 18) — per-session signed proof of compute
19. **CONTINUITY** (Op 19) — incremental KV state checkpoint
20. **THERMOSTAT** (Op 20) — predictive thermal throttle prevention; subsumes THERMAL_FEEDBACK
21. **DETERMINISM** (Op 21) — reproducible dispatch fingerprint
22. **PULSE** (Op 22) — hardware fault early warning
23. **CARBON** (Op 23) — per-session carbon estimate
24. **FAIRNESS** (Op 24) — per-tenant work quota (incl. FAIRNESS_SHM cross-process)
25. **TOPOLOGY** (Op 25) — NVLink/PCIe peer adjacency
26. **LOOP** (Op 26) — agentic runaway detection
27. **PIPELINE** (Op 27) — multi-agent session correlation
28. **TRACE** (Op 28) — bounded kernel-trace exporter
29. **COMPLY** (Op 29) — compliance artifact bundler
30. **VOLT** (Op 30) — SM frequency steering via NVML; subsumes POWER_CAP partially
31. **HIBERNATE** (Op 31) — idle SM power gating; subsumes POWER_CAP partially
32. **STRAGGLER** — local slowdown + algo hint
33. **NCCL_P2P** — multi-node NCCL proxy (Class D deferred to Phase 5+)

## The 33×24 matrix

Columns are WL01–WL24 per `PHASE_4_ARCHITECTURE.md` taxonomy.

```
                  WL01 WL02 WL03 WL04 WL05 WL06 WL07 WL08 WL09 WL10 WL11 WL12 WL13 WL14 WL15 WL16 WL17 WL18 WL19 WL20 WL21 WL22 WL23 WL24
                  ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
01 CLASSIFY        A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
02 ORACLE          A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
03 SUBSTITUTE      s    s    A    A    A    A    A    A    A    A    s    A    A    A    A    A    A    s    A    A    A    A    A    A
04 COMMIT          A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
05 SAMPLE          A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
06 RING_WRITE      A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
07 REMEMBER        s    s    s    A    A    s    A    s    s    s    A    A    A    A    A    A    A    s    s    s    A    A    A    s
08 VALIDATE        A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
09 AUDIT           A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
10 SPECULATE       A    A    A    A    s    s    s    s    s    A    s    A    A    A    s    A    s    s    s    s    A    s    s    s
11 ADAPT           .    .    s    A    A    .    A    s    s    s    A    A    s    A    s    A    A    s    s    s    s    s    A    s
12 ARBITRATE       s    s    s    A    A    .    s    A    A    s    A    s    s    s    A    s    A    A    s    A    s    A    A    s
13 SENSE           s    s    s    A    A    s    s    s    s    s    A    A    s    s    s    s    s    s    s    s    A    A    s    s
14 SHIELD          A    A    s    A    A    s    .    s    s    A    A    .    A    s    A    s    .    .    s    s    A    A    s    s
15 SUSTAIN         A    A    A    A    A    .    .    .    s    A    A    s    A    s    A    A    .    .    .    A    A    A    s    .
16 GUARD           s    s    s    A    A    .    s    .    .    s    A    .    A    .    s    A    s    .    .    s    A    A    s    .
17 PREDICT         s    s    A    A    A    A    A    A    s    s    s    A    s    A    s    A    A    s    A    A    s    A    s    A
18 RECEIPT         A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
19 CONTINUITY      s    s    s    A    A    .    A    s    .    s    A    s    s    s    s    A    A    s    .    s    s    A    A    s
20 THERMOSTAT      s    s    A    A    A    s    A    A    s    s    s    A    s    s    A    s    A    s    A    A    s    s    s    s
21 DETERMINISM     A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
22 PULSE           A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
23 CARBON          A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
24 FAIRNESS        s    s    s    A    A    .    s    .    .    .    A    A    .    .    s    .    s    A    .    .    .    s    s    .
25 TOPOLOGY        .    .    .    .    s    .    .    .    .    .    .    .    .    .    .    .    .    A    .    .    .    .    .    .
26 LOOP            .    .    .    s    s    .    .    .    .    s    A    .    s    .    .    .    .    .    .    .    A    s    .    .
27 PIPELINE        .    .    .    s    A    .    .    .    .    .    A    .    .    .    .    .    .    s    .    s    .    A    s    .
28 TRACE           A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
29 COMPLY          A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A    A
30 VOLT            s    s    A    A    A    s    A    A    s    s    s    A    s    A    A    s    A    s    A    A    s    s    s    s
31 HIBERNATE       A    A    s    A    A    s    s    s    s    A    A    .    s    s    s    A    .    s    s    s    A    A    A    s
32 STRAGGLER       .    .    .    s    s    .    s    .    .    .    .    s    .    .    s    .    A    A    .    .    .    .    .    .
33 NCCL_P2P        .    .    .    .    .    .    s    .    .    .    .    .    .    .    s    .    s    A    .    .    .    .    .    .
```

## Per-op justification

### Stage 0 — universal (rows 01-06)

- **CLASSIFY, ORACLE, COMMIT, SAMPLE, RING_WRITE** — Stage 0 critical path runs every kernel launch; structural to the runtime. `A` on every workload.
- **SUBSTITUTE** — strong on workloads with substitute headroom (prefill GEMMs, conv, quant). `A` on WL03, WL05–WL24 broadly. Weaker on tight decode (WL01, WL02) where the dominant kernel is memory-bound matvec with limited substitute options — `s`. Weaker on WL11 (agentic) where dispatch is heterogeneous; `s`. Weaker on WL18 (multi-GPU TP) where NCCL kernels dominate; `s`.

### Stage 1 — observational (rows 07-10)

- **REMEMBER** — CfC hidden state benefits from long-horizon observations. `A` on continuous/predictable workloads (vLLM serving WL04, multi-tenant WL05, training WL17, batch WL12). `s` on short bursty workloads.
- **VALIDATE, AUDIT** — observer + integrity. `A` on every WL (compliance posture).
- **SPECULATE** — strong where dispatch patterns are predictable: decode loops (WL01–WL04), speculative decoding itself (WL10), torch.compile (WL14), prefix caching (WL16). `s` on bursty/diverse (MoE WL15, model switch WL23).

### Stage 2 — arbitration (rows 11-12)

- **ADAPT** — Koopman weight swap. Needs runtime long enough for EDMD to converge. `A` on long-running training (WL07, WL17), serving (WL04), multi-tenant (WL05), agentic (WL11). `.` on short-lived (WL01 decode single-stream, WL06 embeddings, WL19 CLIP).
- **ARBITRATE** — SM partition rebalance via Green Context. `A` on multi-tenant (WL05), multi-stage (WL08, WL09, WL20), MoE expert routing (WL15), training (WL17), multi-GPU (WL18), multi-agent (WL11), RAG pipeline (WL22), model switch (WL23). `s` elsewhere. Note: ARBITRATE includes the partition_router + sm_packer + green_ctx aspects.

### Overlay — session intelligence (rows 13-15)

- **SENSE** — `A` on sessions that benefit from band classification: agentic (WL11), batch (WL12), multi-tenant (WL05), serving (WL04), code (WL21), RAG (WL22). `s` elsewhere.
- **SHIELD** — latency-sensitive workloads: decode B=1 (WL01), B=8 (WL02), serving (WL04), multi-tenant (WL05), speculative (WL10), agentic (WL11), long context (WL13), MoE (WL15), code (WL21), RAG (WL22). `s` mid-priority. `.` on training/batch where latency budget is loose.
- **SUSTAIN** — KV pressure detection. `A` on KV-heavy workloads (WL01-WL05, WL10, WL11, WL13, WL15, WL16, WL20, WL21, WL22). `.` on non-KV (WL07 LoRA, WL08 diffusion, WL17 training, WL18 multi-GPU, WL19 CLIP, WL24 quant native).

### Overlay — compliance/observability (rows 16, 18, 21, 22, 23, 28, 29)

- **GUARD** — multi-tenant privacy enforcement. `A` on WL04 (serving), WL05 (multi-tenant), WL11 (agentic), WL13 (long context), WL16 (prefix cache), WL21 (code gen), WL22 (RAG).
- **RECEIPT, DETERMINISM, PULSE, TRACE, COMPLY, CARBON** — universal observers, always on. `A` on every WL.

### Overlay — L2 / memory (row 17)

- **PREDICT** — proactive L2 preload (includes L2_PERSIST, PERSIST_ENGINE). `A` on workloads with predictable hot regions: prefill (WL03), serving (WL04), multi-tenant (WL05), embeddings (WL06), LoRA (WL07), diffusion (WL08), batch (WL12), compile (WL14), prefix cache (WL16) **strongly**, training (WL17), CLIP (WL19), LLaVA (WL20), RAG (WL22), AWQ (WL24).

### Overlay — DVFS / thermal (rows 20, 30, 31)

- **THERMOSTAT** — `A` on long-running compute-bound (WL03, WL04, WL05, WL07, WL08, WL12, WL15, WL17, WL19, WL20). `s` elsewhere.
- **VOLT** — DVFS steering. `A` on compute-bound long-runs (WL03 prefill, WL04, WL05, WL07, WL08, WL12, WL14, WL15, WL17, WL19, WL20). `s` elsewhere. Pod-degraded on Lambda.
- **HIBERNATE** — idle SM gating. `A` on bursty patterns with idle gaps: decode (WL01, WL02), serving (WL04), multi-tenant (WL05), speculative (WL10), agentic (WL11), prefix cache (WL16), code gen (WL21), RAG (WL22), model switch (WL23). `.` on continuous compute (WL12, WL17 training).

### Overlay — multi-tenant (rows 19, 24)

- **CONTINUITY** — stateful KV checkpoint. `A` on serving (WL04), multi-tenant (WL05), LoRA (WL07), agentic (WL11), prefix cache (WL16), training (WL17), RAG (WL22), model switch (WL23).
- **FAIRNESS** — `A` on multi-tenant (WL05), serving (WL04), agentic (WL11), batch (WL12), multi-GPU TP (WL18). `s` on workloads with implicit quota (LoRA, batch processing).

### Overlay — agentic (rows 26, 27)

- **LOOP** — agentic runaway. `A` on agentic (WL11), code gen (WL21). `s` on serving with potentially-runaway prompts.
- **PIPELINE** — multi-agent correlation. `A` on agentic (WL11), multi-tenant (WL05), RAG (WL22). `s` on multi-stage pipelines.

### Overlay — multi-GPU (rows 25, 32, 33)

- **TOPOLOGY** — `A` only on WL18 multi-GPU TP. `s` on WL05 (multi-tenant could benefit from NVLink-aware partition; single H100 = trivial). `.` elsewhere.
- **STRAGGLER** — local slowdown signal. `A` on multi-GPU (WL18), training (WL17 may distribute). `s` on multi-tenant where one tenant's slowdown shouldn't drag others.
- **NCCL_P2P** — deferred Class D. `A` on multi-GPU (WL18). `s` on training (WL17) and MoE (WL15) which use NCCL all-to-all.

## Cluster declared-workloads-set (derived from matrix)

For each P4.x cluster, the **declared-workloads-set** = union of WLs with `A` or `s` for any op in the cluster.

| Cluster | Ops (cluster member subset) | Declared-workloads-set | Complement |
|---|---|---|---|
| **P4.2 SM partition** | ARBITRATE (subsumes PARTITION_ROUTER, SM_PACKER, GREEN_CTX) | WL01, WL02, WL03, WL04, **WL05**, WL07, WL08, WL09, WL10, WL11, WL12, WL13, WL14, WL15, WL16, WL17, WL18, WL19, WL20, WL21, WL22, WL23, WL24 (23 of 24) | WL06 only |
| **P4.3 DVFS/thermal** | VOLT, HIBERNATE, THERMOSTAT, PULSE, CARBON, SUSTAIN | all 24 (PULSE+CARBON are universal observers) | — |
| **P4.4 L2 cluster** | PREDICT (subsumes L2_PERSIST, PERSIST_ENGINE, MEM_LAYOUT), SUBSTITUTE | WL03, WL04, WL05, WL06, WL07, WL08, WL09, WL12, WL14, WL16, WL17, WL18, WL19, WL20, WL21, WL22, WL24 (17 WLs, all where PREDICT is A or s — see matrix) | WL01, WL02, WL10, WL11, WL13, WL15, WL23 |
| **P4.5 Weight cluster** | SUBSTITUTE (subsumes WEIGHT_SHARE, WEIGHT_COMPRESS, RECIPES, MARLIN) | WL03–WL24 (broad) | WL01, WL02 (decode B=1/B=8 — limited substitute headroom; `s` only) |
| **P4.6 KV/attention cluster** | SUSTAIN, SHIELD, GUARD, CONTINUITY, ADAPT (subsumes ATTN_KOOPMAN) | WL01-WL05, WL10, WL11, WL13, WL15, WL16, WL20, WL21, WL22, WL23 (KV-heavy) + WL14 (graph capture context) | WL06, WL08, WL09, WL17, WL18, WL19, WL24 |
| **P4.7 Fusion+agentic+compliance** | SENSE, SHIELD, GUARD, LOOP, PIPELINE, CONTINUITY, FAIRNESS, STRAGGLER, TRACE, AUDIT, DETERMINISM, RECEIPT, COMPLY, CARBON | all 24 (universal observers + selective enforcement) | — |

## Validation rules

1. **Each cluster's declared-workloads-set must be measured** to lift MFU at the cluster checkpoint.
2. **Complement set must show no MFU regression** beyond ±5% noise.
3. The matrix is **binding**: cluster scope cannot be expanded mid-implementation to claim a workload not in the declared set.
4. Adding a new op (post-T4.0.8) requires updating the matrix and obtaining explicit approval before that op's measurements count toward gating.

## Surprises / observations

1. **Universal-observer ops (CLASSIFY, ORACLE, COMMIT, SAMPLE, RING_WRITE, VALIDATE, AUDIT, RECEIPT, DETERMINISM, PULSE, TRACE, COMPLY, CARBON) cover all 24 workloads.** These are the always-on Stage 0 critical path + compliance posture. Their cluster's checkpoint complement is empty → they cannot regress any workload.
2. **WL06 Embeddings is the lone complement of P4.2 (SM partition)** — embeddings on MiniLM-L6-v2 is so small the SM partition decision is meaningless. SM partition cluster verification must explicitly test that WL06 is not regressed.
3. **WL01/WL02 decode B=1/B=8 are the lone complement of P4.5 (Weight cluster)** — decode dominant kernel is matvec with limited substitute paths. P4.5 cluster verification must confirm decode is not regressed.
4. **WL18 multi-GPU TP is mostly NCCL-deferred** — `A` for NCCL_P2P (Class D), STRAGGLER, TOPOLOGY; `s` for many others. P4 measurement on WL18 is limited; this workload partially gates on Phase 5.
5. **TOPOLOGY (Op 25) is essentially a no-op on single-H100 pod** — `A` only on WL18, `s` on WL05 (multi-tenant could benefit from NVLink-aware partition but single H100 = trivial). On this pod, TOPOLOGY measurement is degenerate.
6. **NCCL_P2P (Op 33) is Class D deferred** — its row in the matrix predicts what becomes interesting in Phase 5+, not Phase 4.

## End-of-T4.0.8 preconditions check

| Check | Result |
|---|---|
| `cipher_kmod.ko.v0.2.0` md5 | `55ab8c0cd8309ca7cc0fc40fe556aa19` ✓ unchanged |
| `libcipher_v2.so.v0.2.0` md5 | `86618c30896470b642fcc6985d8dc632` ✓ unchanged |
| cipher_kmod loaded | 0.3.1, srcversion `B1AF5E2A...` ✓ |
| Taint | 12288 ✓ |
| /proc/cipher | all 3 readable ✓ |

T4.0.8 complete. Proceeding to T4.1.1 (cipher_kmod 0.4.0 baseline).
