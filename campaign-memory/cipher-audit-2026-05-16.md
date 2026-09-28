---
name: cipher-audit-2026-05-16
description: Code-reality audit ground truth — 33 ops are in may13 build NOT production runtime; Phase 4 not closed
metadata: 
  node_type: memory
  type: project
  originSessionId: 8482a673-2517-464b-974c-6170c963b21b
---

Code-reality audit 2026-05-16 — `AUDIT_REPORT_2026_05_16.md` (top-level),
detail in `cipher-fusion-evidence/audit_section_{1a,1b,2,3,4}.md`.

**Load-bearing finding:** the 33 canonical ops (12 core + 21 overlay) live in
`cipher-may13-evidence/src/` (the `cipher_*.cpp` legacy build). **Zero of
them are in the production Phase 4 runtime `libcipher_rt.so` `c2c5d313`**
(verified `nm -D`). `cipher_rt_phase4/` is a separate, smaller codebase
(`cipher_rt_*` family) that re-implements only a subset: ARBITRATE, VOLT,
SUBSTITUTE (Marlin/cuBLAS), AUDIT, + the new attn substrate. The fusion
campaign's per-CP job is migrating canonical ops into the Phase 4 runtime.

**Op surface (may13 build):** 19 WORKING, 14 PARTIAL, 0 true STUBs. The
"15 stubs" memory framing was wrong — the 14 PARTIAL ops have real logic;
they are detectors not wired to actuators, not empty stubs.

**Phase 4 is IN-FLIGHT, not closed.** `t4_6_4_report.md` says "Phase 4
closes" but `PHASE_4_ARCHITECTURE.md` requires P4.7 (fusion+agentic) and P4.8
(integration + G4 24h soak) — both have zero work product. Honest status:
4.3/4.5 shipped (regime caveats), 4.6 partial (T4.6.1-4 done, 5-6 not),
4.4 partial, 4.7/4.8 not started.

**Goal 4 (O(1) Koopman substitution):** fires ZERO times across all stress
logs. Four disconnected substitution substrates, each gated to passthrough;
`cipher_koopman_runtime.cpp` is fully built with zero callers. Closing it is
architectural/multi-week, not a single TODO.

**Anchors:** sound. One action queued — re-anchor libcipher_v2 `cc0479b8`
(un-anchored Phase-3 CUPTI successor to `86618c30`); see [[cipher-cp25-closed]].
