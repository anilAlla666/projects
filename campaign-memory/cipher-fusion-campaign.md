---
name: cipher-fusion-campaign
description: "The CIPHER \"absolute-success build-out\" — close all 23 canonical Phase 0-4 CPs, one atomic STEP per CP, design-memo-then-approve discipline"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

The user launched an "ABSOLUTE-SUCCESS BUILD-OUT" (2026-05-15): close every
canonical Phase 0-4 checkpoint (the audit found 6/23 shipped), no deferrals.

**Why:** the phase audit exposed hard drift — 5/23 CPs shipped, the rest
partial/not-done/superseded. The campaign re-runs every CP to clean SHIPPED.

**How to apply — the working discipline (strict):**
- One CP at a time. One atomic STEP per CP. One report per CP. **WAIT for the
  user's explicit adjudication between every CP** — never start the next CP
  without "adjudicated — proceed to CP X".
- Adjudication is against the on-disk artifact, not partial/running data. A CP
  is not closed until its artifact exists on disk AND the user has read it.
- For hard CPs: audit-before-build → **design memo → show → wait for approval
  → no code until approved** → build atomically → gate → one report.
- Evidence under `cipher-fusion-evidence/cp_<x>_<y>/`. Report artifacts with
  paths AND md5s.
- Honest accounting: if a measurement misses the gate, report it and rebuild —
  never reframe. Re-audit-then-build before declaring anything structurally
  impossible (code-level evidence required).
- CP order: 0.4, 0.5, 0.6, 3.3, 3.4, 2.1, 2.4, 2.5, 4.1, 4.2, 4.3, then the
  PARTIAL CPs (0.2, 1.3, 1.5, 2.3, 3.1, 3.2), then composed measurements.

**Binding targets the campaign measures against:** MFU > 85% on real
workloads; 3x tok/W vs unaccelerated; 100 concurrent tenants on one H100;
LD_PRELOAD removed at v1.0 (CUDA_INJECTION64_PATH only). Claim only if measured.

**Anchors frozen:** kmod `55ab8c0c`, libcipher_v2 `86618c30`. T4.6.4 kmod
`b263ad30…` was the working baseline at campaign start; kmod rebuilds record a
fresh md5, anchors unmoved.

**Status (2026-05-15):** Phase 0 closed — CP 0.4/0.5/0.6 SHIPPED (6/23 → 9/23).
Headline Phase-0 finding: one architectural ceiling (~8.9 tok/s aggregate,
peak MFU 0.0185%) in both continuous and burst workloads — the eager-mode
prototype harness, not silicon. CP 3.3 SHIPPED + adjudicated (→10/23): continuous
hardware FLOP counting + per-tenant MFU, kmod 0.4.7 `2a69f9de…`. CP 3.4 SHIPPED +
adjudicated (→11/23): Grafana + ClickHouse first-light dashboard, per-second
per-tenant silicon state — pure observability, zero kmod/ABI change. CP 2.1
SHIPPED + adjudicated (→12/23): libcipher_hook port/keep/delete inventory
(+ addendum: NCCL row PORT→DELETE). **CP 2.4 CLOSED + adjudicated (→14/23)** (2026-05-16) — `CP_2_4_REPORT.md`
md5 `573c50d6`. Composed gate (Mistral-7B B=1, n=5): **all-on/vanilla 3.617×
tok/W [3.591, 3.642]**, 1.795× tok/s; clean multiplicative composition.
Clears the old 2.96× scorecard but with a different lever mix — Marlin under
at 1.31× tok/W (B=1 weak regime); DVFS dominant at +62% (null-validated).
Gate (a)(b)(c)(e) PASS; (d) NOT MET — Llama model-draft 1.75× deferred to
Phase 5. Detail [[cipher-cp24-closed]].

**CP 2.5 IN FLIGHT** — Phase 2 close (14/23 → 15/23). Scope: drop LD_PRELOAD
for CUDA_INJECTION64_PATH-only deployment; ship `cipher-platform.deb` for
clean neocloud-node install. Self-contained, ~3–5 days, the deployment story
for the May 28 Ditlev demo. Atomic STEP, design-memo-first. Progress in
`cipher-fusion-evidence/cp_2_5/`. Anchors held: kmod 0.4.8 `e2f50452`,
libcipher_v2 `86618c30`, libcipher_rt `5e304549`, taint 12288.

**Audience note:** "Ditlev" is a stakeholder who reads the CIPHER brief. The
differentiator he is expected to recognise is **real-time per-tenant MFU as a
free byproduct of existing telemetry** — foreground that in CP reports/briefs.
