# Phase 4 / Fusion-Campaign P3 Audit — Which Prior CP Measurements Engaged the F1 Path

**Date:** 2026-05-18. **Type:** paperwork audit, no GPU. CP 5.6 close-out
Task 1. Finding F1 = the Marlin full-GPU GEMM, pinned to the primary context
by `PrimaryCtxGuard`, races a consumer on another context → degenerate decode
([[cipher-f1-fullgpu-marlin-broken]]).

## F1-affected substrate window

`PrimaryCtxGuard` ("Marlin Fix A") entered with libcipher_rt **`5e304549`**
(CP 2.4). The full-GPU path stayed F1-broken through **`c2c5d313`** (CP 2.5
anchor) and **`dc804eb3`** (CP 5.3 STEP 2 — fixed only the *partitioned/green*
path; full-GPU still broken). It was fixed in **`a7ac8e97`** (CP 5.6 P1).

**A measurement is F1-suspect iff:** (a) substrate ∈ {`5e304549`, `c2c5d313`,
`dc804eb3`}, AND (b) it engaged the Marlin INT4 full-GPU GEMM path, AND
(c) it had no output-correctness gate. Pre-`5e304549` substrates have no
`PrimaryCtxGuard` → no F1. fp16-only decode does not engage Marlin INT4 → no F1.

## Classification

Three buckets per the close-out spec: **VALID** (F1 does not affect the claim,
or a correctness gate covers it), **RETRACTED** (F1 failed it), **INCOMPARABLE**
(different/experimental methodology — not retroactively classifiable).

| CP | substrate | Marlin full-GPU? | gate? | verdict |
|---|---|---|---|---|
| CP 0.4 / 0.5 / 0.6 (Phase 0) | pre-`5e304549` | yes (early) | varies | **VALID** w.r.t. F1 — pre-Fix-A, `PrimaryCtxGuard` did not exist |
| CP 2.1 | pre-`5e304549` | minimal | — | **VALID** w.r.t. F1 — pre-Fix-A |
| **CP 2.4** (composed gate) | `5e304549` | **yes** + green ctx | **none** | **RETRACTED** — 3.617× tok/W / 1.795× tok/s; F1-loop artifact (F1_TPW_RECORDS_REVIEW, CP 5.6 P2) |
| **CP 2.5** (LD_PRELOAD-free deploy) | `c2c5d313` | gate (b) re-ran composed gate | none on (b) | **SPLIT** — deployment mechanism + `.deb` **VALID**; gate-(b) **3.602× tok/W RETRACTED** (same F1-broken composed gate, injection-only) |
| CP 3.3 / 3.4 | n/a | no | — | **VALID** — not Marlin/throughput CPs; F1 N/A |
| CP 4.4 (L2 weight pin) | `c2c5d313` | no (closed at memo) | analytical bound | **VALID** — closed via an L2-capacity bound; no F1-affected measurement is load-bearing |
| CP 4.6.5+6 (KV-dedup) | `c2c5d313` | **no** — decode is **Llama-3.1-8B fp16** | substrate-scale + capacity | **VALID** — dedup ratio (1.866×) + 13-tenant ceiling are structural/capacity; fp16 decode never engages Marlin INT4 |
| CP 4.7 (kernel fusion) | `c2c5d313` | no (closed at memo) | hook-coverage null check | **VALID** — closed via the 1.1%-hook-reachable bound; the confirming measurement is a coverage null, not a Marlin tok/W gate |
| **CP 4.8 Task B** (CP 2.4 re-measure) | `c2c5d313` | **yes** | **none** | **RETRACTED** — 3.6166× "reproduction"; reproduced the F1-broken composed gate (F1_TPW_RECORDS_REVIEW: "Task B's conclusion is itself now suspect") |
| CP 5.1 (vLLM KV integration) | `c2c5d313` | no — KV-cache path | KV token-agreement gate, PASS | **VALID** w.r.t. F1 — a KV-correctness claim, not a Marlin-throughput claim; F1 N/A (gate methodology is a separate question, out of this audit's scope) |
| CP 5.2 (KV offload) | `c2c5d313` | no — KV-cache path | correctness gate, PASS | **VALID** w.r.t. F1 — as CP 5.1 |
| CP 5.3 STEP 1 (partition-grid sweep) | `c2c5d313` | yes (grid sweep) | per-grid reference target; **CP not closed on it** | **INCOMPARABLE** — experimental side-build, CP 5.3 explicitly not closed on STEP 1; had a per-grid correctness reference but is not a closed claim |
| CP 5.3 STEP 2 | `dc804eb3` | partitioned path (fixed in dc804eb3) | A-num oracle, PASS | **VALID** — the CP that *surfaced* F1; A-tok degeneracy correctly classified invalid, A-num is the binding oracle; conclusion accounts for F1 |
| CP 5.6 (P1 fix, P2 re-measure) | `a7ac8e97` | yes — fixed path | teacher-forced KL ≤ 0.1 | **VALID** — the F1 fix + honest re-measurement |

## The headline retraction

The composed-gate **≈3.6× tok/W** was measured **three times**, and all three
are the same F1-broken composed gate (Marlin full-GPU, green ctx active,
**no output-correctness gate** — `spec_varied_driver.py` recorded only timing):

- CP 2.4 — **3.617×** [3.591, 3.642] (`5e304549`)
- CP 2.5 gate (b) — **3.602×** [3.529, 3.676] (`c2c5d313`, injection-only)
- CP 4.8 Task B — **3.6166×** (`c2c5d313`)

All three **RETRACTED**. The "reproductions" reproduced the blind spot, not a
result. Honest replacements, F1-fixed and correctness-gated: single-tenant
Mistral-7B B=1 **1.54×** tok/W (CP 5.6 P2); verified static cross-tenant
batched **3.69×** (N=8 TinyLlama) / **3.26×** (N=4 Mistral) substrate-
attributable (Phase B Session 2 + Mistral verification).

## What stands

Everything else is **VALID w.r.t. F1** or **INCOMPARABLE**. F1 is a
Marlin-INT4-full-GPU-decode bug; it does **not** reach: pre-Fix-A CPs (no
`PrimaryCtxGuard`); CPs closed analytically at memo (4.4, 4.7); the KV-cache
CPs (4.6, 5.1, 5.2 — fp16/KV paths, not Marlin INT4); or non-throughput CPs
(3.3, 3.4). CP 5.3 STEP 2 caught F1 and is sound; STEP 1 is an unclosed
experimental side-build (incomparable).

**Net:** exactly one measurement is retracted — the composed ≈3.6× tok/W
headline — in the three places it was recorded. No other closed CP claim
falls to F1. The campaign's structural, capacity, KV-correctness, and
analytically-bounded results are unaffected.

## Confidence / provenance

High confidence on the CP 2.4 / 2.5 / 4.8 retractions — grounded in
`F1_TPW_RECORDS_REVIEW.md`, `CP_5_6_P2_REPORT.md`, and the CP 2.5 / 4.8 report
text (substrate anchors and gate (b) verified by direct read). The VALID
classifications rest on each CP's substrate anchor + path (fp16 vs Marlin INT4
vs KV) as stated in its report. CP 5.3 STEP 1 read confirms "experimental
side-build, CP not closed." Any CP whose full methodology (beyond F1) warrants
review is out of this audit's scope — this audit certifies F1 exposure only.
