# T4.6.5 — Decision Artifact

**Date:** 2026-05-16. The pre-specified decision tree (memo §2 / §4) applied
to the measured cross-tenant dedup `m`. Mechanical — no judgement.

## Measured `m` (`dedup_measurement.json`)

739 Claude Code agentic traces (`callanjfox/kv-cache-tester`),
`hash_id_scope: "local"`, structural canonical-shared-prefix model, n=5
windowed.

| resident context | measured `m` | 95% CI | tier (tree) | predicted (memo §2) |
|---|---|---|---|---|
| 25K | **2.35×** | [2.345, 2.353] | **L3** (2–10×) | 2.5× |
| 32K | **1.81×** | [1.807, 1.810] | **L1-only** (<2×) | 1.9× |
| 137K | **1.12×** | [1.115, 1.115] | **L1-only** (<2×) | 1.1× |

Every measured point lands **within the pre-registered prediction band** —
the engineering-marvel "validate within bounds" holds; no reframe.

## Tier crossover

`m = 1/(1−f)`, `f = prefix/C`, mean prefix ≈ 14.3K. `m = 2×` at
`C = 2·prefix ≈ 28.6K`. **The L3/L1 boundary is at ~28.6K resident
context** — below it the substrate clears the L3 bar, above it it does not.

## Decision tree outcome

| Tree branch | Selected at |
|---|---|
| `m < 2×` → **L1-only, 20–30 tenants/GPU** | 32K and 137K |
| `2× ≤ m ≤ 10×` → L3 (already built), 50–100 tenants | 25K only |
| `m > 10×` → RadixAttention | never |

**The §3 gate regime is 32K** (design memo D2d, adjudicated). At 32K the tree
selects **L1-only → 20–30 tenants**. The agentic steady state (median
last-request context 137K) is **firmly L1-only** (`m = 1.12×`).

## What T4.6.6 runs (per D5 — decision-tree-selected count)

- **Operating point:** Mistral-7B fp16, **32K** context, cuIpc shared weights.
- **Tenant target:** §3a fp16 capacity = `16·m` tenants ⇒ `16 × 1.81 ≈ 29`.
  **T4.6.6 targets 30 tenants** (round), gate at sustained operation + 0
  isolation violations + p99 within roofline. NOT a fixed 50 — the tree
  selected this.
- **Stretch:** INT4 weights (§3a: `19·m ≈ 34`) and/or 25K context (L3 regime,
  `16·2.35 ≈ 38`) as reported stretch arms.

## Honest caveats (carried into the report)

1. **Lower bound.** `hash_id_scope: "local"` ⇒ this measures only the
   system+tool-prefix component of cross-tenant dedup. Codebase / shared-file
   overlap (agentic coders on the same repo) is real but unmeasurable from
   this dataset. Production `m` at 32K is plausibly higher than 1.81×.
2. **Substrate-achievement validation deferred to T4.6.6.** This artifact
   measures the dedup *rate* (a workload property, model bounded by trace
   characteristics). Whether the kmod substrate *achieves* the modelled `m`
   at scale is measured directly in T4.6.6's dedup-at-scale instrumentation
   (item 6) — that is the real==sim check for this CP.
3. **Tier is context-regime-dependent**, not a single verdict: L3 below
   ~28.6K, L1-only above. The report leads with the curve, not a scalar.

**Decision:** T4.6.6 proceeds at **L1-only / 30-tenant** target, 32K, fp16.
Phase 4.6 closes on the honest decision-tree outcome, not a pre-set 50.
