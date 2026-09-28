# Phase 5 — CP 5.2: KV offload hierarchy — DESIGN MEMO

**Date:** 2026-05-17. **Status:** Phase 5 CP. Not started — scoped for
adjudication.

---

## §1 — Scope / what CP 5.2 ships

CP 5.2 wires the Phase 4.6 KV substrate primitive into a **live decode
engine's KV cache** and adds a capacity-extending offload hierarchy. It ships:

- **Live-KV wiring** — the cross-process KV-dedup primitive (`cp_4_6_5_6` —
  validated 100-proc, cuIpc cross-process GPU memory confirmed on this pod)
  connected to a real decode engine's paged KV cache, so dedup operates on
  *live* attention KV blocks, not synthetic ones.
- **KV offload tiering** — a KV-block residency hierarchy: HBM (hot) → host
  DRAM (warm) → optionally NVMe (cold), with eviction and prefetch policy, so
  effective KV capacity exceeds the HBM budget.
- **Shared-prefix dedup on real traffic** — measured dedup hit-rate where
  tenants share system prompts / few-shot prefixes (the realistic
  multi-tenant pattern).

## §2 — Why this CP / dependencies

**Dependency in.** Phase 4.6 (`cp_4_6_5_6`) closed the KV-dedup *primitive*:
attention-routing substrate shipped, cuIpc cross-process GPU memory verified,
100-proc synthetic dedup scales. The standing finding was explicit —
**live-KV wiring is Phase 5 work**, and the dedup moat lives at **32K+
context** where KV cache dominates HBM. CP 5.2 is that wiring.

**Dependency out.** Feeds the CP 5.5 100-tenant integration measurement; the
KV-capacity headroom is what makes high tenant density at long context
economically real. Composes with CP 5.1 (the live decode engine CP 5.2 wires
into is the vLLM/TGI engine CP 5.1 integrates).

## §3 — Approach

Hook the decode engine's KV-block allocator (vLLM paged-attention block
manager is the reference target) so KV blocks route through the CIPHER
attention substrate. Dedup identical blocks cross-process via the verified
cuIpc path. Add a tiering manager that demotes cold blocks to host DRAM and
prefetches on cache-miss, with the eviction policy tuned to keep decode-step
latency overhead bounded. Validate on long-context (32K+) multi-tenant
traffic with shared prefixes — the regime where the lever pays.

## §4 — Gate criteria

CP 5.2 PASSES iff:

1. Live KV blocks from a real decode engine route through the substrate with
   output correctness preserved (byte-identical / numerically-equivalent
   substrate-on vs off).
2. A measurable **KV-capacity extension** — effective context length, or
   tenant count at fixed context — beyond the HBM-only budget, with
   per-decode-step latency overhead within a pre-registered bound.
3. Dedup hit-rate measured on realistic shared-prefix traffic at 32K+
   context.

The pre-registered latency bound is set in the CP 5.2 build memo before
measurement (Phase 4 pre-registration discipline).

## §5 — Calendar

**4–6 weeks.** Spread driven by the tiering policy: a straightforward
HBM↔DRAM two-tier with a simple LRU eviction is the lower bound; if NVMe cold
tier and prefetch tuning are needed to hit the capacity target, the upper.

## §6 — Risks / known-unknowns

- **Offload bandwidth vs decode latency.** Host-DRAM KV demotion crosses
  PCIe; if prefetch cannot hide the transfer, decode-step latency regresses.
  This is the central risk — measure offload bandwidth headroom early.
- **Eviction thrash** under bursty multi-tenant traffic — a poor policy can
  oscillate blocks across tiers.
- The dedup moat is **context-length-dependent** (32K+); at short context
  the lever may not clear its own overhead — scope the gate to the regime
  where it pays and say so honestly.

## §7 — Anchors at CP start

kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`.
The attention-routing substrate primitive is held from Phase 4.6; CP 5.2
builds the wiring and tiering layer above it.
