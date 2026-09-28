# ROOT-CAUSE — cold-miss TTFT = K × (bytes_pulled / PCIe), decomposed to physics-vs-design

**Date:** 2026-06-01. **Type:** READ-ONLY diagnosis (interconnect/H2D measured live + K re-measured + 5-agent
root-cause workflow; NO build; deployed `1f305ce6`, staging `9c318ac6` UNCHANGED). **The "PCIe wall" the arc called
a hard ceiling decomposes into: PCIe = physics (working floor), bytes = at the INT4 quant floor, and K = a CIPHER
RESIDENCY-POLICY term the arc's budget-forced measurement INFLATED. The biggest fixable lever is K (residency), and
its substrate (INT4 co-residence) is already built — what's missing is measuring K under REALISTIC load, not a
months-engine.**

## Per-term: physics-floor vs CIPHER-current (all numbers measured live on this pod)
- **PCIe (the divisor) = PHYSICS / working floor: 55 GB/s.** Measured: 5 GiB pinned H2D = 98 ms = 55 GB/s, flat
  across 1/2/4 streams (link-saturated, not single-stream). Root: H100 reports Gen5 x16 (32 GT/s) but the QEMU
  virtual root port reports Gen4 — the 55 (vs Gen5's ~63 electrical max) is a **KVM-layer throttle, recoverable only
  by de-virtualizing the pod** (one-time +12.8%; no NVLink/C2C/GDS path — single VM-isolated GPU). **Not a lever.**
- **bytes_pulled (the numerator) = at the INT4 quant floor.** fp16→INT4 already banked the 4× (14.5→3.6 GB);
  sub-INT4 (INT2) is **quality-gated** (on-pod FP8 already costs +0.567% PPL). Shared-base/delta (10-100×) is REAL
  but **vLLM-multi-LoRA's for shared-base fleets, and ~0 for the stated DISTINCT-base domain** (Llama vs Qwen vs
  Mistral — weight-share fires only on opportunistic hash collision). So bytes is at its honest floor for distinct.
- **K (the multiplier) = CIPHER RESIDENCY POLICY — the ONLY term with order-of-magnitude software headroom.** The
  per-miss cost (98 ms) is hardware-floored and serializes strictly (K=1/2/4 = 98/195/390 ms, PCIe-flat — prefetch
  reorders, doesn't add bandwidth). But the COUNT of PCIe-crossing misses is policy. **The arc's 63%-cold was a
  budget-forced artifact** (~11 GiB cap, 3-of-5 evicted). **Re-measured live at a hot-set-resident budget (18 GiB):
  cold-miss 63%→34%** — confirming K is policy, not physics. Under realistic Zipfian + hot-set-resident (~10-15 INT4
  ceiling), K is small → **the wall becomes a RARE TAIL EVENT, not the common ceiling.**

## The biggest fixable lever: K (cold-miss count) via residency policy
The only term with order-of-magnitude headroom. Mechanisms (all miss-RATE, never bandwidth): (1) **bigger resident
set — INT4-resident, ~4× more models, zero per-token re-paging** (the proven decode-density lever — and partial-layer
was already diagnosed BACKWARDS: K<N resident re-pages every evicted layer/token = 6.1× slowdown); (2) locality-aware
admission/eviction tuned to the Zipfian hot set; (3) the HBM-cold-tier, correctly understood as a **K-attack** (it
relocates WHICH misses pay PCIe — a warm-RAM miss → in-HBM decompress at ~TB/s — NOT a separate physics win;
capacity-bounded ~44 INT2 models, quality-bounded, useless for a true first-touch).

**Hard boundary (the point, not a caveat): hot-set pinning FAILS under heavy correlation** — the cold set is recalled
together, so K×98 ms re-appears for genuinely-cold models. The K-lever turns the wall into a **rare tail event; it
does not delete it.**

## Irreducible floor (after every byte + K + path attack)
**ONE first-touch cold INT4 model pull at K=1 = 98 ms (55 GB/s KVM) / ~85 ms bare-metal**, × K under correlated
true-cold bursts (strictly serialized — one PCIe pipe, no alternate path). Below this is not CIPHER's to take: it's
workload locality (operator's request pattern) or quality loss (sub-INT4). **Honest product line: CIPHER's pager is a
proven distinct-model consolidation primitive whose correlated-cold-burst tail is PCIe-hardware-bound; the lever is
residency policy (shrink K), not bandwidth.**

## What the arc got wrong vs right, and the next move
The arc's "PCIe wall = hard ceiling" was **measured under a budget-forced 63%-cold artifact** (budget=2), inflating K.
The realistic regime (Zipfian + hot-set-resident, the **already-built** INT4 co-residence, `pager-step3-int4-density`)
makes K small → the wall is a rare tail, not the common case. **So the biggest lever (K via residency) is LARGELY
ALREADY BUILT.** What's missing is not a build — it's a **MEASUREMENT**: instrument K under a realistic Zipfian +
hot-set-resident sweep (does K stay small, vs the budget-forced 63%?) + the INT4-quality gate (does INT4 hold across
the fleet, given FP8's +0.567% PPL?).

**NEXT MOVE (the single decision): measure K under realistic load on the built INT4 co-residence — NOT a months-build.**
- Do **NOT** build the multi-model graph-decode engine (months, reimplements vLLM, does NOT relieve the PCIe tail).
- Do **NOT** pursue bare-metal PCIe (one-time +12.8%, requires de-virtualizing the pod).
- DO measure: realistic-Zipfian K + INT4 fleet quality → if K stays small, the pager serves the common case well
  (ship as consolidation primitive); if K stays large even realistic, the correlated-cold tail is the irreducible
  PCIe floor and the product is low-correlation-only. Either way it's a measurement, not a 4th sizing.

Anchors unchanged (read-only). [[cipher-go1-prefetch-pcie-wall]], [[cipher-pager-int4-build]],
[[cipher-go1-binding-limit]], [[cipher-density-axis-cargo]].
