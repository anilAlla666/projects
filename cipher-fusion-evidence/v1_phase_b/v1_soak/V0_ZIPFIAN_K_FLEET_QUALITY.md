# Realistic-Zipfian K + INT4 fleet-quality — the common case is SLO-clean; quality is bounded-but-family-variable

**Date:** 2026-06-01. **Type:** measurement on the built INT4 co-residence (harnesses `pager_zipfian_sweep.py` +
`pager_int4_fleet_quality.py`; deployed `1f305ce6`, staging `9c318ac6` UNCHANGED — no substrate change, OFF
byte-identical). **The decisive common-case number: with a hot-set-resident budget (the K-lever), P99 stays
SLO-clean ACROSS the burstiness sweep — confirming the root-cause that the arc's blowup was a budget-forced
artifact, not physics. INT4 fleet quality is bounded (<2% PPL) but family-variable.**

## PART A — K + P99 vs burstiness (M=5 distinct 4-bit, hot-set budget 18 GiB, Zipfian + growing context)
| burstiness | cold-miss% | p50 TTFT | p99 TTFT | |
|---:|---:|---:|---:|---|
| 1.0 (uncorrelated) | 36% | 630ms | 1842ms | SLO-clean |
| 0.5 | 34% | 405ms | 1777ms | SLO-clean |
| 0.2 (correlated) | 33% | 409ms | 1880ms | SLO-clean |

KL=0 across swap 5/5 (correctness held). **The load-bearing finding is FLATNESS, not "under 2000": P99 is ~1800ms
FLAT across burstiness (1842/1777/1880), and cold-miss barely moves (36→33%) as arrivals go uncorrelated→correlated.
That flatness is the evidence the residency lever NEUTRALIZES correlation — K does not spike at gamma-0.2 because the
hot set absorbs most traffic.** The arc's 3849ms blowup was the **budget=11 GiB forced-paging artifact** (63% cold);
the hot-set-resident budget (18 GiB) removes it. **Caveats on the absolute number (not buried):** (a) the **2000ms
SLO is an arbitrary harness line** — a real interactive TTFT SLO is often 200-500ms, which all three rows blow by
4-9×; "passes" is SLO-dependent. (b) This is **eager** P99 (queue-dominated); engine-speed lowers it. (c) cold-miss
stayed ~flat 33-36% rather than DROPPING at low burstiness — so **the LRU policy is under-tuned** (leaving the
Zipfian win on the table; a frequency-aware policy would push cold-miss toward the ~5% Zipfian ideal). So: **the
lever works (correlation neutralized) AND is under-tuned — both true.**

**Honest caveats (binding, not buried):**
- **M=5, not the ~10-15 ceiling** (pod's distinct-model limit). At M=15 with a hot-set budget, the cold TAIL is
  larger; a correlated-cold burst recalling 5+ cold models = 5×~117ms page-in serialization could approach/exceed
  SLO. The hard boundary (hot-set pinning fails when the cold set is recalled together) is UNTESTED at scale — the
  M=5 result validates the MECHANISM (hot-set-resident → SLO-clean) but not the M=15 correlated-cold tail.
- **P99 is eager-single-thread-queue-dominated** (~1800ms is the eager number). At engine-speed the queue shrinks
  (more headroom) but the page-in tail is the term; at this budget K is low so both are SLO-clean — the SLO-clean
  claim holds at both speeds *because residency keeps K low*, not because of the queue.
- **cold-miss 33% is the LRU policy** (evicts by recency × size); a frequency-aware policy holding the Zipfian-hot
  set regardless of size would push cold-miss toward the ideal Zipfian-tail (~5% at a=1.6) — the next residency-
  tuning lever, not yet applied.

## PART B — INT4 (NF4) fleet quality, REALISTIC eval (~900-token passage, per family)
| family | PPL fp16→nf4 | KL(fp16‖nf4) mean | KL p99 (tail) |
|---|---|---:|---:|
| Mistral-7B | 2.567 → 2.583 (**+0.64%**) | 0.0075 | 0.0619 |
| Qwen2-7B | 2.922 → 2.974 (**+1.79%**) | 0.0108 | 0.0836 |
| Llama-3.1-8B | 2.829 → 2.866 (**+1.31%**) | 0.0194 | 0.1453 |

**NF4 fleet quality is BOUNDED (<2% PPL) but FAMILY-VARIABLE** (Mistral +0.64% best, Qwen2 +1.79% worst — the fleet
bound). The realistic eval surfaced **tail-token degradation** (p99 KL up to 0.145 on Llama-3.1) that the 4-short-
prompt indicative (KL 0.0085) hid — exactly the long-tail concern flagged earlier. Comparable to the FP8 +0.567%
reference. NF4-runtime labeled (CIPHER-Marlin-format is the production number). **Shippable for quality-tolerant
fleets (<2% PPL bound); marginal for quality-critical (Qwen2 +1.79%, tail KL 0.145).**

## Verdict: MECHANISM VALIDATED, NOT PRODUCT-PROVEN
- **Validated (real, encouraging):** the residency K-lever **keeps K flat under correlation** — cold-miss 36→33%
  and P99 ~1800ms flat as burstiness goes 1.0→0.2 at M=5. This confirms the root-cause: the arc's blowup was a
  budget artifact, not physics; the lever (already built) neutralizes correlation in this regime.
- **NOT product-proven — two deciders still open, one unmeasurable here:**
  1. **M=5 tested the lever at its EASIEST** — budget=18 GiB holds ~4 of 5, so only 1-of-5 is evictable: a thin
     test of "correlated cold-set recall." **The binding question — does it hold at M=15 where the cold tail is
     large (a correlated recall of 5+ cold models = 5×~117ms PCIe-serialized)? — is exactly the UNTESTED case, and
     it is UNMEASURABLE on this pod (only 5 distinct models).** Do not read M=5-flat as M=15-fine.
  2. **A real TTFT SLO at engine-speed:** ~1800ms eager passes only a lax 2s line; interactive SLOs (200-500ms) need
     engine-speed + the page-in tail measured at M=15. Open.
- **INT4 quality: bounded <2% PPL but family-variable** (Mistral +0.64%, Qwen2 +1.79%, tail KL to 0.145) —
  shippable for quality-tolerant fleets, marginal for quality-critical.

**Honest line: a promising consolidation primitive whose common-case correlation-resilience is validated at small M;
the product decision needs the scale test.** Cheap next lever (if pursued on-pod): **frequency-aware residency
tuning** (cold-miss below LRU's 33% toward the Zipfian ideal) — a policy refinement, not a months-engine.

## The gating EXTERNAL dependency (so the loop doesn't continue on a pod that can't answer it)
Nine-plus probes have established BOTH the negative (the graph-decode engine is vLLM's regime, correlated-burst tail
is PCIe-bound) AND the bounded-positive (residency keeps the common case flat at M=5). **That is a complete enough
picture for a product decision.** The single thing that would change the answer is NOT another probe on this pod — it
is a **multi-distinct-model fleet at M=15+ on hardware not available here** (this pod has 5 distinct models, PCIe is
KVM-throttled). Name that as the gating external dependency. Anchors unchanged. [[cipher-pcie-rootcause]],
[[cipher-pager-int4-build]], [[cipher-go1-binding-limit]].
