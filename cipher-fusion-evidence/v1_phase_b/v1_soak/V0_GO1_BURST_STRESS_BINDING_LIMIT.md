# G-O1 binding-limit measurement — eager hides it; at engine-speed the PCIe page-in tail IS the gate (confirms, not refutes)

**Date:** 2026-06-01. **Type:** measurement (CIPHER-side correlated-burst stress; harness `pager_burst_stress.py`;
deployed `1f305ce6`, staging `9c318ac6` UNCHANGED — existing pager, no substrate change → OFF byte-identical).
**The decision-relevant number. Two clean results stand (correctness under churn; serialized page-in latency stable).
The headline reframe is CORRECTED per advisor: this measurement CONFIRMS the composition probe's PCIe-tail gate at
engine-speed — it does not refute it. The product call is unchanged: the engine is months to build AND payoff-gated
on the PCIe tail; the pager is sound underneath.**

## Setup (real-workload mock — gamma/Zipfian, NOT synthetic uniform, NOT HF-generate-toy)
5 distinct REAL 4-bit models (Mistral-7B 4.8GB, Qwen2-7B 7.7GB, Llama-3.1-8B 7.8GB, TinyLlama 1.1GB, Llama-3.2-1B
1.1GB) in ONE process behind the pager; budget ~11 GiB → ~2 resident, 3 evicted (forced paging). Traffic: gamma
burstiness=0.2 (correlated) arrivals, Zipfian model selection (hot/cold), growing per-agent context (256→1024,
multi-turn). Single-thread server, **eager decode** (the deferred-engine substitute). 160 requests, 101 cold misses.

## Results — what STANDS (clean)
- **Correctness (G-O8): KL=0 across swap — 5/5 models** first-decoded-token identical to solo reference after the
  burst churn. The pager preserves correctness through heavy paging under load. **Solid.**
- **Serialized page-in latency is STABLE under the burst: p50=117ms, p99=176ms** (4-bit 5-8GB models), ≈ the
  sequential baseline. **Solid — but note what it is and isn't (below).**
- P99 TTFT = 3849ms (max 4294ms) under gamma-0.2; p50=1123ms.

## Results — what was MIS-ATTRIBUTED, now corrected (advisor)
- **Page-in is ~32% of the serialized cost, NOT ~5%.** The naive `p99 page_in_wait (176ms) / p99 TTFT (3849ms) ≈ 5%`
  compares ONE request's page-in against a ~13-deep backlog — apples-to-oranges. Honest share = page-in's fraction
  of total work: **~101 cold × 117ms ≈ 11.8s page-in vs ~160 × 158ms ≈ 25s prefill → page-in ≈ 32%.** The
  "queue=3660ms" is the SUM of preceding requests' (page-in + prefill), so it LAUNDERS the page-in cost into "queue"
  and then reports page-in as negligible. The earlier "limit is dispatch, not PCIe" was that artifact — **retracted.**
- **The contention catch was NOT tested.** The harness is a sequential for-loop — page-in and serve never run
  CONCURRENTLY, so there's no PCIe contention to degrade the swap. "Stable at 176ms" = **serialized** page-in latency
  is stable (true, keep) — NOT the concurrent-page-in-under-load stress the catch named (that arises only with
  prefetch/async overlap, which is deferred). **The contention question remains open.**

## The honest binding-limit read: eager hides it; the engine EXPOSES the PCIe gate
At **eager** speed prefill (158ms) ≈ page-in (117ms), so prefill+queue dominate and page-in *looks* small. **But the
engine you would build collapses prefill/decode toward ~20ms while page-in stays ~117ms (PCIe-bound, unchanged).** So
at engine-speed the per-cold-request floor ≈ 117ms page-in + ~20ms compute → **page-in becomes ~85% of it**, and under
a correlated burst the K cold page-ins serialize into exactly the **K×~70-117ms PCIe tail the composition probe
predicted** ([[cipher-go1-inprocess-composition]]). **This measurement CONFIRMS the PCIe gate at engine-speed — it
does not refute it.** At eager speed the single-thread queue dominates (page-in ~32%); at engine speed the PCIe
page-in dominates. Both are real; the PCIe one governs the engine that would actually ship.

## What it settles for the product call (unchanged in shape, sharper)
- **The pager is sound under realistic correlated load** — KL=0 through the churn, serialized swap stable at ~117ms.
  The residency primitive holds. (Concurrent-page-in-under-overlap is still untested.)
- **The engine is BOTH months to build AND payoff-gated on the PCIe cold-miss tail** — building it does not relieve
  the PCIe gate; at engine-speed the page-in serialization is the binding limit (this measurement shows it, once you
  subtract eager's prefill mask). The composition probe's gate stands, now measurement-backed.
- **Honest alternative unchanged:** ship the pager as a consolidation primitive for low-concurrency / non-correlated
  serving (where neither the queue nor the PCIe tail explodes), and not build the engine.

This is the 8th converging probe — same place as the prior seven: **the pager is the proven differentiated artifact;
the decode/dispatch regime is vLLM's; the consolidation's gate under correlated load is the PCIe page-in tail.** No
9th probe needed. Harness committed; KL=0 passes; anchors unchanged. STOP — the product call is the user's.
[[cipher-go1-engine-increment1]], [[cipher-go1-inprocess-composition]], [[cipher-pager-delta-assessment]].
