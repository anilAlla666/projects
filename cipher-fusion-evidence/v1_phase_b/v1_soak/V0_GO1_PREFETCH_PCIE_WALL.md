# G-O1 prefetch probe — the correlated-burst ceiling is PCIe BANDWIDTH (a hardware constant), not a software gap. Technical arc CLOSED.

**Date:** 2026-06-01. **Type:** READ-ONLY probe (prefetch overlap + correlated PCIe sharing; NO build, NO commit
beyond this doc; deployed `1f305ce6`, staging `9c318ac6` UNCHANGED). **Probe-first settled the prefetch question
before any build: prefetch REORDERS within idle PCIe but CANNOT beat the correlated-burst wall, which is ~55 GB/s of
hardware. This is the 9th converging probe; it converged hardest — the binding limit is now a measured hardware
constant. The technical question is CLOSED; the decision is product/strategic.**

## Probe results (`pager_prefetch_overlap_probe.py`, throwaway — torch streams, numbers cited here)
- **Async H2D page-in under concurrent compute: bit-identical correct.** The async-prefetch primitive (cudaMemcpyAsync
  on a side stream) is coherence-safe — buildable correctly if ever wanted. Keep this fact (reusable).
- **The correlated case (decisive): K=1/2/4 concurrent prefetches all share ~55 GB/s → K×97ms aggregate, no speedup.**
  PCIe is one shared pipe. **Prefetch can move a page-in off the critical path (reorder), but cannot create PCIe
  bandwidth.** Under gamma-0.2 (K simultaneous cold-misses), the page-ins serialize on PCIe regardless of prefetch.
- (The single-overlap line read "serializes 1.03x" — a calibration artifact: the compute proxy ran 3ms, nothing to
  overlap. Hiding ONE 97ms page-in behind a >97ms serve window is physically sound — but it only helps where idle
  PCIe + a serve window exist, i.e. **the moderate-burstiness regime where the GPU was NOT the bottleneck anyway.**
  Helping the non-binding case is not a product win, so the full stack was not built to measure it.)

## The honest synthesis: the binding limit is a hardware constant
**Under correlated load at M past the resident ceiling, cold-miss TTFT = K × (model_bytes / PCIe) ≈ K × 97ms at the
measured ~55 GB/s — and nothing in software changes that.** Every remaining lever is miss-RATE reduction, not
bandwidth creation (enumerated, no build needed):
- Bigger resident set (KV-vs-weight budget) — already capped ~10-15 models (KV starves weights, density-axis).
- Smaller models — INT4 done (~11/80GB); further quant trades quality.
- Better locality — workload-dependent, not CIPHER's to control.
- Hot-set pinning — helps only if the hot set fits AND bursts respect it; under heavy correlation, by definition,
  they don't (the cold set gets recalled together → K cold misses → the PCIe wall).
**None creates PCIe bandwidth.** Prefetch was the last software lever standing against the tail; it reorders, it
doesn't beat the wall.

## The technical question is CLOSED (9 converging probes)
fusion(vLLM-subsumed) → 85%-MFU(bandwidth-bound) → dispatch-mechanism(single-thread) → real-dispatch(383ms,
regime-split) → composition(no external-allocator hook) → engine-increment-1(reimplements vLLM, fragile) →
binding-limit(PCIe tail at engine-speed) → prefetch(reorder ≠ bandwidth, PCIe is hardware). **They converge on one
place, and this probe nailed it to a hardware constant. Further probes are NOT the move — the remaining decision is
product/strategic and belongs to Anil, not to a 10th measurement.**

## The product call (stated without flinching)
**CIPHER's pager is a PROVEN, differentiated consolidation primitive:** N distinct 4-bit 7-8B models co-resident in
one process (~11/80GB), KL=0, copy-free evict (3-19× faster swap than vLLM sleep), physical HBM reclaim, correctness
held under burst churn (KL=0 5/5). Its **density is real**; its **correlated-burst tail is PCIe-hardware-bound** (a
constant no software relieves). The **multi-model graph-decode engine** that would sit on top is **months of
reimplementing vLLM's core AND does not relieve the PCIe bound** (the binding limit is the same with or without it).
The decode/throughput regime is vLLM's.

**The honest options for Anil (technical arc is closed; this is the strategic call):**
1. **Ship the pager as the consolidation primitive** — for low-correlation / predictable-locality multi-model serving
   (where the working set fits resident or bursts don't recall the cold set together), where its density + fast-swap
   are real wins and the PCIe wall isn't hit.
2. **Build the engine** — months, eyes open: it gets graph-decode throughput per model but the correlated-burst tail
   stays PCIe-bound; it does not change the binding limit.
3. **Close the ORCHESTRATE arc on the pager** as CIPHER's differentiated artifact, and stop — the compute/dispatch
   goals are vLLM's regime (9 probes), the memory-residency goal (the pager) is CIPHER's and is shipped.

Anchors unchanged (read-only). No harness committed this turn (the probe is throwaway; numbers cited above). STOP —
the technical work is done; the decision is yours. [[cipher-go1-binding-limit]], [[cipher-go1-inprocess-composition]],
[[cipher-pager-delta-assessment]], [[cipher-density-axis-cargo]].
