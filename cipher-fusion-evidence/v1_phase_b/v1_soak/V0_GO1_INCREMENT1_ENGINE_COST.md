# G-O1 engine increment 1 — the first probe confirmed the COST side; surfacing before the months-grind

**Date:** 2026-06-01. **Type:** READ-ONLY probe (increment-1 feasibility; NO build, NO commit; deployed `1f305ce6`,
staging `9c318ac6` UNCHANGED). **Increment 1 was authorized to build the engine skeleton (pager-as-allocator +
per-model graph-decode + dispatch loop). The first probe instead delivered decision-relevant cost information: the
engine's core is reimplementing vLLM's graph-decode, fragile from probe 1, payoff still PCIe-gated. Held the build to
surface it — the cheaper path to the same product question is measurable now.**

## What the probe measured (`pager_graphdecode_probe.py`, TinyLlama in the pager region)
- **Pager + EAGER decode over the cuMemMap region: WORKS** — 201/201 weights in region (va=0x302000000, 2190 MiB),
  eager decode coherent (26 tok/s, output sensible). **The substrate is sound; weights decode correctly from the
  pager region.**
- **Graph-decode (the dispatch speedup that takes agents-per-GPU above ~2): NOT proven over the pager.** transformers'
  own compiled-generate recipe (`cache_implementation="static"` + `torch.compile(mode="reduce-overhead",
  fullgraph=True)`) **fails on a known CUDA-graph-trees aliasing bug**: "accessing tensor output of CUDAGraphs that
  has been overwritten by a subsequent run" (transformers/utils/generic.py:900 — the graph output buffer is reused
  and generate reads stale logits). **This is a transformers×torch.compile interaction, NOT the pager** (the pager
  part works; the isolation is clean).

## The decision-relevant read (advisor): increment 1 confirmed the COST, not just a blocker to route around
The only remaining graph-decode path is **hand-rolled static-KV `torch.cuda.graph` capture (CIPHER owns the buffers
+ cloning)** — which is **exactly vLLM's hardest, most-fragile component**, the thing the whole arc has repeatedly
found is vLLM's regime. So the calculus the product call was weighing is now sharper on the cost side:
- The engine build = **confirmed months of reimplementing vLLM's graph-decode core** (fragile — the first probe hit
  it even via the shortcut; manual capture fails slow over many turns of CUDA-graph debugging, not fast).
- Its payoff is **still gated on the unmeasured PCIe-cold-miss-tail SLO at M>10-15** ([[cipher-go1-inprocess-composition]]).
- Grinding into manual capture now = committing to the expensive branch by momentum — the opposite of the
  surface-when-the-premise-is-shakier discipline that's held this session.

## The cheaper path to the SAME product question (measurable now, turns not months)
**The swap-latency advantage (3-19×, [[cipher-pager-delta-assessment]]) does NOT need a full graph-decode engine to
be measured.** The pager is a proven consolidation primitive (pager-step4: N distinct models resident, KL=0,
copy-free evict, physical reclaim). If the product is "host more distinct models per GPU, swap them fast," then:
- It can ship as **pager + eager (or non-graph) decode** for the cold-swap path — slower per-token, but functional.
- The **head-to-head-vs-N-sleep** (the question deferred from the start) is **measurable TODAY on eager decode**:
  both sides pay the same per-token decode cost, so the axis under test — **swap-vs-sleep latency + memory overhead +
  models-per-GPU at matched correctness** — is isolated WITHOUT the engine. The graph-decode reimplementation does
  not need to be won to answer "does in-process fast-swap consolidation beat N-sleep."

## Recommendation (the honest service) — your call
**Measure the fast-swap-vs-N-sleep head-to-head on eager decode (turns) BEFORE committing months to the graph-decode
engine.** If CIPHER wins that on memory-overhead + swap latency + models-per-GPU, the engine is worth building (and
increment 1's manual-capture is the next step, eyes open to the cost). If it ties/loses, the months are saved and the
honest close stands: the pager is CIPHER's proven differentiated artifact (consolidation primitive); the
graph-decode/throughput regime is vLLM's. Anchors unchanged (read-only). STOP — do not start manual CUDA-graph
capture by momentum; this is a product call. [[cipher-go1-inprocess-composition]], [[cipher-go1-held-regime-map]].
