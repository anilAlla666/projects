# G-O1 PROBE-FIRST — the multiplexing dispatch mechanism (read-only; settle before the scheduler build)

**Date:** 2026-06-01. **Type:** READ-ONLY (measured + cited; NO build, NO commit; deployed `1f305ce6`, staging
`9c318ac6` UNCHANGED, all tags stand). Probe: `pager_dispatch_probe.py`, 3 distinct resident 4-bit 7-8B models.
**Settles HOW agents interleave at dispatch without the GIL serialization the G-O3 probe hit, and confirms the
refined regime split. Three regime findings + the honest magnitudes.**

## M1 — the dispatch/GIL wall: DIAGNOSED → single-thread batched dispatch (GIL and streams both ruled out)
3 distinct models, one forward each, measured:
| dispatch path | fwd/s | vs serial |
|---|---:|---:|
| serial single-thread | 22 | 1.00× |
| multi-stream, 1 thread | 23 | **1.02× — NO GPU overlap** |
| python-threaded (3) | 14 | **0.65× — GIL serializes** |
**The prior concurrent<serial was the GIL** (python-threaded 0.65×). **Multi-stream gives no overlap (1.02×)** —
the forwards are memory-bound and contend for the same HBM bandwidth, so separate streams don't add throughput.
**Mechanism = SINGLE-THREAD batched dispatch (an async engine loop), NOT python-threaded `generate()` (GIL) and NOT
multi-stream concurrency (no overlap).** mechanism-clear.

## M2 — same-model batching (the G1∩G3 cell): CONFIRMED, but it's vLLM's lever, and this is the PREFILL number
model0, forward B={1,2,4,8,15}, S=8: latency stays ~flat (46.5→50.7 ms) while tok/s scales **172→2368 (13.77×)** —
the weight read is fully amortized over the batch → toward compute-bound. **Same-model batching IS the real
density+MFU-coincide lift.** Two honest caveats (advisor): (1) **this is a PREFILL GEMM (S=8), NOT batched decode** —
batched decode is [B,1]+KV and scales flatter, AND its **KV cache competes with the weight residency the pager
bought** (at high B × long context KV can dwarf weights); the decode-batch-with-KV number is **measured-separately-
needed**, do not headline 13.77× as decode. (2) This lift **is vLLM's continuous batching** — CIPHER does not add the
batching kernel; its only role here is routing same-model agents into a shared batch (which one vLLM engine already
does). mechanism-clear (= vLLM-subsumed).

## M3 — distinct-model temporal interleave (consolidation): mechanism clear, but "~100 agents" is a TARGET, not a result
A burst can't share a GPU forward with another model (M1: streams don't overlap) → consolidation is **TEMPORAL**
(serve bursts efficiently; the agent count = `(burst+idle)/burst`, gated by per-burst GPU time). Measured:
- GIL-bound `generate()` 30-tok burst = **1723 ms (17 tok/s single-agent!)** → only **~2-4 agents/GPU** (idle 2-5s).
- Arithmetic bandwidth-ceiling burst = **31 ms** → **~65-160 agents/GPU** (idle 2-5s).
**The ~100 is THREE stacked optimistic assumptions (advisor) — state them or it's fantasy:** (a) bursts never collide
(true P99 under correlated arrivals needs the real scheduler — not this probe); (b) burst_t actually reaches the
31 ms ceiling — but the **measured** dispatch is 46 ms/8-tok forward and 1723 ms/30-tok `generate()`; **the 31 ms is
arithmetic, nothing here shows the dispatch reaching it** — a **~30× gap of unproven dispatch work**; (c) idle stays
2-5s. Honest: **agents/GPU = (burst+idle)/burst; MEASURED (GIL-bound) ~2-4, ARITHMETIC ceiling ~65-160; the 30× gap
is the scheduler's burden to earn, not a finding.** mechanism-clear-in-principle; the agent count is unproven.

## Synthesis — the scheduler build shape (and the open question it must confront)
- **Dispatch:** single-thread batched async engine loop (GIL-free, no python-threaded `generate`; streams add
  nothing for memory-bound decode). **This is precisely what vLLM's async engine + continuous batching already is.**
- **Same-model agents → batch** (B=N, real lift, **vLLM-subsumed** — CIPHER routes, vLLM batches).
- **Distinct-model agents → temporal interleave in ONE process** — the **only CIPHER-specific residual**; gated by
  dispatch efficiency (close the 30× burst_t gap toward the bandwidth ceiling) + collision/P99 handling.
- **The gap-closer is vLLM-style dispatch**, so the scheduler's only novel burden is **multi-model-in-one-process
  temporal interleave beating N vLLM processes (each with its own efficient dispatch)** — the delta-vs-N-sleep
  question, now reached from a 4th direction ([[cipher-pager-delta-assessment]]). The scheduler must NOT be greenlit
  on "100 agents" without confronting it.

## Verdict (per regime) + next
Dispatch: **mechanism-clear** (single-thread batched). Same-model batch: **mechanism-clear (= vLLM; decode+KV TBM)**.
Distinct-model consolidation: **mechanism-clear-in-principle, agent-count unproven** (30× dispatch gap + P99 + must
beat N-sleep). Anchors unchanged (read-only). **One measurement worth exactly one run IF the build is greenlit (not
now): real batched DECODE tok/s vs B with KV at a realistic context length — the number M2 stands in for and the one
the scheduler's capacity math needs.** STOP — the mechanism is settled; the scheduler build shape (and whether it
clears the N-sleep bar) is Anil's call. [[cipher-go3-mfu-refuted]], [[cipher-pager-step4-tight-reserve]].
