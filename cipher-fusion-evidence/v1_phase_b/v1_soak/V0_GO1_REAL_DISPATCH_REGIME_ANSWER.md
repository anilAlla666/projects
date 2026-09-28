# G-O1 on REAL graph-decode dispatch — the regime-split agent count (the number you asked for)

**Date:** 2026-06-01. **Type:** READ-ONLY measurement on the real engine (vLLM graph-decode, NOT HF generate; NO
build, NO commit; deployed `1f305ce6`, staging `9c318ac6` UNCHANGED, all tags stand). **The 4.7s/burst WAS a
harness artifact — real graph-decode is 383ms. The honest agent count is regime-split, and that split IS the
answer.**

## The real number (measured, vLLM graph-decode — `vllm_burst_probe.py`)
- **Single-agent 64-token decode burst: 383 ms = 167 tok/s/agent** (HF generate was 4172 ms / 15 tok/s →
  graph-decode is **~11× faster**; the ~2-agent figure was the HF strawman, now retired).
- **SAME-MODEL agents → 100 is reachable TODAY (vLLM continuous batching):** N=1/8/32/64 → 167 / 1027 / 4880 /
  **8808 tok/s** (53× at N=64); higher B fits more. **An agent fleet that is one base model × many instances is a
  solved problem — vLLM's continuous batching, not a CIPHER gap.** This is the G1∩G3 cell.
- **DISTINCT-MODEL agents → ~6-14 (fp16, measured) / ~12-29 (4-bit, reasoned):** agents/GPU = (burst+idle)/burst.
  fp16 383ms burst → ~6 (idle 2s) / ~9 (3s) / ~14 (5s). 4-bit burst ~180ms (DERIVED from the measured fp16:
  weight-read 4.2ms/tok fp16 → ~1ms at 4-bit + ~1.8ms overhead; vLLM-4bit load failed on this pod, so this is
  reasoned-from-measured, NOT measured-4bit) → ~12 (2s) / ~18 (3s) / ~29 (5s). Bandwidth+dispatch-bound; **not 100
  unless duty is very low.**

**So G-O1's "100 agents": YES for same-model (vLLM, today); ~12-56 for distinct-model (graph-decode + bandwidth
bound). The product decision is which regime it needs.**

## The architectural composition gap (stated at the confidence the evidence supports)
The distinct-model consolidation (CIPHER's differentiated cell) needs a graph-decode engine serving N DISTINCT
models in ONE process. Status:
- vLLM is **one-model-per-engine** (cited, established in the vLLM probe). A *different* model cannot run under one
  engine's captured graphs + model-specific config (sound).
- **No off-the-shelf multi-distinct-model graph-decode engine exists.** CIPHER's pager holds N distinct models'
  weights in one process (proven), but there is no engine to graph-decode them all from that process.
- **The fix is NOT binary "reimplement vLLM (months)."** The UNTESTED middle path: **N small vLLM engines in one
  process, the pager picking who's awake** — possibly a probe, not a rewrite. (NOT tested whether 2 vLLM engines
  coexist in one process; do not assert impossible. The earlier "43.9 GiB each won't fit" was a
  `gpu_memory_utilization=0.55` config artifact, not a physical limit — retracted.)

## Head-to-head vs N-sleep (exact — what's measured vs what isn't)
- **Swap primitive: CIPHER pager is 3-19× faster than vLLM sleep/wake** (MEASURED: page_out 6ms + page_in 308ms =
  314ms vs vLLM sleep 5640ms + wake 416ms; [[cipher-pager-delta-assessment]]).
- **End-to-end agents-per-GPU: UNMEASURED** — CIPHER lacks the multi-model decode engine, so its consolidation side
  can't be run without HF generate (the rejected 4.7s strawman) or a non-existent engine. Running the vLLM-sleep-
  cycling baseline alone would be one-sided (proves nothing without CIPHER's side), so it was NOT run.
- **Verdict: swap-advantage real and measured; end-to-end realization gated by an engine that doesn't exist
  off-the-shelf.** Neither "CIPHER wins" nor "blocked" — gated.

## The decision (this is the 6th probe converging — name it, stop the loop)
Six probes (fusion, MFU, density, dispatch-mechanism, build-held, this) converge on ONE finding: **the pager is
CIPHER's differentiated artifact (multi-model residency, proven); the decode/throughput regime is vLLM's.** This
does not change by measuring a 7th variant. **The open question is no longer technical-measurable — it is a PRODUCT
decision:**

> **Is an in-process faster-swap multi-model consolidation layer (which still needs a multi-model decode engine —
> at minimum the untested in-process-multi-vLLM-engine probe) worth building, GIVEN that the 100-agent case is
> already solved by same-model continuous batching (vLLM), and the distinct-model case tops out at ~12-56/GPU
> bandwidth-bound, and CIPHER's only proven edge there is a 3-19× swap primitive whose end-to-end value is
> unrealized without that engine?**

Anchors unchanged (read-only). STOP — the number is found (regime-split), the gap is precise, the next move is a
product call, not another probe. [[cipher-go1-held-regime-map.md]], [[cipher-pager-delta-assessment]],
[[cipher-go3-mfu-refuted]].
