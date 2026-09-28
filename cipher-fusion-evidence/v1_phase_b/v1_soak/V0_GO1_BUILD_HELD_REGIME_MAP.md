# G-O1 multiplexer — BUILD HELD (probes determined the outcome) + the ORCHESTRATE regime map

**Date:** 2026-06-01. **Type:** probe-first → build HELD (READ-ONLY; NO build, NO commit; deployed `1f305ce6`,
staging `9c318ac6` UNCHANGED, all tags stand). **The gate-critical probes already determine G-O1's outcome; building
the multiplexer would measure HF-`generate()` dispatch overhead, not G-O1 — the don't-build-cargo line this session
held five times (fusion / partial-layer / compaction / 85%-MFU / and the dispatch concurrency paths).**

## The two gate-critical probes (the deciders)
- **(A) Correctness ACHIEVABLE:** batched `generate(B=N)` is **per-agent bit-identical** to solo — A1 (same-prompt
  ×4: every row == solo) and A2 (4 different padded prompts: each generated tail == solo). Batching does not perturb
  any agent → the G-O8 per-agent KL=0 gate would pass. (`pager_batch_correctness_probe.py`)
- **(B) Dispatch is the binding constraint, and it's vLLM's regime:** batched decode tok/s-vs-B with KV (256-ctx +
  64 decode): 15→202 tok/s, B=1→15 = **13.18×** (decode batching lift is real, ≈ the prefill 13.77×). BUT the
  absolute is the story: **B=1 = 15 tok/s = ~73 ms/token** — ~70× above the ~1 ms/token HBM-bandwidth ceiling. HF
  `generate()` is Python-per-token-bound. Batching amortizes the weight READ (aggregate 13×) but each agent's
  64-token burst still takes ~4.7s.

## Why the build is held — the arithmetic G-O1 already failed on
Agents/GPU = `(burst + idle)/burst_t`. At HF-generate's 73 ms/tok, a 64-token burst = **4.7s**, which *dominates* a
3s idle (duty ~60%, not the 2% the consolidation claim assumes) → **~1-2 agents/GPU**. That's not a number worth a
build — it's arithmetic on a measured value. Closing the 70× gap requires **CUDA-graph decode with no Python-per-
token = vLLM's engine core** — not "like" it, it *is* it. vLLM is one-model-per-engine + subprocess, so the two
efficient-dispatch options are: **(a) N vLLM processes = the N-sleep baseline CIPHER must BEAT (not beat-with)**, or
**(b) reimplement vLLM's graph-decode engine in-process (months, re-deriving the artifact whose existence already
refuted three prior CIPHER actuators).** Either way the novel surface is multi-model-in-one-process, whose entire
edge over N-sleep is the ~633 MB/process context + copy-free evict already measured — **a consolidation edge, NOT a
throughput win. G-O1's 100-agents was a throughput claim. It dies for the SAME root reason as the 85% MFU headline:
the lever is batched compute-efficient dispatch = vLLM's existing regime, not a CIPHER substrate gap.**

## The ORCHESTRATE regime map (the bigger picture for Anil)
Across the arc, the goals/actuators that touch the **compute/dispatch** regime have each resolved to "vLLM already
occupies it":
- G-O4 fusion → vLLM already fuses ([[cipher-fusion-not-vllm-cargo]]). Marlin/FP8 → redundant-or-regress vs vLLM.
- G-O3 85% MFU → physically unreachable for distinct-model decode (bandwidth-bound; [[cipher-go3-mfu-refuted]]);
  high MFU = batched same-model = vLLM continuous batching.
- G-O1 100-agents → dispatch-engine-bound; efficient dispatch = vLLM's engine = N-processes-or-reimpl (this doc).
- G-O7 graph-gate / G-O5 membership → mechanisms proven, but ride vLLM's graphs.

**CIPHER's differentiated ground is the ONE thing vLLM structurally does NOT do: multi-model MEMORY RESIDENCY — the
pager.** That is a genuine, proven artifact (N distinct 4-bit models co-resident in one process, KL=0, copy-free
evict, ~11 models/80GiB; pager-step1..step4). It is a **consolidation/density play**, NOT a throughput or MFU play.
This is a RE-SCOPING, not a failure: the honest product is **"more distinct models per GPU at correct outputs and
bounded P99,"** and it has ONE unanswered, load-bearing question.

## The highest-value next move (advisor): the measurement deferred from FIVE directions
Not a scheduler — the **delta-vs-N-sleep measurement**: CIPHER's pager (N distinct models in one process) **vs N
vLLM sleep-mode instances**, head-to-head, on **cold-start/swap latency + memory overhead (agents-or-models-per-GPU)
at matched correctness**. The whole arc's defensibility rests on it: if CIPHER wins there, the pager is a real
product; if not, better to know before building anything on top. It is a measurement (where CIPHER's evidence has
been strongest), not a build. ([[cipher-pager-delta-assessment]] sized it ~3× on copy-free evict, conditional and
unmeasured head-to-head.)

## Anchors / discipline
deployed `1f305ce6`, staging `9c318ac6` UNCHANGED (read-only; no build, no commit). Held the directed build because
the probe-first the user mandated already settled it — surfacing the finding + the regime map rather than building a
harness against a headline the physics killed. STOP for Anil's direction. [[cipher-go1-dispatch-mechanism]].
