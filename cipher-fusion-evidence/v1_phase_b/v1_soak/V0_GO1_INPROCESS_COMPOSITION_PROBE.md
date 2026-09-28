# G-O1 in-process composition probe — coexistence WORKS; consolidation doesn't compose under vLLM; the engine is CIPHER's to build (with a falsifiable gate)

**Date:** 2026-06-01. **Type:** READ-ONLY (measured + 5-agent cited source workflow; NO build, NO commit; deployed
`1f305ce6`, staging `9c318ac6` UNCHANGED). **Verdict: doesnt-compose-diagnosed for CONSOLIDATION — coexistence of
fitting models WORKS (measured), but CIPHER's pager cannot manage multi-distinct-model RESIDENCY under in-process
vLLM engines. CIPHER must own the multi-model engine; its binding limit is now precisely a hardware SLO gate.**

## Probe 1 — coexistence: WORKS (measured; corrects the workflow's over-reasoned "no")
`pager_coexist_probe.py` (clean GPU, after killing a 44 GiB zombie from a prior failed run — the earlier "won't
fit" was that artifact): **2 distinct-model in-process vLLM engines (TinyLlama + Llama-3.2-1B, VLLM_ENABLE_V1_
MULTIPROCESSING=0) coexist in ONE process+CUDA-context, both serve correctly, A unchanged after B loaded** (HBM
20.7→40.0 GiB). The workflow's `multiengine` agent claimed "can't coexist (torch.distributed singleton)" — **the
measurement refutes that**: for 2 tp=1 single-GPU engines the `if not is_initialized()` guard makes the 2nd reuse
the trivial world, so the distributed conflict doesn't manifest. **Primary evidence (measurement) wins; coexistence
of fitting models is real.**

## Probe 2 — pager-fed swap / consolidation: does NOT compose under vLLM (source-verified, dispositive)
Coexistence ≠ consolidation. The VALUE is paging idle models OUT to fit MORE distinct models than HBM holds — which
needs the pager to manage the engines' weight RESIDENCY. Two independent, source-verified blockers (either is
dispositive; both confirmed in `vllm/device_allocator/cumem.py` + `gpu_worker.py`):
- **CuMemAllocator is a process-wide SINGLETON** (cumem.py:106-128 — the C-ext free callback is ONE global var;
  `get_instance()` enforces one instance) and **sleep mode is one-instance-per-process** (gpu_worker.py:213-215
  asserts `get_current_usage()==0`). sleep()/wake_up() are whole-pool CPU-offload keyed by coarse tags, NOT a
  per-model residency manager. → two models can't be independently paged.
- **No external-allocator hook** (ext-alloc, composes=no): weight loading is hardwired through
  `use_memory_pool_with_allocator → CUDAPluggableAllocator(lib, "my_malloc","my_free")` (cumem.py:66-87,
  gpu_worker.py:316-323, with a FIXME admitting the hack). CIPHER's STEP-1.5 pager captures weights via the SAME
  scoped `use_mem_pool` mechanism vLLM monopolizes for its singleton → **CIPHER and vLLM contend for the single
  pluggable-allocator slot rather than composing.** Per-agent-KL=0-across-swap was therefore untestable under vLLM
  (no composable swap exists to test).

## Probe 3 — head-to-head: CIPHER's consolidation side can't run under vLLM → end-to-end UNMEASURED
For models that FIT resident: in-process multi-engine works (measured), but the edge over N subprocesses is only
~633 MiB/engine context savings ([[cipher-pager-coresidence-build]]) — modest, and the pager isn't even needed
(both resident). For CONSOLIDATION (>fit): CIPHER's measured 3-19× swap advantage ([[cipher-pager-delta-assessment]])
is **unrealizable under vLLM** (no composable residency manager). So the head-to-head consolidation number is
UNMEASURED — gated on an engine that doesn't exist off-the-shelf.

## The decision this settles: CIPHER must OWN the multi-model engine — and its win is FALSIFIABLE
The pager primitive is proven (STEP-1.5: 100% residency, page_out physically frees HBM, byte-identical restore);
what's unavailable is the HOST. **CIPHER's next build = a multi-model graph-decode runtime where the weight memory
IS the pageable cuMemMap region (pager-as-CUDAPluggableAllocator across N models), riding vLLM's already-shipped
pre-captured batch-bucket graphs per model ([[cipher-graph-membership-decision]]) — NOT living inside a vLLM
process.** The CuMemAllocator singleton is the DIAGNOSIS of why vLLM-composition fails, NOT the next engine's limit
(that engine owns its allocator → the software blocker disappears).

**The next engine's binding limit (precisely diagnosed, falsifiable) = PCIe cold-miss page-in tail under CORRELATED
bursts.** CIPHER's own numbers: ~70 ms to page in one INT4 7B over a flat ~51 GB/s aggregate PCIe; K simultaneous
misses serialize to ~K×70 ms added to TTFT ([[cipher-density-axis-cargo]]). The engine WINS iff, at the target
distinct-model count, the working-set cold-miss rate keeps **P99 TTFT under SLO**; it LOSES the moment correlated
wake-ups pile page-ins faster than 51 GB/s drains. Resident ceiling ~10-15 models (KV starves weights) is the INPUT
to the miss rate. NOT binding (already solved): graph recapture (pre-captured buckets), the singleton (gone once
CIPHER owns the allocator), compute-kernel actuators (Marlin/fusion/FP8 redundant vs vLLM).

**Workload mock spec (for the head-to-head once the engine exists) — built to TRIGGER the tail, not just be
realistic:** M distinct INT4 7B models, sweep M={8,12,16,24,32} (crosses the ~10-15 ceiling); **gamma bursty
arrivals, burstiness ∈ {1.0 control, 0.5, 0.2}** (correlation is the decisive knob) + Zipfian model affinity
(hot/cold rotation = the agent-fleet pattern); input 1024 / output 128 multi-turn (ShareGPT). **Primary output: P99
TTFT decomposed into queue + page-in-wait + prefill, as f(M, burstiness)** + observed page-in concurrency / effective
PCIe GB/s. **Win: CIPHER serves M past the ceiling at P99-under-SLO where N-vLLM OOMs/cold-loads; loses if
burstiness=0.2 page-ins blow P99.** Run the burstiness=1.0 control (uncorrelated would falsely pass).

## The product call (7th converging probe — now crisp, no more probes)
The consolidation is real and the pager is proven, but realizing it requires CIPHER to build its **own multi-model
graph-decode engine** (months — pager-as-allocator + ride vLLM's per-model graphs). The strategic question, now
sharp: **is that engine worth building, given (a) the 100-agent SAME-model case is already vLLM's (solved); (b)
coexistence-of-fitting-models works today with only ~633 MiB/engine savings vs N-processes; (c) the consolidation
(>fit) win is the 3-19× swap, realizable ONLY after the engine; and (d) that engine's win is itself gated on a
falsifiable PCIe-cold-miss-tail SLO at M>10-15 under correlated bursts?** Anchors unchanged (read-only). STOP — this
decides the build + the product call. [[cipher-go1-held-regime-map]], [[cipher-pager-delta-assessment]].
