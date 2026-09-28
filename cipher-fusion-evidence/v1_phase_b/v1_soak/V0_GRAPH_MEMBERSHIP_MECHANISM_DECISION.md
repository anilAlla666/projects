# ARCHITECTURAL DECISION — CUDA-graph dynamic membership vs persistent kernel (ORCHESTRATE G-O1 + G-O5)

**Date:** 2026-05-31. **Type:** READ-ONLY mechanism decision (measurement + cited analysis; no production build,
no commit, anchors unchanged). **DECISION: GRAPH-SUFFICIENT. A persistent kernel is NOT required for bursty
dynamic membership.** The mechanism is **pre-captured batch-size BUCKETS + padding** — and it is **already
shipped in vLLM** (`cudagraph_capture_sizes`) / TensorRT-LLM. So ORCHESTRATE does **not** build a membership
dispatcher; it **rides vLLM's existing bucketed-CUDA-graph continuous batching**, and CIPHER's actual leverage
is the convergent finding of every prior task — **making the actuators (Marlin/FP8/fusion) fire *inside*
vLLM's captured graphs (the graph gate)**.

---

## 0. THE FRAME WAS WRONG — it's not {recapture-per-join / conditional-nodes / persistent-kernel}

The prompt's trichotomy omits the mechanism the entire production LLM-serving stack uses for dynamic batch
membership under CUDA graphs: **capture a small SET of fixed-shape graphs once (batch-size buckets); at
runtime route the live active-set to the nearest bucket ≥ active, pad idle slots.** Membership churn = change
which slots are padding (zero recapture); crossing a bucket = switch to another *pre-captured* graph. This
dodges per-event recapture AND the persistent kernel. The decision below is decided by measured components.

## 1. PER-EVENT RECAPTURE IS INFEASIBLE (measured) — this kills the naive option

`recapture_cost.py`, Mistral-7B, cu13: **capture+instantiate = ~25-30 ms, flat across B=8/32/64/128** (B=8
26.8, B=32 24.2, B=64 26.7, B=128 29.8 ms). Burst cadence: 100 agents × bursts every 2-5 s ⇒ ~28 membership
events/s ⇒ ~35 ms inter-arrival. **Recapture (~27 ms) ≈ inter-arrival (~35 ms) → recapturing on every
join/leave consumes ~75-150% of wall-clock** — the GPU would stall in capture more than it computes.
Per-event recapture is out.

## 2. CONDITIONAL/DYNAMIC NODES — present in the driver, NOT exposed by torch (immature)

`torch.cuda.CUDAGraph` exposes only `capture_begin/capture_end/instantiate/replay/pool/raw_cuda_graph` — **no
conditional-node API**. `libcudart` *does* carry the symbols (`cudaGraphConditionalHandleCreate`,
`cudaGraphAddNode_v2`), so conditional nodes are reachable by hand-building graph nodes via raw ctypes outside
torch's capture — immature and unnecessary. Not the path.

## 3. BUCKET + PADDING — the mechanism (cu13-reachable; co-residence + rates measured)

`bucket_coresidence.py`, Mistral-7B, cu13:
- **K=4 bucket graphs (B=8/16/32/64) co-reside in 15.1 GiB of 80** (model 13.5 + ~0.4 GiB/bucket) — room for
  dozens of buckets. Multiple fixed-shape graphs hold simultaneously; **per-event churn = pick the
  pre-captured bucket (dict lookup) + replay — NO recapture, NO allocation → switch overhead is structurally
  ~0** (the under-churn *timing* number hit the StaticCache cache-position confound — a harness artifact,
  consistent with prior tasks; not chased, and not load-bearing).
- **Per-bucket replay rate (cu13): B=8 106 / B=16 94 / B=32 84 / B=64 67 steps/s — every bucket beats the 43
  steps/s eager ceiling.** So under churn, effective steps/s = the replayed bucket's rate (51-106), holding
  the launch-elim advantage; recapture (27 ms) is paid once-per-bucket at startup, never per-event.

## 4. PADDING IS NOT FREE (corrected) — economics the user is deciding on

Earlier intuition "padding fills idle SMs, nearly free" is **false in the graph regime**: replay is
work-proportional — padding 8-active up to a B=64 bucket takes the B=64 rate (67 steps/s, 14.9 ms) vs the
tight B=8 rate (106 steps/s, 9.5 ms), ~**1.6× latency**. (The "nearly free" intuition is the *eager* regime,
where steps/s is flat across batch — not here.) So **fine-grained buckets matter** (vLLM uses 1,2,4,8,16,…) to
bound padding waste. It does not flip the decision — graph beats eager at *every* bucket — but the economics
are: you pay the bucket you pad *up to*, not the tight-fit rate.

## 5. CORRECTNESS — cited, not claimed

The "graph-sufficient" verdict rests on padded-bucket correctness (replay a B=64 graph with k active + (64−k)
padded → active slots correct, padding doesn't corrupt). I did **not** verify this here — my StaticCache+graph
harness produces garbage (the cache-position confound), so it cannot cleanly test padded correctness. **It is
the production-proven vLLM / TRT-LLM mechanism (ships in serving, correct).** My contribution is scoped to what
I measured: **cu13 reachability (capture/replay/co-residence work) + cost (27 ms recapture, per-bucket rates).**
The graph *mechanism's* execution-correctness was independently proven last task (zero-buffer-replay-recompute,
max_rel 8e-4).

## 6. THE DECISION (the two-way split)

**GRAPH-SUFFICIENT.** Not graph-insufficient-needs-persistent-kernel.
- **G-O1 (launch-elim):** CUDA graph, 2.36× over the 43-steps/s eager ceiling (measured last task), reachable today.
- **G-O5 (dynamic membership, zero-recapture):** pre-captured buckets + padding — co-residence + per-bucket
  rates measured here; per-event cost ~0; **already implemented in vLLM.**
- **Persistent kernel: NOT required.** Its only advantage over bucketing is eliminating padding-waste in the
  compute-bound regime — a *bounded efficiency optimization*, net-new research (established in
  `V0_PERSISTENT_KERNEL_FOUNDATION_SIZING.md`), a **deferred play, not a prerequisite.**

## 7. THE SHARP CONCLUSION — the mechanism is vLLM's; CIPHER's job is the graph gate

The whole campaign targets vLLM serving, and **vLLM already implements bucketed-CUDA-graph continuous
batching** (the exact G-O1+G-O5 mechanism). So **ORCHESTRATE does not build a membership dispatcher** — it
rides vLLM's graphs. CIPHER's leverage is **making the actuators (Marlin/FP8/fusion) graph-compatible so they
fire *inside* vLLM's captured graphs** — the single blocker every prior sizing converged on (Marlin/FP8/attn
are GOT-patch-bypassed under graphs; `V0_G3_…`, `V0_WSERIES_CP55_ASSEMBLY_INVENTORY.md`). The product does not
stand on a GPU-resident dispatcher; it stands on (1) graphs (launch-elim + dynamic membership, mostly vLLM's
already) and (2) continuous batching (density, proven). **Build the graph gate, not a persistent kernel.** No
code changed; measurement-only; anchors unchanged.
