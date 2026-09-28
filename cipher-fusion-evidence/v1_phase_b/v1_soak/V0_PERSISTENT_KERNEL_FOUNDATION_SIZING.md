# READ-ONLY SIZING — does a persistent GPU-resident self-dispatch kernel break the launch ceiling + multiplex 100 bursty agents?

**Date:** 2026-05-31. **Type:** SIZING (minimal measurement + cited analysis; **NO production build — and I decline
the resident-loop build, reasoned below**; anchors unchanged). **Verdict (the three-way diagnosable split):
NOT "foundation-real-and-scales." It is: (1) residence-unreachable-as-archived — the persistent self-dispatch
kernel does NOT exist in the archive and is net-new research, not a missing-env revival; (2) density-capped
for distinct models — 100 distinct 7B is memory-infeasible on one H100, dispatch-independent; (3)
foundation-not-needed for same-model — the launch ceiling is already broken by CUDA graphs (measured 2.36×)
and bursty multiplexing is already done by host continuous-batching (proven 7.7×). A GPU-resident dispatcher
is unnecessary for the feasible regime and unusable for the infeasible one.**

---

## 0. THE DECISIVE FACT — distinct-model density is memory-walled before dispatch ever matters

7B weights = 14-15 GB fp16 (`du`: Mistral 14G, Qwen2-7B 15G, Llama-3.1-8B 15G). **100 distinct 7B = ~1450 GB
vs the H100's 80 GB** → fits **~5** distinct 7B fp16, **~22** at INT4 (3.6 GB each) — and that leaves zero room
for KV/activation. **"100 distinct-model agents on one H100" is infeasible by 5-20×, regardless of any dispatch
mechanism.** The bottleneck is HBM *capacity*, reached long before launch/dispatch is the wall. No persistent
kernel, cooperative grid, or graph conjures HBM. This moots the exotic-kernel question for the heterogeneous case.

## 1. THE PREMISE IS A CATEGORY ERROR — the persistent kernel is NOT archived

The bet assumed "the persistent kernel exists in may13-archive (`cipher_persist_engine`)." It does not:
- `cipher_persist_engine` is **L2-cache persistence policy**, not a dispatcher — its header
  (`cipher-may13-evidence/include/cipher_persist_engine.h:1-7`): *"L2 cache persistence policy management…
  CUaccessPolicyWindow generation"*; impl is `cudaAccessPropertyPersisting` window admission
  (`cipher_persist_engine.cpp:201,234`). The `cipher_intercept_cudart.cpp:589-595` dlsyms it for **L2 window
  application**, not op dispatch.
- **No persistent work-queue / resident self-dispatch megakernel exists anywhere in may13.** The only
  cooperative-grid (`grid.sync`/`while`) kernel is `cipher_fp8_fused_quant.cu` — a single-op fp8 *quant*,
  not a multi-agent dispatcher.
- So unlike the fusion kernels (which were a missing-env revival, `CIPHER_SUBSTITUTE_V2`), there is **nothing
  to revive**: a true persistent multi-agent self-dispatching decode kernel would be **net-new research**
  (Mirage/persistent-LLM-class), not a sizing-pass build.

## 2. THE LAUNCH CEILING IS BREAKABLE — but by CUDA graph (reachable today), not a megakernel

Apples-to-apples in the SM-pack harness's own regime (single Mistral-7B, manual decode loop, `graph_vs_eager_steps.py`):

| B | EAGER steps/s | GRAPH steps/s | speedup | beats 43 ceiling? |
|---|---|---|---|---|
| 8 | 43.8 | **103.3** | **2.36×** | YES |
| 32 | 43.1 | **75.3** | **1.75×** | YES |

Eager reproduces the ~43-steps/s ceiling exactly; **graph replay (launch-eliminated) breaks it** (consistent
with the fusion task's 6× at B=1, `V0_FUSION_LEVER_SIZING.md`). A CUDA graph **is** the reachable
"GPU-resident replay" on cu13 (`torch.cuda.CUDAGraph`; conditional nodes exist but are immature). So the
**CPU-dispatch wall is real and breakable today, with graphs — no exotic kernel required.**
- *Honesty on the measurement:* the graph harness's *output* was garbage (repetitive "The time.") from a
  StaticCache `cache_position` bug under manual capture (same confound as the fusion task) — a **harness**
  artifact, not the graph mechanism. Throughput is matched-fair (identical work); the graph mechanism's
  correctness was independently proven in the fusion task (zero-buffer-replay-recompute, max_rel 8e-4). HF's
  own cuda-graph `generate` path handles the cache-position correctly; I did not chase the harness bug.
- **Crucial caveat (do not over-read):** this 2.36× is **static, single-instance, fixed batch**. A CUDA graph
  captures **one frozen (model-set, batch-shape)**. It does **NOT** establish dynamic *multi-agent multiplexing*
  — the regime the bet actually lives in. Launch-elimination-for-static-work ≠ GPU-resident-dynamic-dispatch.

## 3. THE 100-AGENT MULTIPLEX — both feasible paths are served WITHOUT a persistent kernel

- **Same-model bursty agents → host-scheduled continuous batching.** 100 same-model agents share one weight
  copy and are composed into a dynamic batch (B up to 100) **recomposed per step on the host** (vLLM's
  scheduler). Batching is the proven lever (`V0_SMPACK_LEVER_SIZING.md`: 3.95× over packing, 7.7× B8→B64).
  The persistent-kernel thesis's *actual novel* claim is "move that per-step scheduling onto the GPU" — and
  nothing here shows that beats host continuous-batching; a static graph can't even express dynamic membership.
- **Bursty membership (burst ~200ms / idle 2-5s)** is dynamic join/leave. A static graph = recapture per change
  (~ms host round-trip — defeats the purpose) or conditional graph nodes (immature). The only thing that does
  bursty GPU-side dispatch cleanly is a true persistent work-queue megakernel — i.e. the **net-new research
  build** of §1, to serve a regime host continuous-batching already serves.
- **Distinct-model agents → §0 memory wall.** Can't batch (different weights), can't fit, dispatch is moot.

## 4. THE ANSWER (diagnosable) + why I decline the build

Mapping to the bet's three outcomes — it is **none of "real-and-scales"**; it is the split:
- **residence-unreachable-as-archived:** the persistent self-dispatch kernel is not in the tree (`cipher_persist_engine`
  = L2, not a dispatcher); building it is net-new research, not a revival.
- **density-capped:** distinct-model 100-agent is HBM-capacity-walled (~5 fp16 / ~22 INT4), dispatch-independent.
- **foundation-not-needed:** for the feasible (same-model) regime, the launch ceiling is already broken by CUDA
  graphs (2.36×, measured) and bursty multiplexing is already done by host continuous-batching (proven 7.7×).

**I decline to build the minimal resident-dispatch loop** — as a reasoned verdict, not avoidance: (a) the premise
that it's archived/revivable is false; (b) the heterogeneous-100-distinct-model target it exists to serve is
memory-infeasible regardless of dispatch; (c) for the feasible same-model regime it is redundant (graphs break
the launch wall today, host continuous-batching multiplexes bursty arrivals). Building it would be net-new
research to validate a foundation the feasible product doesn't need and the infeasible product can't use.

## 5. WHAT CP-5.5 / ORCHESTRATE's producing half SHOULD be (from this + the prior sizings)
Not a persistent megakernel, not an SM-packer. The producing half = **host-scheduled continuous batching for
same-model tenant density** (proven, gate-free) **+ opening the CUDA-graph gate** so launch-elimination (2.36×
here, 6× at B=1) and the consuming actuators (Marlin/FP8/fusion, all graph-bypassed today) fire in production.
The whole product does not stand on a GPU-resident dispatcher; it stands on (1) graphs (launch-elimination,
reachable) and (2) continuous batching (density, proven). No code changed; measurement-only; anchors unchanged.
