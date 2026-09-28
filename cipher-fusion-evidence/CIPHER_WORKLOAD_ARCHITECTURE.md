# CIPHER — WORKLOAD-DRIVEN ARCHITECTURE & RE-ARCHITECTURE PLAN

**Date:** 2026-05-19. **Purpose:** the architecture layer the campaign never
wrote — map LLM-inference workload classes to their bottlenecks, then scope
CIPHER's levers to the class CIPHER should actually serve. Grounded in current
(2025–26) industry sources, cited inline.

---

## PART 1 — LLM INFERENCE WORKLOAD TAXONOMY

LLM inference is **two phases with opposite bottlenecks** — this is the single
most important fact, and the campaign never built around it.

| phase | bottleneck | GPU utilization | right metric | why |
|---|---|---|---|---|
| **Prefill** | **compute-bound** | 90–95 % | **MFU** | whole prompt processed in one parallel forward pass — a matmul-heavy op that saturates the SMs |
| **Decode** | **memory-bandwidth-bound** | 20–40 % | **MBU** (mem-bw util), *not MFU* | one token at a time; each step streams all weights + KV from HBM; low arithmetic intensity |

UT Austin measured it: cutting memory bandwidth 40 % raised *prefill* latency
only 17 %; cutting compute 50 % raised *decode* latency only 22 %. The two
phases use **different hardware resources** — which is why the industry has
moved to **disaggregated serving** (prefill and decode on separate workers).
[1]

**This vindicates the core insight: MFU is the prefill metric. For decode —
and therefore for agentic workloads — MFU is the wrong metric. MBU is right.**
"10 % vs 40 % MFU = profitable vs bleeding money" is a *prefill/training*
statement. [4]

### The four workload classes CIPHER could serve

| class | shape | dominant bottleneck | customer metric | CIPHER's fit |
|---|---|---|---|---|
| **A. Interactive single-stream** | one user, prefill + decode | decode memory-bound; TTFT+ITL latency | latency | weak — single stream, nothing to consolidate |
| **B. Batch / offline throughput** | huge batches | compute-bound (big batch lifts arithmetic intensity) | **MFU**, throughput | weak — vLLM/TRT-LLM already own this; Marlin's regime |
| **C. Multi-tenant serving** | N independent models/users | KV-cache pressure, noisy-neighbour contention | tenants/GPU, fleet tok/W, p99 | **strong** |
| **D. Agentic multi-tenant** | N agents, each a loop of prefill→short-decode→tool-call | KV pressure + prefix recompute + GPU idle during tool calls + noisy neighbour | **agents/GPU, cost/agent, per-agent p99** | **strongest — this is CIPHER's class** |

### Class D — agentic — properties, bottlenecks, problems (the target)

The agent loop, per iteration: **prefill** a large context dominated by a
**shared system-prompt + tool-definitions prefix (often 1000+ tokens)** →
**short decode** (reasoning + a tool call) → **tool execution** (external,
GPU idle) → repeat. [2]

The bottlenecks, from the literature:

1. **Shared-prefix recomputation.** Every agent, every iteration, starts with
   the same system prompt + tool schema. Without prefix reuse it is
   re-prefilled every time. SGLang's RadixAttention caches it *within one
   process*. [2]
2. **KV-cache pressure.** Many agents × long contexts contend for HBM; KV
   eviction of a critical agent stalls it. Multi-tenant systems (Oneiros,
   Tokencake) fight this with parameter remapping and dynamic KV partitioning.
   [3]
3. **GPU idle during tool calls.** Tool execution is **30–80 % of iteration
   latency** and individual tool calls can exceed prefill time — the GPU sits
   idle while one agent waits on a tool. [2]
4. **Noisy neighbour.** Agents contending for SMs / memory bandwidth give each
   other unpredictable latency. [3]
5. **Bursty, heterogeneous** control flow — agents are not uniform streams.

**For class D the customer never sees MFU.** They see: how many agents fit per
GPU, cost per agent-hour, and per-agent tail latency. That is the metric set.

---

## PART 2 — CIPHER'S LEVERS, MAPPED TO CLASS-D BOTTLENECKS

The campaign built the right pieces — it just never mapped them to a class.
Here is the map:

| Class-D bottleneck | CIPHER lever that addresses it | built? | status |
|---|---|---|---|
| 1. shared-prefix recompute | **cross-tenant KV-prefix dedup** — the cross-*process* analogue of RadixAttention (which is intra-process only) | yes — kmod `cipher_kvdedup` + `cipher_rt_kv_alloc` (T4.6.3/4) | verified on synthesized pages; **never wired to live decode** (T4.6.5 not done) |
| 2. KV-cache pressure | weight sharing frees HBM for KV; KV dedup shrinks KV itself | yes — Track 2 (weight arena), T4.6 (dedup) | weight sharing closed (76 %); dedup live-wiring missing |
| 3. GPU idle during tool calls | SM arbitration + DSM — reallocate an idle agent's SMs to a busy one; cross-tenant batching fuses concurrent agents' decode steps | yes — CP 5.4 ledger, Track 3 DSM, Phase B batching | substrate built; **not architected as tool-idle-filling** |
| 4. noisy neighbour | SM partitioning — hard spatial isolation per agent | yes — CP 5.4 green-context partitions | built; isolation contention-gated (1.6B-4) |
| 5. density (agents/GPU) | cross-tenant weight sharing — one weight copy for N agents | yes — Track 2 | **closed, 76 % measured — the headline** |
| per-agent metering / billing / fairness | observability ops — FAIRNESS, CARBON, RECEIPT, telemetry | yes — overlay ops + `/proc/cipher` | detectors work; billing not productized |

**Every Class-D bottleneck already has a CIPHER lever.** That is the good
news the campaign's lack of architecture obscured.

---

## PART 3 — THE ARCHITECTURE GAP

What went wrong was not the engineering. It was three architectural omissions:

1. **No workload class was ever declared.** CIPHER was built as a pile of
   actuators and pitched as universal ("lifts MFU + TPW + density + TPS for
   any workload"). Every lever is regime-specific (Part 1); claimed
   universally, they produced the contradictions and retractions.
2. **CIPHER never engaged the industry architecture.** Disaggregated
   prefill/decode serving and prefix caching (RadixAttention) are the 2025–26
   standard. CIPHER built generic SM slicing and generic KV dedup without
   positioning them against — or *as* — these patterns.
3. **The metric was mis-chosen.** MFU was carried as a CIPHER goal. MFU is a
   prefill/batch metric. For Class D it is meaningless. The right metrics —
   agents/GPU, fleet tok/W, MBU, p99 — were never made the campaign's gates.

The fix is not more building. It is **declaring the class and re-positioning
the existing levers against it.**

---

## PART 4 — THE RE-ARCHITECTURE PLAN

### 4.1 — Declare CIPHER v1

> **CIPHER v1 is a cross-tenant GPU substrate for agentic multi-tenant
> inference (Class D). It sits beneath an unmodified inference engine (vLLM /
> SGLang) and does what a single-process engine structurally cannot: share
> weights and KV-prefixes *across* tenant processes, arbitrate SMs between
> agents, and consolidate a fleet of agents onto one GPU. Its metrics are
> agents/GPU, fleet tok/W, and per-agent p99 latency. It does not accelerate a
> single stream and does not target MFU.**

### 4.2 — Re-position the levers (architecture, not new code)

- **Cross-tenant KV-prefix dedup → the headline agentic lever.** Reframe it as
  "RadixAttention across processes." SGLang dedups the shared system-prompt
  prefix within one server; CIPHER dedups it across *every tenant process* on
  the GPU. For 100 agents sharing a 4 K-token system prompt, that prefix KV is
  stored **once**.
- **SM partitioning → dual-purpose:** (a) per-agent noisy-neighbour isolation,
  and (b) **prefill/decode-aware allocation** — give a prefilling agent a wide
  SM slice (compute-bound), a decoding agent a narrow one (memory-bound). This
  aligns CIPHER with disaggregated serving — *on one GPU*.
- **DSM + SM arbitration → tool-call-idle filling.** When an agent blocks on a
  tool call, its SMs migrate to agents that are computing. This directly
  attacks bottleneck 3 (30–80 % idle) — a Class-D-specific value proposition.
- **Weight sharing → density**, the capacity headline (Class-D metric:
  agents/GPU).
- **Observability → per-agent billing/fairness**, productized.

### 4.3 — What to BUILD (engineering, scoped to Class D)

| # | item | why | rough size |
|---|---|---|---|
| B1 | **live-wire KV-prefix dedup** — connect `cipher_kvdedup` to the live decode path; finish T4.6.5 (real-trace validation) | the headline agentic lever is built but unproven on real KV | medium |
| B2 | **weight sharing under vLLM** (FUTURE_SCOPE/A Phase 4) | density composed with the real engine | medium-large |
| B3 | **prefill/decode-aware + tool-idle-aware SM scheduler** | turns the SM ledger from generic slicing into the Class-D scheduler | medium |
| B4 | **agentic benchmark harness** — N agents, shared prefix, tool-loop, bursty arrivals | CP 5.5 done right; metrics = agents/GPU, fleet tok/W, MBU, p99 | medium |

### 4.4 — What to DROP (out of Class-D scope)

- **MFU** as a CIPHER metric or claim — it is a prefill/batch metric.
- **Goal-4 Koopman substitution** — unbuilt, a research bet, not Class D.
- **Marlin for decode** — B≥8 / compute-bound; that is Class B, not ours.
- **Single-stream acceleration** ambition — Class A; CIPHER is transparent
  there by design, not an accelerator.

### 4.5 — Sequence

1. **This document** — adjudicate the Class-D declaration + drop list.
2. **B2 (weight sharing under vLLM)** — already the FUTURE_SCOPE/A Phase 4
   critical path; the density headline.
3. **B1 (live KV-prefix dedup)** — the second agentic lever; obtain a real
   agentic trace for T4.6.5.
4. **B3 (Class-D scheduler)** — re-purpose the CP 5.4 ledger + Track 3 DSM.
5. **B4 + CP 5.5** — the agentic benchmark: CIPHER+vLLM vs vLLM-alone, real
   agentic workload, on the Class-D metric set. **The fundable demo.**

---

## SOURCES

1. *Prefill is Compute-Bound, Decode is Memory-Bound* — Towards Data Science;
   UT Austin disaggregation result (arXiv 2602.02987).
2. *Sutradhara* (arXiv 2601.12967), *Efficient LLM Serving for Agentic
   Workflows* (arXiv 2603.16104), SGLang RadixAttention — agentic shared
   prefix, tool-call latency dominance.
3. *Oneiros / MIRAGE* (arXiv 2507.11507), *Tokencake* (arXiv 2510.18586) —
   multi-tenant KV-cache pressure, noisy-neighbour isolation.
4. *Using Model FLOPs Utilization (MFU)* (Better ML); NVIDIA *Mastering LLM
   Techniques: Inference Optimization* — MFU vs MBU, decode utilization.
