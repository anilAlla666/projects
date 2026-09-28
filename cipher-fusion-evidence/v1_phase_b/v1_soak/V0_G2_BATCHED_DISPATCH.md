# V0 G-O1 ENGINE INCREMENT 2: CIPHER-owned batched dispatch loop over the pager (same-model coalescing)

**2026-06-03. CORRECTNESS PASS (Anil-ratified gate). NO CIPHER `.so` source change** (additive Python over the inc-1
pager+graph-decode bricks). Deployed anchor 1f305ce6 + staging `.so` (Jun-1) UNCHANGED → OFF-byte-identical +
anchor-unchanged BY CONSTRUCTION. inc-1 module (`cipher_engine.py`) UNCHANGED, its gate still PASSES (non-regression).
Artifacts: `cipher_engine_batched.py` (WaveServer), `cipher_sched.py` (reproducible scheduler), probes
`pager_g2_{batched,cotenant,multiwave,latency_decomp}_probe.py`. Run with `CIPHER_RT_DISABLE_AUTO_INIT=1`.

## What was built

A CIPHER-owned dispatch loop that **coalesces concurrently-pending same-model requests into one lockstep batched
captured-graph replay** over the pager-resident model. This is CIPHER's scheduling policy at the driver boundary —
NOT vLLM's engine (vLLM absent from the serving path). It is the batched-replay primitive inc-3's multi-model router
will ride; **standalone it is same-model batching, which is vLLM's core turf** (see honest framing below).

## Correctness — the Anil-ratified gate (no-contamination + greedy-modulo-near-ties)

Exact per-agent greedy-match-vs-solo is FP-brittle at B>1 (no batched engine, vLLM included, gives bit-identical
greedy across batch sizes). Anil ratified the gate as: **(a) zero cross-row contamination + (b) greedy-match-vs-solo
except sub-delta near-tie flips.**
- **(a) ZERO contamination — co-tenant invariance (the decisive, greedy-immune test):** an agent X's first-decode-step
  logits are **bit-identical (max|Δ|=0.000)** whether its batch-mates change or its batch index changes, on BOTH
  TinyLlama AND real Llama-3.1-8B. Batched attention is perfectly per-row isolated. The solo-vs-batched delta is a
  fixed ~0.0156 (TinyLlama)/~0.02 (8B) — the batch-shape GEMM/reduction-tree difference, co-tenant-INDEPENDENT.
- **(b) greedy-modulo-near-ties:** batched-burst vs solo is exact at B≤4; higher-B divergences are CHARACTERIZED as
  sub-delta near-tie flips (observed: B=8 agent7 step52, solo margin 0.0078 < 0.0156 delta → the fixed kernel delta
  flips a near-tie; deterministic, prompt-specific, NOT contamination). Classifier: benign iff solo top1-top2 margin
  at the first diff < TIE(0.05).
- **Scheduler correctness:** the NEW failure surface is slot↔agent routing bookkeeping, not the model. Per-agent
  output-vs-solo PASS (real 8B sample exact=8/8; TinyLlama exact+near-tie, 0 FAULT). **Routing negative control:**
  feeding agent-A's served output against a different-prompt agent's solo is DETECTED as FAULT → the check
  discriminates routing.

## Scheduler mechanism (probe-first, isolated)

- One captured graph CANNOT be reused across waves by eager re-prefill (corrupts the graph's private pool; probe v1:
  total divergence @0). Each wave **RE-CAPTURES** after its prefill (the proven inc-1 prefill→capture→burst, batched).
  Steady-state capture ~17ms/wave (TinyLlama) ≈ **13% of GPU time**; first capture dearer (warmup).
- **Reproducible, NOT live-asyncio:** a deterministic trace-driven POLICY simulator — gamma arrival trace (fixed
  numpy seed), deterministic event order, virtual clock advanced by a MODELED wave occupancy (coalescing exactly
  reproducible, run A==run B), with REAL GPU replays timed for throughput/latency. No async prefill/decode overlap is
  built or claimed.

## Results — measured, honestly framed

**Coalescing is LOAD-dependent (the headline, not "batched is faster when batched"):** sweeping arrival rate (TinyLlama,
BMAX=8, gamma burst=0.5):

| arrival spacing | mean wave | realized fill | speedup-vs-serial |
|---|---|---|---|
| 0.02s (heavy) | 6.86/8 | 0.86 | 4.47× |
| 0.2s (medium) | 1.78/8 | 0.22 | 1.47× |
| 0.6s (light) | 1.14/8 | 0.14 | 1.00× |
| 1.5s (very light) | 1.07/8 | 0.13 | 0.94× |

Under heavy concurrent same-model load coalescing is high; under light/spread load agents rarely overlap → waves of 1
→ no benefit, and at very light load the per-wave capture overhead makes it slightly **negative**. Real 8B run:
mean wave 3.00/4, 2.84× vs serial, per-agent correct, routing-nc detected.

**Throughput speedup decomposed (not pure amortization):** the "4.47× vs serial(B=1)" bundles (i) fewer re-captures
(serial pays ~48, batched ~8) with (ii) weight-read amortization. The CLEAN batching number is the B-sweep: ~3.5×
at B=8 (225→793 tok/s aggregate). Report both; the GEMM amortization alone is ~3.5×, not 4.47×.

**Per-step latency is HOST-OVERHEAD-dominated, NOT bandwidth-floored — and this REFUTES the inc-2 premise.** TinyLlama
weight floor ≈0.88ms/step; measured per-step ≈ flat 4.45 (B=1) → 5.86ms (B=8). The ~4ms fixed excess is constant
across model size (8B: 12.23ms measured vs 6.4ms floor → ~5.8ms excess) = the signature of host-side per-step
`torch.cuda.synchronize()` + CPU argmax/clamp/`.copy_`, NOT HBM physics. So CIPHER's host-driven greedy loop
**reintroduces ~4–6ms/step of its own overhead — the same order as the vLLM tax it set out to remove.** Matched B=1
on 8B: **vLLM 6.25ms/step (≈ its floor, overhead pipelined/GPU-sampled) vs CIPHER ≈12ms/step (floor + host overhead)**
→ CIPHER does NOT beat vLLM per-step. Removing this needs GPU-side sampling + pipelining (vLLM's approach), NOT built
here. Honest: **batching buys aggregate THROUGHPUT under load; per-agent latency is worse than vLLM, not better.**

**Lockstep waste:** within a wave, shorter generations finish early → their slots burn compute = 27% slot-steps
(variable gen 24–64). Bounds realizable amortization.

## Boundary (diagnosed, not scope-down): no mid-flight join

HF `StaticCache.index_copy_(2, cache_position, …)` is **batch-shared** (source-confirmed) → no per-row position. A
wave batches agents AT THE SAME sequence position; left-padding admits heterogeneous-length prompts that START
together, but a new arrival joining a wave already in flight at a different position needs **paged attention = vLLM's
core = inc-3+ / arguably not CIPHER's job**. This is the lockstep limit; the realistic mock here uses common-length
burst starts (variable-length batching via left-pad is the noted extension).

## Honest framing (do NOT overstate)

Same-model batching is vLLM's core turf, and CIPHER's lockstep version is strictly **worse standalone**: no mid-flight
join, re-capture-per-wave cost (~13% GPU, grows with model size / shrinks with burst length), 27% lockstep waste, and
~4–6ms/step host overhead. **Its value is as a CORRECTNESS-VALIDATED coalescing primitive** with quantified overheads
and load-dependent throughput (4.5×→1×) — **the batched-replay building block that inc-3's in-process multi-model
router rides**, which is where CIPHER does what vLLM cannot (N distinct models, one process, the singleton wall).
NOT "CIPHER's loop beats vLLM's overhead."

## Gate status (Mem #11)

1. **Correctness PASS (ratified):** zero contamination (dΔ=0 both models) + greedy-modulo-near-ties + routing
   negative control DETECTED. Real 8B + TinyLlama.
2. **Throughput-vs-B:** ~3.5× clean (B-sweep); coalescing 4.5×→1× load-dependent. **Per-step latency MEASURED vs vLLM
   baseline: host-overhead-dominated, ~2× SLOWER than vLLM at matched B=1 (premise refuted, reported honestly).**
3. **NON-REGRESSION:** inc-1 module unchanged, inc-1 gate (resident/pagecycle/negctrl) PASSES; pager unaffected.
4. **OFF byte-identical + anchor 1f305ce6 UNCHANGED:** by construction (no `.so` change). Subprocesses os._exit-reaped;
   GPU→0MiB.

## STOP — next is increment 3 (Anil's call)

Multi-model co-residence: N distinct models in the pager, CIPHER owns the cudagraph capture → **the singleton-wall
test** (does CIPHER owning the allocator+monitor dissolve vLLM's in-process-composition wall?), route + swap residency
at M>capacity, per-agent KL=0 across swap. THAT is where this primitive earns its keep.
