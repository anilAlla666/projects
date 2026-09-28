# V0 G-O1 ENGINE INCREMENT 1: CIPHER-owned graph-decode over the pager region (driver-level)

**2026-06-02. PROBE-FIRST → INCREMENT-1 CORRECTNESS PASS. NO CIPHER `.so` source change** (pure composition over
the existing driver-level pager substrate). Deployed `/usr/lib/cipher` (anchor, May 27) UNCHANGED; staging `.so`
(`cipher_rt_phase4` Jun-1 build) UNCHANGED — no rebuild, no rotation (the "staging .so rotates" discipline line
presupposes a code change; there is none, so OFF-byte-identical + anchor-unchanged hold **by construction**).
Artifacts: `cipher_engine.py` (deliverable module), `cipher_engine_gate.py` (gate), `pager_g1_capture_probe.py`
(isolation probe).

## Scope (Anil authorized increment 1 only; STOP after)

Engine moat = **driver-level multi-model multiplexing at the CUDA dispatch boundary** (what vLLM structurally can't
do: CuMemAllocator + cudagraph singletons block in-process composition). Increment 1 = the foundation:
**CIPHER-owned single-model static-KV graph-decode over the pager's cuMemMap weight region**, CIPHER owning the
allocator + the graph. vLLM NOT in the serving path. Composes two PROVEN bricks: the pager
([[cipher-pager-coresidence-build]], cuMemMap residency, KL=0, copy-free evict) + today's manual static-KV capture
([[cipher-go1-graphcapture-probe]], KL=0 real 8B). The int4 latency lever is RETIRED
([[cipher-go1-int4-decode-physics]], 1.36×) — the engine's value is the MUX/density, not int4 speed.

## Make-or-break (the advisor's gate, run clean)

"Capture composes with pager residency" ≠ "KL=0 while resident." The distinguishing claim of *this* increment is
that the **captured graph (VA baked in) survives the pager's unmap/remap**. Tested as 3 clean subprocesses (one
condition each, one capture, one monotonic burst — the page-cycle happens BEFORE any replay), greedy token sequence
vs eager-static (KL=0 = greedy match):

| mode | Llama-3.1-8B N=128 | TinyLlama N=64 | meaning |
|---|---|---|---|
| **resident** | **128/128 KL=0** | 64/64 KL=0 | graph-decode over pager-resident weights == eager |
| **pagecycle** (evict→restore→burst) | **128/128 KL=0** | 64/64 KL=0 | **captured graph SURVIVES the remap** |
| **negctrl** (zero 2 down_proj @ pager VA) | 2/128 DIVERGES | 5/64 DIVERGES | KL=0 is non-vacuous (graph reads LIVE pager mem) |

**Why it survives — isolated, not argued:** `cipher_pager_live_cksum` is **bit-identical before/after** the cycle
(`0x87032f0178f3deea`==, TinyLlama) and `base_va` is **stable** (`0x302000000`→same). page_in restores the warm
bytes to the same reserved VA; CUDA graphs bake in *VAs* (not physical), so replay reads identical memory → KL=0.
Both facts measured, not implied.

**The evict is PHYSICALLY REAL (not a no-op vacuously passing the cksum) — the unfakeable check:** even with the
live captured CUDA graph + torch MemPool holding the region, `state` transitions **RESIDENT(2)→EVICTED(0)→RESIDENT(2)**,
`evict_cnt`/`pagein_cnt` both increment 0→1, and `torch.cuda.mem_get_info()` free HBM **jumps by the full model
size on evict then drops back on restore**: **16.71 GB (Llama-3.1-8B)** / 2.50 GB (TinyLlama). So page_out releases
the entire model from HBM and page_in re-pins it — and the captured graph *still* replays 128/128 KL=0 across that
real free. (page_out's physical-free was proven standalone in [[cipher-pager-routing-build]]; this re-confirms it
under the new graph+MemPool interaction this increment introduces.) `REAL_EVICT=True` is a gate condition, not just
a print.

## Probe-first isolation log (scaffolding caught before any wall was costed)

The mechanism was fine throughout; the scaffolding lied 4 times (the arc's recurring pattern). Each isolated to root:
1. **Capture intermittently failed** (`cudaErrorStreamCaptureInvalidated`) — CIPHER's **legacy compute actuators**
   (Koopman/DET-SVD via **cusolver**, which is NOT capture-safe) fire nondeterministically during the captured
   forward, because `ctypes.CDLL(SO)` activates the whole runtime. **Sub-finding (real, deferred):** the engine
   serving path is pager + graph-decode, NOT those actuators → run with `CIPHER_RT_DISABLE_AUTO_INIT=1`
   (+`CIPHER_RT_DISABLE_CUINIT_HOOK=1`); Marlin already has its CAPTURE_SAFE path. `CUDA_LAUNCH_BLOCKING=1` *itself*
   perturbs this (makes the SVD fire in-capture) — not a debugging tool here.
2. **`run_replay` re-prefilled the cache** → an eager forward BETWEEN capture and replay corrupts the graph's
   private memory pool (CUDA-graph aliasing) → divergence/`index_copy_` OOB. Fix: no eager between capture & replay.
3. **Single-step replay non-deterministic** (max|Δ|≈4, argmax stable) — the `.clone()` between replays aliases the
   graph's private pool. The make-or-break "DIVERGES (Δ=8.2)" off this vehicle was a **false finding** (advisor
   caught it). Fix: the **monotonic burst** is the empirically-deterministic vehicle (only a tiny argmax alloc
   between replays); one burst per capture, one capture per condition. Recompared → KL=0.
4. (earlier arc) dangling-StaticCache use-after-free → the module now **co-owns** {region id, MemPool, StaticCache,
   CUDAGraph, static io} under one object lifetime (`PagerGraphModel`).

## Deliverable

`cipher_engine.py`: `CipherPager` (owns the `.so` pager handle + `CUDAPluggableAllocator`) + `PagerGraphModel`
(`load()`→pager region, `capture()`→static-KV decode graph, `serve_burst()`→monotonic replay, `evict()`/`restore()`
→ the residency hooks increment 3's router will drive). All lifetimes co-owned. `cipher_engine_gate.py` reproduces
the 3-mode correctness gate through the module on the real 8B (above). This is the engine's in-process foundation;
CIPHER owns the allocator + the graph, zero vLLM singleton in the path.

## Gate status (Mem #11, correctness FIRST)

- **Correctness PASS:** per-agent KL=0 (greedy) resident AND across evict→page_in on real 8B; negative control
  discriminates; isolation/false-finding caught. Single-model only (contention/multi-model = increments 2–3, NOT
  claimed; the harness reads VA directly, not via serve_begin/serve_end, so the ref-count machinery is NOT exercised
  here — pager eviction-during-use coherence stands on STEP-2: 13180 evicts, 0 corrupt).
- **Non-regression:** ZERO `.so`/pager source change → pager's proven gates unaffected by construction; the module
  gate independently re-confirms begin_load/page_out/page_in/live_cksum/get_stats in the current `.so`. (`test_pager`
  binary is stale/oddly-built `-shared` — a pre-existing artifact, not a regression.)
- **OFF byte-identical + deployed anchor (1f305ce6) UNCHANGED:** by construction (no `.so` change).
- **Operational:** every subprocess `os._exit`-reaped; GPU returns to 0 MiB between conditions (the orphaned-process
  lesson). Latency NOT measured/claimed here — increment 1 is correctness-only (per-step-overhead removal = inc 2).

## STOP — next-increment decision is Anil's

Increment 1 (CIPHER-owned graph-decode over the pager) is correctness-proven on the real 8B. NOT done (held scope,
NO scope-down): inc 2 = CIPHER-owned async dispatch loop (where vLLM's ~4–6ms/step overhead disappears; same-model
batched replay); inc 3 = N models co-resident, CIPHER owns the cudagraph capture → THE test of whether owning the
allocator+monitor dissolves vLLM's in-process wall, route + swap residency at M>capacity, per-agent KL=0 across swap;
inc 4 = the real-workload mock gate (Zipfian/gamma/growing-context) — agents-per-GPU at P99, CIPHER-owned end to end.
