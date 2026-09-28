# BUILD — ORCHESTRATE graph gate: CIPHER's actuator fires INSIDE a captured CUDA graph (G-O7), step 1

**Date:** 2026-05-31. **Type:** BUILD (source change, default-OFF, additive, OFF-path byte-identical; deployed
anchor unchanged). Staging `.so` rebuilt: `7cdc68ef` → `29686678` (cipher_rt_phase4). **One actuator (Marlin)
now fires inside a `torch.cuda.graph` capture and replays bit-identical to eager — single-GEMM AND a 2-GEMM
chain, stable across replays — via capture-SAFETY (not node-rewrite). The graph gate is MECHANISM-PROVEN in a
minimal torch graph (one actuator); vLLM end-to-end is the next op. The last four sizings' "graph bypasses
CIPHER" OUTCOME (Marlin=0 in production graphs) was real — but the CAUSE was misdiagnosed as replay-bypass; it
is capture-illegal per-call ops (the as-built actuator throws/falls-back during capture). Blocker now
understood and mechanically solved, not yet closed in production.**

---

## 0. THE RESULT — the gate works (graph_gate_probe.py, Mistral-shape bf16 GEMM, M=8/N=4096/K=4096)

| arm | capture | Marlin fired in capture | replay vs eager-Marlin | replay vs bf16 | verdict |
|---|---|---|---|---|---|
| OFF (`CIPHER_MARLIN_CAPTURE_SAFE` unset) | **FAILS** (`cudaErrorStreamCaptureInvalidated`) | (n/a) | — | — | byte-identical to pre-build |
| **ON** (`CIPHER_MARLIN_CAPTURE_SAFE=1`) | **SUCCEEDS** | **YES** (handled +4, bf16-sub +4 across the capture block) | **rel=0.000e+00 (bit-identical)** | rel=0.115 | **captured node IS Marlin-INT4** |

The three discriminators (advisor) all pass: (a) Marlin handled-delta>0 *across the capture block* (not warm,
not replay) → it substituted during capture; (b) replay of a **zeroed** output buffer recomputes == eager-Marlin
(rel 0) and ≠ bf16 (rel 0.115) → positively IDs Marlin as the captured node, not a cuBLAS false-green.

## 1. THE MECHANISM — capture-time substitution (B), NOT node-rewrite (A)

The prior framing was **right about the OUTCOME** (Marlin=0 in production graphs) but **wrong about the
CAUSE**: the probe shows the GOT-patch **fires during capture** (handled +4) — it is NOT bypassed by replay.
The real cause is that the as-built actuator does **three capture-illegal per-call operations** that throw
`cudaErrorStreamCaptureInvalidated` (or fall back to cuBLAS), so nothing CIPHER ends up in the captured graph:
1. **`cudaMalloc`/`cudaFree`** for the bf16↔fp16 cast temps (`cipher_rt_marlin_engine.cpp:1683-1697`).
2. **`PrimaryCtxGuard` (`cuCtxSetCurrent`) + per-call `cuEventCreate/Record/StreamWaitEvent`** in the fp16
   dispatch (`:1392-1411`, the F1 cross-context fix).
3. **`cudaMalloc`+`cudaMemset`** (sync) for the per-stream Marlin workspace on a never-seen stream
   (`marlin_ws_for_stream`, `:1041-1044`) — and torch's capture stream is always new.

So the gate is **make the actuator capture-safe** so its substituted kernel is cleanly recorded during
capture → replay fires it. No `cuGraphExecKernelNodeSetParams` node-rewrite needed (it remains the fallback if
a future actuator can't be made capture-safe).

## 2. THE CHANGE (all in `cipher_rt_marlin_engine.cpp`, behind `CIPHER_MARLIN_CAPTURE_SAFE`, default-OFF)

- **Cast alloc → stream-ordered** (`cudaMallocAsync`/`cudaFreeAsync` on the call stream): capture-legal *by
  design* (records as a graph mem node), **per-call so race-free under the N-stream multiplex** (chosen over a
  persistent buffer precisely because the persistent buffer races concurrent agents). Verified
  capture-legal on cu13/torch first (`cudamalloc_async_capture_probe.py`).
- **Skip `PrimaryCtxGuard` + completion-event** in capture-safe mode: inside a single-context captured graph
  (torch's context *is* the primary where Marlin's cubin lives), the GEMM CUfunction is already valid and the
  F1 cross-context race cannot occur (the C-consumer is a downstream graph node ordered by the buffer dep).
- **Workspace: reuse an already-allocated slot** (rekey) instead of `cudaMalloc` on the capture stream
  (requires Marlin to have run eagerly once before capture — standard CUDA-graph warmup).

OFF path: **output byte-identical** — one relaxed `g_capture_safe.load()` atomic-load + branch added per
dispatch (~1-2 ns; Mem #16 ABI-additive). Probe: OFF still fails capture on the unchanged `cudaMalloc`; eager
Marlin unchanged.

**Multi-call validation (advisor hardening — `graph_gate_probe2.py`):** a 2-GEMM chain (`linear(linear(x,W1),W2)`,
different weights, sharing the workspace) captured in ONE graph: capture succeeded, handled +2 (both GEMMs),
replay == eager-Marlin **rel 0.000** and ≠ bf16 (rel 0.170, compounded INT4), and **stable across two replays**
(0.000 both) — validating the workspace-rekey + 2nd-call-HIT + async-cast pool-reuse paths that the single-GEMM
probe didn't exercise (a real decode step is ~155 GEMMs sharing the workspace).

## 3. CORRECTNESS (Mem #11) — gate-correct, labeled honestly

The gate's correctness claim is **graph-Marlin == eager-Marlin (rel 0.000, bit-identical)** — i.e. the graph
path is *faithful to the actuator*, it does not corrupt the substituted output. This is **not** a claim that
Marlin-INT4 == bf16 (it doesn't; rel 0.115 — that is Marlin's separate, already-failed quality gate, and is
exactly what lets us positively ID the kernel). The gate is actuator-agnostic: once capture-safe, ANY actuator
(incl. a future correct one) fires in production graphs.

## 4. SCOPE — what is proven vs next (one op at a time)

- **PROVEN:** one actuator (Marlin), single-stream, fires inside a captured graph, bit-identical to eager,
  default-OFF, OFF byte-identical. The "graph-bypass" blocker is closed at the mechanism level.
- **NEXT (each its own op, NOT claimed here):** (a) **multi-agent / N-concurrent-stream** — the workspace-reuse
  rekey has a known concurrency race (two streams rekeying one buffer); the cast-alloc is already race-free;
  the multiplex needs ≥N pre-warmed distinct workspaces + no-evict-of-in-use (flagged in code, v-next). (b)
  **FP8 / fusion** actuators through the same capture-safe path. (c) **end-to-end vLLM** (ride vLLM's
  bucketed-graph capture with CIPHER armed). (d) throughput under graph (Marlin itself regresses per G3; the
  gate's value is that a *good* actuator now reaches production graphs).

## 5. ANCHORS / discipline
Deployed `/usr/lib/cipher` UNCHANGED. Staging `.so 7cdc68ef → 29686678`. Source change: `cipher_rt_marlin_engine.cpp`
only (the uncommitted `cipher_rt_cublas_shim.c` + `cipher_rt_fairness.*` are pre-existing D.8 strays, excluded
from this commit as in the geom-attn close). Default-OFF, additive, tagged at close. Throwaway probes
(graph_gate_probe.py, cudamalloc_async_capture_probe.py) are not committed to the runtime.
