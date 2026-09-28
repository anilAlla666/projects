# G-O1 8B graph-replay assert ROOT-CAUSED: a dangling-StaticCache use-after-free in my harness (NOT GQA/architecture). Manual capture is KL=0 on the real 8B (128/128); the graph-decode latency lever is real + measured (2.93×, now bandwidth-bound).

**Date:** 2026-06-02. **Type:** read-only isolation + latency measurement (the only "fix" was my PROBE harness's
cache lifetime — NO CIPHER source change; clean torch/transformers). Harnesses: `g1_capture_min.py`,
`g1_graphdecode_latency_v2.py`. **Anchors UNCHANGED: `1f305ce6` / `2edba0d2`.** vLLM = validator.

## ROOT CAUSE (isolated, earned) — a DANGLING StaticCache, not architecture
The prior latency harness's `build_graph()` created the `StaticCache` as a **local variable and did not return it**;
after the function returned, Python **GC'd the cache → its KV buffers were freed/reused**, so the captured graph's
`index_copy_` wrote into **freed memory → `index_copy_(): index out of bounds` on replay** (a use-after-free, not a
`cache_position` value bug — positions were in-bounds: P=17, max replay pos 79 ≪ max_cache_len 145).
- **Proof:** the clean minimal harness (`g1_capture_min.py`, cache **retained in scope**) captures **KL=0 128/128 on
  Llama-3.1-8B**, KL=0 128/128 on Qwen2-7B, KL=0 48/48 on Llama-3.2-1B. The **only delta** vs the asserting harness was
  the cache lifetime. Fix = keep the cache alive.

## GQA REFUTED as the axis
The earlier "assert thread-pattern hints a head/GQA-dim index" was a wrong guess: **TinyLlama is ALSO GQA**
(32 attn / 4 KV heads) and captured KL=0; **every** tested model (all GQA here) captures KL=0 in the clean harness.
GQA is not the differentiator — there was no architecture differentiator; it was the harness.

## META-LESSON (worth recording)
Two turns ago I nearly shipped "GQA / per-architecture capture-safety / months wall" off an **un-isolated** assert —
and it was a **use-after-free in my own scaffolding.** This is the **third harness artifact in this engine arc**
(orphaned vLLM EngineCores holding 48 GiB → false OOM; prefix-cache elision → falsely-favorable latency; now a
dangling StaticCache → false "architecture wall"). **The mechanism kept being fine; the scaffolding kept lying.**
Discipline confirmed: isolate to root (exact op + a clean repro) before costing anything as a wall.

## LATENCY PAYOFF (the number increment-1 wanted) — MEASURED on the 8B
Llama-3.1-8B, solo single-burst, cache retained, **KL=0**:
| burst | eager | graph-decode | speedup | per-token (eager → graph) |
|---|---:|---:|---:|---|
| 16-tok | 810 ms | 132 ms | **6.16×** | 50.6 → 8.2 ms |
| 128-tok | 3085 ms | **1052 ms** | **2.93×** | 24.1 → 8.2 ms |

- **Graph-decode is the lever, and it works:** per-token drops to a stable **8.2 ms**, which is **near the fp16-8B
  HBM bandwidth floor** (16 GB / ~2.5 TB/s ≈ 6.4 ms) — i.e. graph capture **removed the per-token launch gaps exactly
  as diagnosed** (overlap was refuted; this is the right lever), and the residual is now HBM physics, not overhead.
- eager 128-tok (3.1 s) ~matches the async-router's 5.4 s p99 long-burst regime; **graph-decode takes it to ~1.05 s**.

## Sub-second framing (honest: measured floor vs projected path)
- **Measured:** a solo **fp16** 128-tok 8B burst with graph-decode = **~1.05 s — just OVER sub-second**, and it is
  **bandwidth-floored** (8.2 ms/tok ≈ the 6.4 ms physics floor); graph-decode alone at fp16 cannot go materially lower.
- **Projected (NOT measured):** **int4** (4× less weight-bandwidth — the pager's regime) projects to
  ~128 × (4 GB / 2.5 TB/s) ≈ **~205 ms — sub-second.** Labeled a bandwidth-floor projection, not a measurement
  (measuring needs Marlin-int4 capture; bnb-NF4's per-token dequant is a fragile/slow capture and its decision value
  is already delivered by the fp16 result).
- **Concurrency does NOT reduce single-burst latency:** same-model batching amortizes weight-reads → it raises
  *aggregate throughput at fixed latency*, but one agent's 128-tok burst is still ≥ per-token-bandwidth × 128. So the
  honest path to a sub-second long-burst is **graph-decode + int4**, not batching.

## Verdict — increment-1's gate ANSWERED; the rest is the build (Anil's call)
- Manual static-KV CUDA-graph capture is **feasible + KL=0 on the real 8B** (128/128), not just TinyLlama; the 8B
  assert was a **harness use-after-free** (cheap fix), **not** a per-architecture cost.
- The **graph-decode latency lever is real and measured** (2.93× at 128-tok; per-token now bandwidth-bound at the
  fp16 floor). fp16 long-burst → ~1 s (measured); int4 → sub-second (projected).
- The remaining path is the multi-increment engine: **int4 (Marlin, not bnb) capture → async batching in CIPHER's
  loop → multi-model swap over the pager (does CIPHER owning the cudagraph monitor dissolve the singleton wall?) →
  the full both-regimes gate.** The months commitment is **Anil's.**
No CIPHER source change; read-only probe; anchors unchanged. **STOP for the next-increment decision.**
[[cipher-go1-graphcapture-probe]], [[cipher-go1-engine-probe-increment1]], [[cipher-go1-dispatch-router]], [[cipher-graph-gate-build]].
