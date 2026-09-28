# G-O1 engine increment-1 PROBE: manual static-KV CUDA-graph capture is FEASIBLE + KL=0 (TinyLlama) — root-caused; 8B replay asserts on an un-isolated index_copy (NOT my bounds, NOT a costed architecture wall). Mechanism de-risked, engine NOT built.

**Date:** 2026-06-02. **Type:** PROBE-FIRST (the riskiest mechanism, before the months-grind). NO CIPHER source
change (normal allocator, clean torch+transformers; the `.so` is not involved). Harnesses:
`g1_graphdecode_capture_probe.py`, `g1_capture_isolation.py`, `g1_graphdecode_latency.py`. **Anchors UNCHANGED:
deployed `1f305ce6`, staging `2edba0d2`.** vLLM = validator; this probes the CIPHER-owned manual capture the engine
needs (vLLM's cudagraph-monitor singleton blocks in-process multi-model graph-decode → CIPHER must own capture).

**The question (the make-or-break for the whole build): can CIPHER hand-roll static-KV `torch.cuda.graph` capture of
an HF decode and replay it KL=0?** (The transformers `torch.compile`/CUDAGraph-trees path FAILED on an aliasing bug.)

## RESULT 1 (earned) — FEASIBLE + KL=0 on TinyLlama; the initial failure was WARM-UP POLLUTION
- First attempt diverged at token 1 (graph vs eager-static 2/64). **Isolation (`g1_capture_isolation.py`) root-caused
  it:** the 3 capture warm-ups ran at `cache_position=P` on the **real prefilled StaticCache**, advancing its state so
  the captured attention mask was wrong. **Fix: warm up on a THROWAWAY cache (or skip warm-up on the real cache).**
- With the fix, **manual static-KV `torch.cuda.graph` capture + replay = greedy-match KL=0 (48/48) vs eager-static**
  on TinyLlama, under BOTH `none` and `throwaway` warm-up. `eager-static vs eager-dynamic = 64/64` confirms StaticCache
  itself is faithful, so the 48/48 isolates the **graph capture** as correctness-preserving.
- **This de-risks the riskiest mechanism: capturing an HF decode in a raw CUDA graph IS correctness-preserving** (the
  "vLLM's hardest core" fear is reduced for the simple-architecture case). It is NOT the engine.

## RESULT 2 (honest, un-isolated) — Llama-3.1-8B: capture succeeds, REPLAY asserts; cause NOT isolated
- `build_graph` (capture) succeeds on the 8B; the **replay** device-asserts: `IndexKernel.cu:193 index_copy_(): index
  out of bounds`.
- **Ruled out my harness bounds bug:** P=17, N=64, max_cache_len=145, **max replay `cache_position`=79 ≪ 145** — and
  generous headroom did not change it. So the out-of-range index is **NOT** the cache_position I drive; it is an
  `index_copy_` **inside the model/cache forward** (assert thread pattern `[0..32+]` hints a head/GQA-dim index —
  Llama-3.1 has num_kv_heads=8 vs num_attention_heads=32 — but this is a **hint, not an isolation**).
- **What this is and is NOT:** it is a real "manual capture is not turnkey across architectures" signal — the 8B has
  an un-isolated `index_copy` OOB under capture-replay. It is **NOT** a simple harness bounds bug (ruled out), and it
  is **NOT** a costed "architecture capture-safety = months wall" — I did not isolate the cause and will not convert
  "didn't diagnose it" into a costed verdict. Resolving per-model capture robustness is real engineering of unknown
  size; that's all the data supports.

## The latency gate is BLOCKED (not measured) for the regime that needs it
Increment-1's latency gate — "does graph-decode take the 128-tok burst from eager's 5.4s toward sub-second" — is for
the BIG-model long-burst regime. That measurement is **blocked by the un-isolated 8B capture-replay assert**. The
principled expectation stands (graph-decode removes per-token launch gaps → toward the ~74%-peak bandwidth floor → the
overhead-dominated eager latency drops; overlap was REFUTED, graph-decode is the lever), but the big-model number is
**pending the 8B capture-robustness fix**, not delivered here. (TinyLlama works but is not the long-burst regime.)

## Verdict — mechanism de-risked, engine NOT built; the commitment is Anil's
- **Earned:** CIPHER-owned manual static-KV CUDA-graph capture is feasible + KL=0 on a real HF model (TinyLlama);
  the original token-1 failure is root-caused (warm-up pollution) and fixed.
- **Open:** the 8B replay asserts on an un-isolated in-forward `index_copy` (not my bounds); per-model capture
  robustness is unbuilt work of un-costed size; the big-model latency payoff is therefore unmeasured.
- **Frame:** this is "mechanism feasible, de-risked, NOT built" — a green single-model capture probe does NOT mean
  "engine works." The engine is still a real multi-increment build (8B capture robustness → async batching →
  multi-model swap over the pager → the full both-regimes gate), and the months commitment is **Anil's**.
- **Op note:** CUDA device-side asserts (out-of-vocab argmax; in-forward index) poison the context and cascade —
  clamp fed-back token ids, isolate per-variant, and the eventual engine must reap subprocesses on shutdown.
No source change; clean torch/transformers; anchors unchanged. **STOP for the next-increment decision.**
[[cipher-go1-engine-probe-increment1]], [[cipher-go1-dispatch-router]], [[cipher-graph-gate-build]], [[cipher-go1-inprocess-composition]].
