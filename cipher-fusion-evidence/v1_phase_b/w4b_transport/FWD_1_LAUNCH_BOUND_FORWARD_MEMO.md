# FWD-1 — launch-bound decode-forward CUDA-graph probe — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO — awaiting Anil approval before build
(design-memo → approve → build; one atomic substep; wait between).
**Provenance:** W.4b.7 item 1 surfaced the real ceiling-limiter — the per-token decode
**forward is kernel-launch-latency bound** (~52 ms/step for B=4 Mistral-7B, *identical*
at 1980 MHz and 1005 MHz ⇒ clock-independent). W.4b closed on branch A 2026-05-29
(`w4b-close` → `dbb7dd9`); Anil directed the **forward axis as the next substep, ahead
of W.7 NCCL** (deferred). This probe is **orthogonal to branch A/B** — cutting the
forward lifts branch A's absolute tok/s *and* bounds any future R-4.

**Anchors this probe will NOT touch:** cipher_rt_phase4 `8b5e928` / cipher_kmod
`02fc2d1` / cipher_kv_bridge `5a3db034`. This is a **userspace measurement probe**
(stock HF + torch CUDA-graph path) to decide whether the forward is reducible. If the
lever proves real, *productionizing* it through the substrate actuators is a SEPARATE,
later substep with its own memo (and only then a possible anchor change).

---

## §0 — Why this, why now

Branch A's density gate cleared at 3.239× (W.4b.6), but the **absolute** throughput is
poor: B=4 Mistral-7B decodes at ~72–76 tok/s (~52 ms/step). A 14 GB-weight read is a
~4 ms bandwidth floor, so the forward is **~10× over** what a bandwidth-bound decode
would cost. vLLM-class decode (CUDA graphs + static KV) runs Mistral B=4 at single-digit
ms/step. **If the forward is launch-overhead bound, a CUDA-graph replay should collapse
the inter-kernel bubbles and cut the step several-fold** — multiplying branch A's
delivered tok/s and tok/W with no new transport, no branch-B risk.

## §1 — The finding being chased (grounded)

From `W4B_7_FINDINGS.md`: the executor `gpu_forward` **cuda-Event** span (pure GPU
timeline, no Python) is 50.9 ms/step at B=4. Clock-independence (52 ms @ 1980 MHz ==
51 ms @ 1005 MHz) rules out compute/clock binding. Two candidates remain:
- **(a) launch-overhead bound** — 32 layers × many tiny kernels; the GPU timeline has
  idle bubbles between launches (CPU can't enqueue fast enough). CUDA-graph replay
  (one launch for the whole graph) removes the bubbles. **This is the hypothesis.**
- **(b) memory-latency bound** — each tiny kernel is itself latency-bound on HBM round
  trips that graphs won't fix. If so, the probe shows ~no improvement and the answer is
  "the ceiling is real at this batch; the lever is larger B / fusion, not graphs."

The probe is designed to **distinguish (a) from (b)** — either outcome is a result.

## §2 — The probe (stock toolchain; verified available here)

torch 2.11.0+cu130, transformers 5.8.1: `StaticCache`, `torch.cuda.CUDAGraph`, and
`torch.compile(mode="reduce-overhead")` all present. Standard static-shape CUDA-graph
decode path:
1. **Static KV** — drive the decode with HF `StaticCache` (fixed max length) so KV
   tensor addresses/shapes are stable across steps (the prerequisite for graph capture;
   the growing-cache/growing-mask is exactly why naive capture fails — §8 R-1).
2. **Graph-captured forward** — `model = torch.compile(model, mode="reduce-overhead")`
   (CUDA-graph backend) OR a manual `torch.cuda.graph` capture of the single-token
   decode step. Warm (compile/capture), then time steady-state.
3. **Compare** eager (current) vs graphed: per-step GPU time (cuda-Event), tok/s,
   SM clock, board power, **tok/W** — at **B=1, 4, 8** Mistral-7B (PLEN=12, GEN≥64).
   B-sweep matters: graphs help most when launch overhead dominates (small B); larger B
   shifts toward bandwidth/compute.

Self-contained probe script (`fwd1_cudagraph_probe.py`) — **does not modify** the
W.4b prototype; it measures the *same* decode the executor runs (inputs_embeds /
StaticCache / 1-token step). Reuse, not fork.

## §3 — Decision gate (what the probe must produce)

1. **Eager-vs-graphed per-step GPU-time table** at B=1/4/8 (the deliverable regardless).
2. **Verdict:**
   - **LEVER REAL** if graphed cuts the forward materially (proposed ≥ 2× at B=4) →
     scope a follow-up memo to productionize through the substrate (and re-measure
     branch A's engagement tok/W with the faster forward).
   - **LEVER ABSENT** if graphed ≈ eager → the forward is memory-latency bound, not
     launch bound; the absolute ceiling stands at this B; document and proceed to W.7.
   - **PARTIAL** (middle) → report the honest factor and let Anil weigh productionization
     vs W.7.

## §4 — Correctness (Memory #11 — every measured config)

CUDA graphs/`StaticCache`/compile change *kernel scheduling and cache layout*, never the
math — but a capture bug (stale address, wrong cache slot, mask off-by-one) silently
corrupts. So each graphed config must reproduce the eager reference:
- exact-greedy token match vs the eager B=1 reference per row;
- teacher-forced per-step KL ≤ ~7e-5 (the W.4b.6/.7 bar);
- any divergence adjudicated by the near-tie gap (Memory #11), not waved off.
A faster-but-wrong forward is the central risk and is exactly what this gate catches.

## §5 — Execution plan (7 atomic items; approve before item 1)

1. **Baseline (eager) reference** — B=1/4/8 Mistral-7B decode with `StaticCache`,
   eager: per-step GPU-time (cuda-Event), tok/s, clock, power, and the gold token /
   logit reference for the correctness gate.
2. **Graphed forward** — `torch.compile(mode="reduce-overhead")` over the StaticCache
   decode; warm; capture; verify it actually graph-replays (no recompiles per step —
   assert via a step-time floor / compile counter).
3. **Correctness gate** (§4) on the graphed path vs the item-1 eager reference.
4. **B-sweep measurement** — eager vs graphed per-step GPU-time + tok/W at B=1/4/8;
   localize where graphs help (and confirm/refute launch-bound via the B-trend).
5. **(if LEVER REAL) cross-check** the graphed forward inside the branch-A full-handoff
   executor path (does the density tok/W actually rise?) — measurement only, no
   substrate edit.
6. **Write `FWD_1_FINDINGS.md`** — the eager-vs-graphed table, the (a)-vs-(b) verdict,
   tok/W deltas, and the productionize-vs-W.7 recommendation.
7. **Commit** (author `Anil`, no co-author trailer; userspace probe; anchors unchanged;
   tag proposal `fwd-1-probe` only if Anil directs). Update memory. **STOP for Anil's
   productionize-vs-W.7 adjudication.**

## §6 — Risks

- **R-1 (static-shape capture, HIGH):** the growing KV/mask breaks naive graph capture.
  *Mitigation:* `StaticCache` (fixed max length) is the supported path; this is the
  first thing item 2 must get right, and the correctness gate (§4) catches a mis-sliced
  cache.
- **R-2 (it's actually memory-latency bound):** graphs don't help. *Mitigation:* that is
  a valid verdict (§3 LEVER ABSENT), not a failure — and the B-sweep evidences it.
- **R-3 (compile recompiles / falls back):** torch.compile silently de-optimizing would
  fake a null result. *Mitigation:* assert graph replay (no per-step recompile) in
  item 2 before trusting the number.
- **R-4 (probe ≠ production):** a userspace graphed forward is not the shipped substrate
  path. *Mitigation:* the probe only decides *whether* the lever exists; productionizing
  is a separate gated memo — no anchor change in this substep.
- **R-5 (first-token/prefill excluded):** measure steady-state decode only (the regime
  W.4b cares about), windowed as in W.4b.7.

## §7 — One-line ask

Approve this probe (or adjust the §3 ≥2× gate / the B-set / the designation `FWD-1`)
and I execute items 1→7, stopping at the §3 verdict for your productionize-vs-W.7
adjudication. Userspace measurement only; anchors unchanged; Memory #11 enforced per
config.
