# READ-ONLY SIZING — SM-pack lever: does filling the idle GPU lift aggregate throughput at B≥8?

**Date:** 2026-05-31. **Type:** SIZING (no build, no commit, anchors unchanged). Sizes ORCHESTRATE's
"producing half" (the unbuilt G3 batched-decode lever): the Marlin run left the GPU at ~110-143W/700W —
underutilized. Does filling the idle lift throughput at B≥8? **YES — and we measured HOW. The same 32
sequences of work: as one BATCHED instance (B=32) = 1383 tok/s; as 4 PACKED concurrent B=8 streams = 350
tok/s. Batching is 3.95× packing (same harness, same work, gate-free). So the idle is fillable by making the
work BIGGER per launch (batching), NOT by packing more concurrent small instances — which gives ~1.0× (no
overlap; host-launch-bound, GPU stays ~51% util / ~200W). Green-Context SM-partitioning has no throughput
cargo (it's isolation). For SAME-model tenants the producing lever is shared batching (proven, gate-free);
for DISTINCT-model tenants (can't batch) eager concurrency fails here and graphs/MPS *might* fix it — a
hypothesis, not measured this pass.**

---

## PREMISE — is the GPU underutilized at the product's batch? **YES (confirmed quantitatively)**

Computed from measured tok/s (more decisive than any occupancy proxy; `nvidia-smi dmon sm%` is temporal, not
spatial, and there's no ncu/dcgm here):
- Mistral-7B fp16 ≈ 14.5 GB weights, reread once per decode step. **steps/s is ~flat across batch**: B=8 →
  43 steps/s, B=64 → 41 steps/s (tok/s scales 323→2645 *only* because each step does more tokens, not faster
  steps) — the **launch/overhead-bound signature**.
- 41 steps/s × 14.5 GB ≈ **~595 GB/s = ~18% of H100's 3350 GB/s** at B=64. Single-tenant decode: util 51%
  (temporal), **power ~200 W of 700 W**. Neither compute- nor bandwidth-bound → overhead/launch-bound, with
  large idle. **Matches prior `cipher-phase-a-multitenant` "HBM 4-18%".** Premise PASS.

## THE LEVER — N concurrent tenants, packed vs not (Mistral-7B replicas, B=8 each, eager, K=64 steps)

Single process, N replicas, each tenant a manual on-stream decode loop (no host sync mid-loop; argmax→next
on-stream). Warm baseline (one-time init paid before timing). `nvmlDeviceGetUtilizationRates`/power sampled.

| N | mode | agg tok/s | wall | meanW | maxutil | vs solo | correct |
|---|---|---|---|---|---|---|---|
| 1 | solo (warm) | 343 | 1.49 | 202 | 51% | 1.00× | ref |
| 2 | serial | 342 | 2.99 | 201 | 52% | 1.00× | ✓ |
| 2 | concurrent (streams) | 346 | 2.96 | 206 | 52% | **1.01×** | ✓ |
| 2 | threads | 319 | 3.21 | 199 | 52% | 0.93× | ✓ |
| 3 | concurrent | 350 | 4.39 | 206 | 53% | **1.02×** | ✓ |
| 4 | serial | 345 | 5.94 | 201 | 52% | 1.01× | ✓ |
| 4 | concurrent (streams) | 350 | 5.86 | 204 | 52% | **1.02×** | ✓ |
| 4 | threads | 186 | 11.04 | 167 | 52% | 0.54× | ✓ |

- **Concurrent ≈ serial ≈ solo-rate (~1.0×) at every N.** 4 tenants on 4 streams take the same wall as
  running them one-at-a-time → **~0% useful overlap**. Adding tenants adds **no** aggregate throughput.
- **Multi-threaded launcher is WORSE** (0.93×→0.73×→0.54× at N=2/3/4) — Python GIL serializes launches and
  adds thread-switch overhead. A more aggressive host launcher does **not** recover overlap.
- **Power flat ~200 W, util ~51% at every N** — never rises toward 400W+/full. Per the discriminator: this is
  the **HOST-LAUNCH CEILING with the GPU still idle**, NOT GPU saturation (the bandwidth calc shows ~18%).
  A single eager process cannot issue kernels fast enough to build a concurrent queue that fills the SMs;
  each tenant's host-launch time ≈ its GPU time, so the streams serialize.
- **Correctness ✓**: every tenant's greedy tokens == solo reference — co-residence never perturbs output
  (Mem #11 holds).

## THE AFFIRMATIVE LEVER — BATCH the work, don't PACK instances (same harness, apples-to-apples)

The same **32 sequences** of decode work, two ways (identical manual-loop harness, warm):

| 32 seqs as… | tok/s | meanW | util | bandwidth |
|---|---|---|---|---|
| **4 PACKED B=8 concurrent streams** | **350** | 204 | 52% | ~18% |
| **1 BATCHED instance @ B=32** | **1383** | 252 | 62% | ~18% |

**Batching = 3.95× the throughput of packing**, gate-free, no partitioner, no graph. (Single-tenant B=8→B=64
is 343→2645 = **7.7×** — batching scales hard.) Batching fills the idle by putting **more tokens per launch**
(43 steps/s is unchanged B=8→B=32→B=64 — the step rate is overhead-capped); packing more *small* launches
does not, because the host can't issue them fast enough. **This is the measured "fill the idle GPU" lever.**

## THE ANSWER — split by same-model vs distinct-model (they differ)

The idle is real and large (premise PASS), and **fillable — measured.** But the mechanism that fills it
depends on whether tenants share weights:
- **SAME-model tenants → BATCH (proven, gate-free): 3.95× over packing, 7.7× B=8→B=64.** Route N same-model
  streams into one shared batch. This is the cross-tenant density lever (= prior `cipher-phase-a-multitenant`
  3-6× tok/W). **Real and growing with tenant count** (unlike Marlin/fusion which decayed). No partitioner,
  no graph gate needed.
- **DISTINCT-model tenants → can't batch (different weights) → need concurrent execution → eager FAILS here**
  (~1.0×, host-launch-bound, this test). Whether **graphs/MPS** would recover concurrency for distinct models
  is a **HYPOTHESIS, NOT measured this pass** — the host-launch diagnosis (power flat ~200W/51% at saturation)
  *implies* launch-elimination would help, but I measured only that *batching* fills the idle and that *eager
  concurrency* does not.
- **Green-Context SM-partitioning has no throughput cargo either way** — free concurrent streams already give
  ~1.0×; partitioning only caps further. It's an *isolation/QoS* primitive (confirms `cipher-cp54-deferral-
  audit` "contention-gated" / `cipher-phase-a-multitenant` "substrate 1.00×"). Building an SM-packer would not
  move this number.
- **Reconciliation with prior "concurrency 2.0×":** NOT achievable via naive eager concurrent streams (this
  test isolates exactly that → ~1.0×, threads worse). The prior lift came from batching (measured here, 3.95×)
  or would need launch-elimination — not from stream/partition packing.

## ORCHESTRATE / CP-5.5 implication
ORCHESTRATE's "producing half" is **not** an SM-packer/partitioner — that lever is empty. It splits:
- **For tenant density (same-model): route into shared batches.** Proven 3.95–7.7×, gate-free — **this is the
  buildable producing lever**, and it's ORCHESTRATE job-2's real cargo.
- **For distinct-model concurrency + the consuming actuators (Marlin/FP8/fusion): open the launch/graph gate**
  (CUDA graphs / MPS). For the consuming half this is a **measured** blocker (graph-bypass); for distinct-model
  concurrency it is a **hypothesis** (launch-bound diagnosis, untested).
Caveat: even batched B=32/B=64 sits at ~18% bandwidth with flat steps/s — batching helps a lot but does **not
saturate** (per-step overhead persists), which is *why* graphs still matter eventually. But batching is the
immediate, measured, gate-free win. No code changed; measurement-only; anchors unchanged.
