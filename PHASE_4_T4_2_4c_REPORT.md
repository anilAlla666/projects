# Phase 4 T4.2.4c — Noisy-neighbor tail-latency isolation A/B

**Date:** 2026-05-14 midday session (continuing post-T4.2.4b_v2).
**Hypothesis being tested:** GREEN_CTX SM partitioning isolates a
latency-sensitive decode tenant's per-token latency from a
co-resident saturating prefill "bomb" tenant.
**Kmod substrate:** 0.4.5 (md5 `b47db4500e226f5ddc010f9148d3d6ca`),
**RELOADED at 12:27:19 UTC** to clear T4.2.4b_v2 pool state. Fresh
post-reload: `alloc_failures=0`, slot pool clean.
**Lib under test (A):** `libcipher_rt.so` md5 `8fbc350fb322b0733ce6d4af1468a88f`
(T4.2.4b_v2 build — hash-mix + runtime API CUPTI + push instrumentation).
**Lib under control (B):** `libcipher_v2.so` (no GREEN_CTX, no partition).
**Pre-T4.2.4c artifact saved:** `libcipher_rt.so.v0.2.0_T4_2_4b_v2.pre_T4_2_4c`
md5 `8fbc350fb322b0733ce6d4af1468a88f`.

## Reframing trigger — why a noisy-neighbor experiment

`PHASE_4_T4_2_4_LIFT_REFRAME.md` (this session) walked the bin-packing
mechanics for aggregate-throughput lift and found no WL01–WL24 row
satisfies `S << C AND compute-bound at scale`. The aggregate-TPW
hypothesis is mechanically unsupported. The pivot is tail-latency
isolation: explicit SM partitioning's published value-prop is QoS
under contention, not aggregate throughput.

## Experimental setup

| Component | Spec |
|---|---|
| Bomb tenant | Mistral-7B-v0.1 prefill B=8, prompt_len=1024, single tenant |
| Victim tenant | TinyLlama-1.1B-Chat decode B=1, max_new_tokens=32 |
| Bomb lead | 5 s (bomb warms SMs before victim joins) |
| Run duration | 120 s victim window (bomb runs 125 s total) |
| Library binding | `CUDA_INJECTION64_PATH` swap per A/B condition |
| Sampler cadence | 1 Hz device-wide via /metrics on :9402 |
| Latency capture | HF generate `streamer=` hook → `time.perf_counter()` per token |
| Token-count metric | **B11-fixed** wl01 (new tokens only, prompt echo removed) |

Memory footprint (single-GPU H100 80 GB): Mistral-7B fp16 ≈ 14 GB +
TinyLlama-1.1B fp16 ≈ 2.5 GB = ~17 GB resident. Within capacity.

A and B captured in same wall-clock window (12:34–12:38 UTC, ~4 min
apart) — satisfies B9 matched-pair rule.

## Headline measurement — victim inter-token latency

| Metric | A (partition ON) | B (partition OFF) | Δ A vs B |
|---|---:|---:|---:|
| `mean_inter_token_ms` | 18.74 | 18.56 | **+1.0%** |
| `p50_inter_token_ms` | 19.17 | 19.11 | **+0.4%** |
| `p95_inter_token_ms` | 19.40 | 19.37 | **+0.2%** |
| `p99_inter_token_ms` | **34.56** | **34.13** | **+1.3%** |
| `max_inter_token_ms` | 286.18 | 75.05 | +281% (single outlier) |
| `stdev_inter_token_ms` | 6.33 | 3.66 | +73% (driven by 1 outlier) |
| `min_inter_token_ms` | 13.98 | 13.98 | 0% |

**The p50/p95/p99 percentiles are statistically indistinguishable between A and B.**
Partition is OFF in B by construction (no GREEN_CTX); partition is verified
ON in A (cuGreenCtxGetDevResource confirms 8 SMs for the victim's group;
push observation #1/#2/#3 logged for both bomb and victim streams).

A's max-outlier at 286 ms is a single-token tail event (1 of 6368 tokens,
0.016%); B's tail is bounded at 75 ms. Cause undetermined; outside the
99th-percentile window (the standard tail-latency metric), which is
essentially identical across A and B.

## Aggregate-throughput cross-check

| Metric | A | B | Δ |
|---|---:|---:|---:|
| Victim decode iters (120 s) | 199 | 202 | -1.5% |
| Victim new tokens (B11-fixed) | 6,368 | 6,464 | -1.5% |
| Victim tok/s | 53.0 | 53.7 | -1.3% |
| Bomb prefills (125 s) | 11,728 | 11,808 | -0.7% |
| Bomb prefills/s | 93.8 | 94.4 | -0.6% |
| Device mean SM util % | 77.5 | 77.8 | -0.4% |
| Device mean power W | 489.9 | 493.0 | -0.6% |
| Device max power W | 644.3 | 641.6 | +0.4% |

**Bomb throughput is identical in A and B** (Δ < 1%). This is the
strongest diagnostic in the run: if GREEN_CTX were actually constraining
the bomb to its 8-SM partition (12 SMs out of 132 = 6% of device),
bomb prefills/s should have dropped by an order of magnitude under A.
It didn't. Therefore the partition is **not enforced at the kernel
scheduling level** for the bomb's PyTorch-launched streams in this build.

The same conclusion applies to the victim. Both run at full unrestricted
throughput regardless of partition setting.

## Why the partition isn't reaching kernel scheduling

The substrate verifications from T4.2.4b_v2 (cuGreenCtxGetDevResource
returns 8 SMs; push fires on cudaStreamCreate observations #1–#3)
established that the API-level partition is real. The remaining gap:
the *stream-to-green-context binding* is not catching the streams
PyTorch actually launches kernels on.

Plausible mechanism (post-experiment hypothesis, not verified):

- PyTorch's CUDA backend creates a per-device default stream early
  during `torch.cuda` initialization, before our CUPTI subscriber
  binds.
- After our subscriber is active, PyTorch reuses the existing default
  stream for the bomb's prefill kernels rather than calling
  `cuStreamCreate` per kernel.
- The 3 `push fired` observations the build logs capture small
  stream-creation events (e.g., NCCL/comm streams or CUDA Graph
  internals), not the hot prefill path.
- Therefore the bomb's kernels execute on the original 132-SM context,
  not the 8-SM green context.

This is consistent with the empirical fact that bomb prefills/s is
identical A↔B and device SM util saturates at 100% peak in both.

T4.2.4b's "push fires" smoke check verified push *exists* but not
that push affects the kernels carrying user work. A diagnostic for a
future sub-phase (T4.2.4d or P4.2 close-out): instrument
`cuLaunchKernel` / `cudaLaunchKernel` directly and log
`cuStreamGetCtx` to confirm which context the launching stream is
bound to.

## Verdict on the T4.2.4c hypothesis

**Tail-latency isolation hypothesis is not supported in this experiment,
with the workload-pair caveat below.** Three independent indicators:

1. p99 of victim inter-token latency: A 34.56 ms, B 34.13 ms (Δ +1.3%).
2. Bomb throughput: A 93.8 prefills/s, B 94.4 prefills/s (Δ -0.6%).
3. Device SM util: A 77.5%, B 77.8% (Δ -0.4%).

None of these moves at noise-band level (|Δ| ≤ ~1%), let alone the
multiplicative effect that genuine SM-budget isolation would produce.

This is **substrate-verified, product-value-undemonstrated** for the
P4.2 GREEN_CTX actuator on this noisy-neighbor pattern, in addition to
the prior T4.2.4b_v2 finding of no aggregate-TPW lift on WL05.

## Caveats and confounders

The bomb (Mistral-7B prefill, **compute-bound** — saturates SMs) and
the victim (TinyLlama decode B=1, **memory-bound** — saturates HBM)
bottleneck on different resources. The GREEN_CTX partition mechanism
reserves SMs, not HBM bandwidth, so even if enforcement reached kernel
scheduling, isolation gain would be small on this workload pair: the
victim's decode is HBM-limited and would see roughly the same per-token
latency whether it has 8 SMs or 132 SMs allocated to it.

This means the headline p99-flat result has **two** non-exclusive
explanations:

- (A) **Partition is not enforced at kernel scheduling under PyTorch's
  stream management** in this libcipher_rt build (the bomb-prefills/s-
  identical-A↔B finding directly supports this — if enforcement worked,
  the bomb at 8 SMs out of 132 would be ~16× slower).
- (B) Even if (A) were fixed, **this workload pair would not show large
  isolation gain** because the bomb's compute-bound prefill kernels
  don't actually starve the victim's HBM-bound decode kernels at warp
  granularity — the warp scheduler already interleaves them.

The bomb-throughput diagnostic discriminates (A) cleanly: the partition
*is not reaching* kernel scheduling. The p99-flat headline is jointly
consistent with (A) and (B), so it does **not by itself** falsify the
tail-latency-isolation hypothesis in the strong form. A clean test
requires two tenants competing for the *same* resource — e.g., two
prefill tenants both SM-saturating — and a libcipher_rt build where
the green-ctx binding actually reaches the hot-path streams.

## What it means for Phase 4.2

`PHASE_4_ARCHITECTURE.md` G1–G4 ship gates are MFU-based:

| Gate | Target | Current state |
|---|---|---|
| G1 | ≥85% MFU per WL | Passing where measured (WL05 100%, WL03 96%, WL14 100%) |
| G2 | ≥90% MFU on WL05 ×8 | **Passing at 100%** (T4.2.4b_v2 600 s) |
| G3 | ≥30 tenants on WL05 | Not measured at T4.2.4 scale (8 in scope) |
| G4 | zero oops, taint ≤+1, 24 h soak | Passing (taint 12288 stable) |

The cluster-checkpoint rule is "lift MFU on declared-workloads-set; no
regression on complement set." MFU is at the ceiling on the headline
WL05 workload — no headroom. G1/G2/G4 currently pass without
GREEN_CTX adding measurable enforcement.

**P4.2 substrate is ship-eligible** on existing MFU gates, but the
**partition mechanism's product value is not demonstrated** in any of
the dimensions tested so far (aggregate TPW on WL05, tail latency on
noisy-neighbor). The substrate ships; the actuator's measurable benefit
remains an open finding.

## Honest finding statement

The T4.2.4c noisy-neighbor A/B finds no measurable tail-latency
isolation from the GREEN_CTX actuator as built in T4.2.4b_v2, on the
Mistral-7B-prefill-bomb + TinyLlama-decode-victim workload pair, at
this kmod 0.4.5 + libcipher_rt revision. As called out in the Caveats
section, this result has two non-exclusive explanations — kernel
scheduling enforcement is not reaching the hot path (directly supported
by bomb-throughput-identical), and the workload pair's compute/HBM
asymmetry would itself dampen any isolation gain — and is therefore
**not** a strict falsification of the isolation hypothesis.

The substrate works at the API surface (push fires; green ctx contains
the requested SM count; hash-mix distributes tenants to distinct
groups). The substrate does **not** translate to kernel-level
enforcement on the user's hot path under PyTorch's stream management.

This finding does not constitute a regression. Both A and B are within
the WL05 noise band and within the WL01-decode reference. Phase 3 ABI
12/12 PASS. No taint delta. No dmesg oops/WARN/BUG.

## What would close this gap (future work, not this task)

- **Direct cuLaunchKernel instrumentation:** log `cuStreamGetCtx(stream)`
  per launch to confirm which context is active.
- **PyTorch stream-injection path:** override
  `torch.cuda.set_stream()` / `torch.cuda.Stream()` to bind every new
  stream to the green ctx explicitly.
- **Compute-only microbenchmark:** replace the PyTorch bomb with a
  hand-launched cuLaunchKernel of a known 100ms saturating kernel,
  re-run A/B. If isolation works there but not under PyTorch, the gap
  is confirmed to be stream binding.
- **NSight Systems trace:** capture kernel placement under A. If
  kernels show on all 132 SMs, partition isn't reaching scheduling.

Out of scope for T4.2.4c (3-hour cap). These are P4.2-close-out or
T4.2.5 candidates.

## Discipline gate at session end

| Gate | Result |
|---|---|
| Phase 3 ABI 12/12 PASS | ✅ (happy 3 + negative 6 + root 3) |
| Fallback kmod md5 55ab8c0c | ✅ unchanged |
| Fallback libcipher_v2 md5 86618c30 | ✅ unchanged |
| Taint 12288 | ✅ unchanged |
| Kmod 0.4.5 loaded (srcversion 4618FD1FC28BEE5EBC32944) | ✅ |
| dmesg oops/warn/bug since reload | none |
| alloc_failures post-reload | 0 |
| Daemons running (cipher-gpustate, cipher-exporter :9402) | ✅ |

## Artifact map

| Artifact | md5 | Purpose |
|---|---|---|
| `libcipher_rt.so.v0.2.0_T4_2_4b_v2.pre_T4_2_4c` | `8fbc350fb322b0733ce6d4af1468a88f` | discipline pre-snapshot |
| `libcipher_rt.so` (live) | `8fbc350fb322b0733ce6d4af1468a88f` | T4.2.4b_v2 build, unchanged by T4.2.4c |
| `cipher_kmod.ko` (live) | `b47db4500e226f5ddc010f9148d3d6ca` | 0.4.5 unchanged |
| `drivers/wl_noisy_neighbor.sh` | new | A/B orchestrator |
| `drivers/wl_noisy_neighbor_bomb.py` | new | Mistral prefill bomb |
| `drivers/wl_noisy_neighbor_victim.py` | new | TinyLlama decode + per-token latency capture |
| `measurement/nn_analyze.py` | new | p50/p95/p99 + device aggregator |
| `cipher-phase4-evidence/t4_2_4c/A_summary.json` | new | run A aggregate |
| `cipher-phase4-evidence/t4_2_4c/B_summary.json` | new | run B aggregate |
| `cipher-phase4-evidence/t4_2_4c/B_nn_victim_latencies.json` | new | raw per-token timestamps (B run; A's were overwritten by B's run) |
| `cipher-phase4-evidence/t4_2_4c/B_nn_device.csv` | new | 1 Hz device timeline (B run) |

Note on A raw timestamps: a single `/tmp/nn_victim_latencies.json`
location is used by the driver, and run B overwrote run A's file.
The A run's aggregate statistics (from analyzer JSON) are preserved.
For future A/B runs the orchestrator should rename the latency file
per-condition; this is a backlog clean-up.

## Cross-link to backlog

- B11 (token-count overstatement) — wl01 fix verified live in this run.
  The 53 tok/s victim throughput is now the *real* tok/s (32 new
  tokens × 199 decode iters / 120 s = 53.07).
- T4.2.4b_v2's "push fires confirmed" smoke check is now contextualized:
  push exists, but does not bind the streams carrying PyTorch's hot-path
  kernels. This is an addendum to the T4.2.4b_v2 report's interpretation,
  not a contradiction.
