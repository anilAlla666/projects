# T4.2.4 lift framing — reframe to tail-latency isolation

**Date:** 2026-05-14 morning, after T4.2.4b_v2 closed neutral.
**Trigger:** advisor binding-check on T4.2.4c plan draft.

## The mechanical bin-packing argument

For SM partitioning to lift aggregate throughput, we need:
  **S (per-tenant SM demand) << C (device capacity)  AND  workload compute-bound.**

Walking the WL01–WL24 matrix against this filter:

| Workload | S vs C | bound | Partition lift? |
|---|---|---|---|
| WL01 decode B=1 | S << C | **memory** | No (HBM is the bottleneck) |
| WL02 decode B=8 | S < C | **memory** | No (same reason) |
| WL03 prefill B=8 | **S ≈ C** | compute | No (single-tenant already saturates 96% SM) |
| WL04 vLLM serving | S ≈ C | compute | No |
| WL05 multi-tenant ×8 (TinyLlama decode) | S << C × 8 | **memory** | No (advisor's T4.2.4b prediction, verified) |
| WL13 long-context | S ≈ C | compute | No |
| WL14 torch.compile | S ≈ C | compute | No |

There is **no row in the matrix** where `S << C AND compute-bound` at multi-tenant scale.
Single-tenant compute-bound workloads pin the device; multi-tenant memory-bound
workloads bottleneck on HBM, which doesn't partition with SMs.

**Conclusion:** the aggregate-throughput-lift hypothesis for GREEN_CTX on
WL01–WL24 is mechanically unsupported. T4.2.4b's null result was not a bug
in measurement — it was the right answer for that hypothesis on that
workload set.

## The pivot — tail-latency isolation

The literature on MPS / partitioning (NVIDIA whitepapers; CUDA-MPS docs;
Hopper green-context release notes) frames the value-prop as **QoS** /
**predictable per-tenant tail latency**, not aggregate throughput. The
default NVIDIA warp scheduler already maximizes aggregate throughput at
warp granularity; partitioning trades aggregate throughput for guaranteed
per-tenant SM budget.

The mechanical foothold:

> **Without partition, a saturating "prefill bomb" tenant starves a
> latency-sensitive decode tenant's SMs at warp granularity. Decode
> p99 token latency degrades multiplicatively under contention.
> With partition, the decode tenant's SM budget is reserved — its p99
> token latency is bounded regardless of the prefill bomb's behavior.**

This is falsifiable on existing infrastructure:
- One Mistral-7B prefill tenant (~14 GB, ~96% SM util alone)
- One TinyLlama decode B=1 tenant (~2.5 GB, latency-sensitive)
- Memory fits: 14 + 2.5 ≈ 16.5 GB << 80 GB
- A/B: partition on vs off
- Metric: decode tenant's p99 inter-token latency

## P4.2 ship-gate re-reading

`PHASE_4_ARCHITECTURE.md` G1–G4 are MFU-based. Current measurements under
T4.2.4b_v2:

| Gate | Target | Current state |
|---|---|---|
| G1 | ≥85% MFU per WL | WL05 100%, WL03 96%, WL14 100% — passing where measured |
| G2 | ≥90% MFU on WL05 multi-tenant ×8 | **100% MFU** — passing |
| G3 | ≥30 tenants on WL05 | not measured at T4.2.4 scale (8 in scope) |
| G4 | zero oops, taint ≤+1, 24h soak | passing (taint stable at 12288) |

Cluster checkpoint rule: "lift MFU on declared-workloads-set; no
regression on complement set." MFU is already 100% on WL05 (the headline
P4.2 workload) — there is no MFU headroom to lift. The cluster-rule's
intent is met by "substrate verified working, no regression."

The aggregate-TPW dimension was never a binding gate. T4.2.4b's TPW-flat
finding does not block ship.

## What this means for T4.2.4c

Two valid paths forward:

**Path A — noisy-neighbor demonstration (tail-latency isolation).**
Add the prefill-bomb + decode A/B experiment. Demonstrates GREEN_CTX
value in the regime where it actually has a mechanical foothold. Outcome
is binary (p99 isolation works or doesn't), not a noise-band fight.

**Path B — declare P4.2 ship-ready, move to P4.3.**
G1/G2/G4 currently pass; G3 (≥30 tenants) needs separate scaling work
which arguably belongs to a P4.2.5 task, not T4.2.4. The substrate is
verified: GREEN_CTX creates real per-tenant SM partitions; push fires on
user streams; hash distributes across 16 buckets. Phase 3 ABI 12/12.

Path A is cheaper (~2 hours, no new code beyond the experiment runner)
and answers the open question. Path B leaves the open question in
the record.

**Recommendation: Path A first, then Path B's decision is informed.**

## Discipline carry-over

- This is a planning doc, not a build. No artifact md5s change.
- T4.2.4b_v2 remains the live libcipher_rt.
- kmod 0.4.5 unchanged.
- Phase 3 ABI 12/12 unchanged.
