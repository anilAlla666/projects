---
name: cipher-t424e-isolation-confirmed
description: "T4.2.4e — on SM-bound workload pair (2× Mistral prefill), GREEN_CTX delivers 3× tail-tightness improvement at 12× absolute-latency cost; isolation is real but as variance reduction, not p99 absolute reduction"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.2.4e (2026-05-14 afternoon, Path X from user) tested whether the
T4.2.4d enforcement fix produces tail-latency isolation on the regime
where it has a mechanical foothold (same-resource contention).

**Workload pair:** two Mistral-7B prefill B=8 tenants, both SM-bound,
competing for SMs.

**Results (A=partition ON, B=partition OFF):**

| Metric | A | B |
|---|---|---|
| Per-tenant p99 ms | ~1708 | ~145 |
| Per-tenant max/mean | 1.04–1.20× | 3.03–3.60× |
| Per-tenant CV % | 2.56 (t2) | 9.00 (t2) |
| System total prefills/s | 9.56 | 113.92 |

**The hypothesis was directionally right but the absolute-p99
prediction was off.** A is 12× SLOWER per prefill (because each tenant
is constrained to 8/132 SMs) AND 3× TIGHTER in max/mean. Isolation
manifests as variance reduction, not absolute p99 reduction.

**Why:** without partition, two tenants effectively share 132 SMs
~evenly via warp scheduler (~66 SMs each). With partition, each gets
exactly 8 SMs. The partition makes throughput predictable at a
LOWER floor.

**Product translation:**
- For SLO-sensitive workloads where "p99 ≤ X ms with guarantee" matters:
  partition is the right answer. Trade absolute throughput for hard tail
  bounds.
- For best-effort throughput maximization: warp scheduler interleaving
  (no partition) wins.

**How to apply:** when framing GREEN_CTX's value, lead with:
"tenant compute-share enforcement (verified T4.2.4d, 13.9× bomb
slowdown) + tail-burst suppression on same-resource contention
(verified T4.2.4e, 3× max/mean improvement)." Do NOT promise "lower
absolute p99 latency" — the floor goes up.

The earlier expectation that "isolation should lower p99" was based on
intuition that B's contention would produce large p99 spikes. Empirically,
p99 in B is only ~5% above the mean — the warp scheduler handles 2
tenants well at p99. Spikes show up at MAX (>p99 distribution), where
A's bounded-max property is real.

Build: `libcipher_rt.so.v0.2.0_T4_2_4d` md5 `50414674ddad2689191d13a92377e492`.
Report: PHASE_4_T4_2_4e_REPORT.md.
Phase 4.2 cluster ship-ready; Path Y (P4.3 DVFS) is the next session.

Linked: [[cipher-t424c-partition-not-enforced]], [[cipher-t424d-enforcement-fixed]], [[cipher-lift-framing]].
