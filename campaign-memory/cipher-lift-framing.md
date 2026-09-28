---
name: cipher-lift-framing
description: "For CIPHER P4.2 (SM partition cluster), aggregate-TPW lift is mechanically unsupported on WL01-WL24; reframe to tail-latency isolation OR substrate-only ship"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

Bin-packing argument: SM partitioning lifts aggregate throughput only
when `S << C AND compute-bound at multi-tenant scale`. Walking
WL01-WL24:
- Compute-bound workloads (WL03/04/13/14) have S ≈ C single-tenant
  (saturate one device); partitioning wastes idle SMs.
- Memory-bound workloads (WL01/02/05) bottleneck on HBM, which
  doesn't partition with SMs.
- No row in the matrix satisfies the joint condition.

**Why:** committing to demonstrate aggregate-TPW lift on WL01-WL24
will produce null results regardless of how well the actuator works,
because the lift mechanism has no foothold. T4.2.4b WL05 null result
(-0.43%, neutral) and T4.2.4c noisy-neighbor null result (p99 Δ
+1.3%) are both consistent with this.

**How to apply:** P4.2 ship-gate evaluation should rest on the
existing G1/G2/G4 MFU gates (currently passing at 100% on WL05)
rather than on a TPW-lift demonstration. Reframe any future P4.2
measurement task to ask "does the substrate work?" rather than
"does the substrate produce visible lift?"

For genuine lift demonstrations of partition-style actuators, look
to workloads OUTSIDE the WL01-WL24 set (e.g., adversarial neighbor
patterns, multi-tenant with same-resource contention) — and only
after fixing the partition-not-enforced gap (see
[[cipher-t424c-partition-not-enforced]]).

See `PHASE_4_T4_2_4_LIFT_REFRAME.md` for the full walkthrough.
