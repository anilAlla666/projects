---
name: cipher-marlin-primary-ctx-pin
description: Marlin GEMM is structurally full-GPU; CP 2.4 Fix A pins all Marlin engine work to the primary context; libcipher_rt anchor 5e304549
metadata: 
  node_type: memory
  type: project
  originSessionId: 6147a283-c0ca-4daf-83c4-310b1e306011
---

Marlin's INT4 GEMM kernel is **structurally full-GPU**: `grid = SM count`
(132 on H100), persistent-style, with inter-CTA split-K coordination through a
`locks` buffer — its CTAs must be co-resident. Launched into an SM partition
(e.g. the 8-SM green context) it **deadlocks** (resident CTAs spin on `locks`
signals from CTAs that can't be scheduled). This surfaced once kmod 0.4.8
restored `/dev/cipher` — that activated CUPTI's T4.2.4d green-context
enforcement, which made the Marlin GEMM land in the green context.

CP 2.4 **Fix A**: a `PrimaryCtxGuard` RAII in `cipher_rt_marlin_engine.cpp`
wraps both `cipher_rt_marlin_engine_quantize_repack` and
`cipher_rt_marlin_engine_dispatch` — all Marlin engine GPU work runs in the
device-0 primary context (full 132 SMs). New libcipher_rt campaign anchor:
**`5e304549`** (clean Fix A; supersedes `a0d6cdda`). kmod anchor unchanged:
0.4.8 `e2f50452`.

**Why:** Marlin and green-context SM-partitioning cannot compose as built —
an architecture-level constraint, not a coding bug.
**How to apply:** never expect Marlin to run SM-partitioned; for CP 2.4
(single-tenant) Fix A pins it full-GPU and is correct. Multi-tenant Marlin +
partitioning is Phase 5 architectural work — see
`cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`,
`cp_2_4/MARLIN_HANG_ROOT_CAUSE.md`. Related: [[cipher-devnode-codified]],
[[cipher-t45-substrate-marlin]].
