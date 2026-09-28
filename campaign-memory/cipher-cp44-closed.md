---
name: cipher-cp44-closed
description: CIPHER CP 4.4 (L2 weight pin) CLOSED 2026-05-16 via bound characterization — not built
metadata: 
  node_type: memory
  type: project
  originSessionId: 48591690-eff3-4458-9d0c-6f3597471e76
---

CP 4.4 (L2 persistent weight pin) **CLOSED 2026-05-16 at the design memo** —
§6 adjudicated to path 6.2 (skip the build). Canonical artifact:
`cipher-fusion-evidence/cp_4_4/CP_4_4_DESIGN_MEMO.md`. No code, no build, no
measurement — the working-set bound was decisive.

**Why killed before build:** H100 persisting-L2 budget is 31.25 MiB (hardware
cap, `cudaDevAttrMaxPersistingL2CacheSize`, live-queried). B=1-decode weight
working set is read once per token with no intra-token reuse: Mistral-7B INT4
≈3.7 GB (118× over budget), Llama-3.1-8B fp16 ≈16 GB (524× over). Pinnable
fraction S/W = 0.2–0.9% → predicted tok/s lift +0.1–0.6%, a null inside n=5
noise. The canonical `cipher_l2_persist.cu` was an LNN actuator (<3 MB hot
set, fits L2); 7B transformer decode is a different regime the lever can't
serve. Lever earns only when hot weight set ≤ ~31 MiB.

**State:** L2-weight-pin actuator NOT migrated into cipher_rt_phase4; deferred
indefinitely unless a small-W regime surfaces in Phase 5. Anchors UNCHANGED —
no rebuild: kmod e2f50452, libcipher_v2 86618c30, libcipher_rt c2c5d313.

**Phase 4 sub-cluster progress:** 4.3 DVFS shipped (CP 2.4), 4.4 L2 CLOSED
(this memo, not built), 4.5 Weight/Marlin shipped (CP 2.4), 4.6 KV/attention
CLOSED ([[cipher-cp4656-closed]]); 4.7 fusion+agentic and 4.8 integration-soak
remain. Next CP candidate: **CP 4.7**.

Precedent: bound-first engineering-marvel discipline killing a CP before the
build week — the marvel is the bound, not the lever. Relates to
[[cipher-fusion-campaign]], [[cipher-lift-framing]].
