---
name: cipher-cp47-closed
description: CIPHER CP 4.7 (kernel fusion) CLOSED 2026-05-17 via bound characterization — not built
metadata: 
  node_type: memory
  type: project
  originSessionId: 48591690-eff3-4458-9d0c-6f3597471e76
---

CP 4.7 (kernel-fusion throughput lever) **CLOSED 2026-05-17 at the design
memo** — §6 adjudicated to path 7.2 (close at memo). Canonical artifact:
`cipher-fusion-evidence/cp_4_7/CP_4_7_DESIGN_MEMO.md`. No build performed; one
authorized launch-count spot-check run during memo writing.

**Why killed before build — two converging lines:** (1) Hook reachability:
measured 2,789 kernel launches/token (Mistral-7B B=1 decode on c2c5d313) —
225 GEMM (8%), 32 SDPA, ≈2,532 elementwise/reduce (91%). The cuBLAS GEMM
dispatch hook the CP scope specified sees only the 8% GEMM surface; Mistral/
Llama projections are bias-free, pruning reachable fusion to ≈32 launches/token
= 1.1%. (2) Direct prior measurement: may13 canonical fusion-only (far more
aggressive Python-level fusion) measured B=1 Mistral +0.88%, B=8 −0.23% —
already in n=5 noise. Predicted CP 4.7 lift +0.09–0.34%, null.

**State:** kernel-fusion actuator NOT migrated into cipher_rt_phase4; deferred
indefinitely. A CUDA-graph-capture re-scope (memo §6.3) — the mechanism that
actually reaches per-launch overhead wholesale — remains available as a
separate future CP. Anchors UNCHANGED — no rebuild: kmod e2f50452,
libcipher_v2 86618c30, libcipher_rt c2c5d313.

**Phase 4 sub-cluster progress:** 4.3 DVFS shipped (CP 2.4), 4.4 L2 CLOSED at
memo ([[cipher-cp44-closed]]), 4.5 Weight/Marlin shipped (CP 2.4), 4.6
KV/attention CLOSED ([[cipher-cp4656-closed]]), 4.7 fusion CLOSED at memo (this
CP). **Only 4.8 (integration soak) remains to close Phase 4.**

Second consecutive bound-first closure (CP 4.4 hardware budget, CP 4.7
architectural hook-reachability) — bound-first discipline is now an established
campaign pattern: the spec hits a constraint it didn't model, the memo surfaces
it before the build week. Relates to [[cipher-fusion-campaign]],
[[cipher-cp44-closed]].
