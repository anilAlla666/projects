# FUTURE_SCOPE/B — Track 2: Cross-Tenant Weight-Sharing Capacity Primitive

## Goal

Let N same-model tenants share **one** physical copy of the model weights in
HBM instead of N copies. A **capacity** primitive — it raises the tenant
ceiling per GPU; it is **not** a tok/W lever (Diagnostic 1 established this:
shared pages give capacity, not bandwidth — the tok/W ceiling of weight-sharing
alone is ~1.05×). Its value is enabling higher N, which the cross-tenant
batching lift is then multiplied against.

## Architectural approach

Per `PHASE_B_WEIGHT_SHARING_DESIGN.md`: tenant 0 loads the weights; peers map
the same physical copy via **cuIpc** — directly reusing the T4.6 cross-process
page-pool primitive (`cipher_kv_bridge.cpp`, `cipher_rt_kv_alloc.c`, proven
real==sim 1.000 for cross-tenant KV-dedup). Recommended: a single contiguous
weight arena exported as one cuIpc handle; peers `cuIpcOpenMemHandle` and
rebind it as their model parameter storage (PyTorch post-load rebind — the
fiddly part). Sharing applies within a **model-group** (matching model + dtype
fingerprint, tracked by libcipher_v2); a unique-model tenant degrades
gracefully to its own copy (today's behaviour, no regression).

## Build scope + dependencies

- **Depends on:** the T4.6 cuIpc page-pool code (exists).
- **Scope:** cuIpc weight-arena export/import; the HF parameter-storage rebind
  path; model-group fingerprinting in libcipher_v2; correctness validation
  (a shared-weight tenant must pass the teacher-forced KL gate bit-for-bit vs
  its own-copy run). No kmod changes.
- **Composition:** orthogonal to cross-tenant batching — not a prerequisite
  for it (the batched executor runs on one process's one copy regardless).
  It should land before the N=16+ regime, where N redundant weight copies
  would otherwise cap density (16× a 7B model ≈ 224 GB ≫ 80 GB).

## Time estimate

3–5 days.

## Success criteria

- N same-model tenants run with weight HBM = 1× model size, not N×.
- A shared-weight tenant passes the teacher-forced KL ≤ 0.1 gate, bit-identical
  to its own-copy baseline.
- Tenant-count ceiling per GPU rises accordingly (measured: e.g. Mistral-7B
  from ~5 to many tens of tenants by weight memory).
- No regression for unique-model tenants.
