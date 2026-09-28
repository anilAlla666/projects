# Phase C / Track 2 — Cross-Tenant Weight-Sharing — Design Memo

**Date:** 2026-05-18. **Type:** design, no GPU. Sub-component 1 of the
weight-sharing capacity primitive. Anchor `a7ac8e97` unchanged (design only).

**Goal:** N same-model tenants share **one** physical copy of the model
weights in HBM. A **capacity** primitive — Diagnostic 1 established the
weight-sharing tok/W ceiling is ~1.05× — its value is raising the tenant
ceiling per GPU, which the verified cross-tenant batching lift (Phase B:
3.69× / 3.26× substrate-attributable) is multiplied against. It is the hard
prerequisite for CP 5.5 (100-tenant soak): 100 independent Mistral-7B copies
≈ 1.4 TB ≫ 80 GB.

## §1 — Current state (measured, Diagnostic 1)

Each tenant is a separate process running stock
`AutoModelForCausalLM.from_pretrained(...).cuda()`. The Diagnostic-1 process
memory-map probe measured, for 4 concurrent TinyLlama tenants: **4 × 2.72 GB
disjoint copies** (per-process GPU memory 4×2720 MiB; `data_ptr`s in four
disjoint ranges; default `cudaMalloc` caching allocator, no VMM). The
duplication lives entirely in HF + the PyTorch caching allocator; CIPHER's
substrate does not touch weight allocation today. The four copies are disjoint
physical pages — there is no sharing and, as things stand, none is possible.

## §2 — Target state

One physical copy of the weights; N processes hold virtual mappings to the
same physical bytes. Weight HBM goes `N × W → 1 × W` (W = model weight bytes).
Per-tenant residual = activations + KV only (tens of MB). Weights become a
per-GPU constant; tenant count is then bounded by KV/activations, not weights.

## §3 — Architectural decision tree

### 3a. Interception point — where weights get placed in shared memory

| candidate | assessment |
|---|---|
| **(a)** `libcipher_rt` `cudaMalloc`/`cuMemAlloc` hooks | **Rejected as primary.** PyTorch's caching allocator packs weights, activations and KV into a few large `cudaMalloc` reservations — there is no clean "weights block" at the `cudaMalloc` layer to isolate and share. Sharing requires controlling *which* allocation holds weights, which means controlling the allocation anyway. Viable later as a transparency layer, not now. |
| **(b)** kmod-level mmap interception | **Mis-framed.** CUDA device memory is not file-backed mmap; the kmod has no hook for `cudaMalloc`/VMM allocation. The kmod's role here is **pool lifetime ownership / refcounting** (it already does this for the T4.6.4 dedup pool), not allocation interception. |
| **(c)** model-load path, via the T4.6 VMM bridge | **Recommended.** `cipher_kv_bridge` already produces VMM-backed `torch.Tensor`s (`vmm_zeros`), and `cipher_rt_kv_alloc.c` is a working CUDA-VMM allocator (one VA pool, 2 MiB pages, slabs). Loading weights into bridge-VMM tensors and exporting them is a *natural extension* of existing, verified infrastructure — not a hack. |

**Decision: (c).** Option (a)'s "transparency" appeal does not survive the
weights-vs-activations discrimination problem; (b) is the wrong layer for
allocation but the right layer for lifetime (use it for that). (c) is clean
*because the VMM bridge exists*.

### 3b. Cross-process sharing mechanism

| candidate | assessment |
|---|---|
| legacy `cuIpcGetMemHandle` | Works on `cuMemAlloc` pointers; the April T4.6 probe used it, then **moved off it**. |
| **CUDA VMM — `cuMemCreate` + `cuMemExportToShareableHandle(POSIX_FILE_DESCRIPTOR)` + `cuMemImportFromShareableHandle`** | **Recommended.** This is T4.6.4's *verified* cross-process path. POSIX-FD handles pass through the kmod cleanly (it already stores/`fput`s them). VMM also allows `cuMemAddressReserve` at a chosen VA — the shared weight arena can be mapped at the **same virtual address** in every tenant, so PyTorch tensors referencing it need no per-process pointer fix-up. |

**Decision: CUDA VMM, POSIX-FD shareable handles** — reuse the T4.6.4 path.

### 3c. Coupling note — necessity depends on the CP 5.5 tenancy model

Weight-sharing across *tenant processes* is load-bearing only if CP 5.5's
100-tenant architecture has many processes each holding a model. The
CIPHER+vLLM composed architecture (`FUTURE_SCOPE/A`) decides that. Under every
plausible 100-tenant topology you cannot hold 100 full copies, so *some*
sharing (this primitive) or *few-executor* consolidation is required
regardless — building this is sound — but the memo flags: confirm the A
tenancy decision so weight-sharing is deployed where it is actually the
binding constraint (peer-process topology), not where an executor already
holds a single copy.

## §4 — Reuse of T4.6 primitives

**Use as-is:**
- `cipher_rt_kv_alloc.c` — the CUDA-VMM allocator: one reserved VA pool, 2 MiB
  physical-page granularity, slab sub-ranges, per-page tags. The weight arena
  is one large slab.
- `cipher_kv_bridge` (pybind) — `bridge_init`, `vmm_zeros(shape, itemsize)`
  produce VMM-backed `torch.Tensor`s; `page_info(devptr)` reports the physical
  pages behind a pointer (→ used in §6 verification); `get_stats`.
- T4.6.4's `cuMemExportToShareableHandle(POSIX_FD)` / `cuMemImportFromShareableHandle`
  export/import path, and the **kmod-owned pool** with refcount + release-fop
  teardown + the `do_exit` reaper.

**Extend:**
- A **weight-arena slab type** — T4.6's slabs are sized/tagged for KV pages
  (per-tenant,seq,layer,head). Add a single contiguous weight-arena slab tag.
- A **model-load integration**: tenant 0 materialises each weight tensor into
  the bridge VMM arena (load on `meta`, then assign VMM-backed storages, copy
  the safetensors bytes in); exports the arena's VMM handle(s). Peer tenants
  **import** the arena and wrap it as their parameter storages — skipping the
  safetensors read entirely.
- A **model-group registry** in libcipher_v2 (it already tracks tenant
  identity) keyed by a model fingerprint (§5).

## §5 — Failure-mode handling

- **Model identity / variant detection.** Weights must be byte-identical to
  share. `Mistral-7B-v0.1` vs `Mistral-7B-Instruct` share an architecture but
  **not** weights — a config hash is insufficient. Fingerprint = hash of the
  safetensors file set (index + per-file checksums) + dtype. Only matching
  fingerprints join a model-group; the first tenant of a group owns/exports,
  the rest import.
- **Tenant crash mid-mapping.** The shared arena's lifetime must outlive any
  single tenant. The arena is **kmod-owned** (extend the T4.6.4 pool
  ownership); POSIX-FD handles are refcounted, `fput` at 0; the kmod's
  existing `do_exit` reaper releases a crashed tenant's mapping without
  affecting peers. The owner role is the kmod's, not tenant 0's — so even the
  first tenant crashing does not strand the arena.
- **Partial unmaps.** A tenant leaving unmaps its VA range only; the physical
  pages persist while refcount > 0.
- **Fallback.** A tenant whose fingerprint matches no group, or whose import
  fails (VMM error), falls back to independent `from_pretrained` allocation —
  today's behaviour, no regression — and logs the fallback.

## §6 — Verification plan — proving sharing actually happens

Three independent checks; (b) is the structural proof.

- **(a) Total FB.** Run N same-model tenants; total `nvidia-smi` framebuffer
  must be ≈ `1 × W + N × ε` (one weight copy + per-tenant activations/KV), not
  `N × W`. This is the Diagnostic-1 measurement inverted: 4 TinyLlama tenants
  should total ≈ 2.72 GB + N·ε, not 10.9 GB. (Caveat: per-process
  `--query-compute-apps` accounting for VMM-imported memory may be
  ambiguous — **total FB is the ground truth**, not per-process numbers.)
- **(b) Physical-page identity.** `cipher_kv_bridge.page_info(devptr)` on a
  weight tensor in tenant 0 and the corresponding tensor in tenant 1 must
  report the **same physical pages** (same VMM physical handles). This is a
  direct structural proof of sharing, independent of memory accounting.
- **(c) Correctness.** A shared-weight tenant runs the *same physical bytes*
  as the owner → its decode must be **bit-identical** to its own-copy
  baseline. Teacher-forced logit-KL must be ~0 (not just ≤ 0.1) — any nonzero
  KL indicates the shared memory is not faithfully the weights.
- **(d) Tenant-ceiling.** Measure how many same-model tenants fit before OOM,
  with sharing vs without — should rise ≈ N×.

## §7 — Build sub-components + time estimates

| SC | scope | est. |
|---|---|---|
| **2** | Weight-arena VMM allocation + export/import. Extend `cipher_rt_kv_alloc` with a weight-arena slab; verify a peer process can import tenant 0's arena and read identical bytes (DtoH `memcmp`). | ~1 day |
| **3** | Model-load integration. Tenant 0: load weights into bridge-VMM tensors + export. Peers: import + wrap as `nn.Parameter` storages, skip the safetensors read. The fiddly part (HF/PyTorch storage rebind). | ~2 days |
| **4** | Model-group fingerprinting (libcipher_v2) + fallback path. | ~0.5 day |
| **5** | Lifetime: kmod-owned weight arena, refcount, crash teardown (extend T4.6.4 ownership). | ~0.5–1 day |
| **6** | Verification at scale: N-tenant FB measurement, `page_info` physical-page identity, correctness gate, tenant-ceiling. | ~0.5 day |

**Total ≈ 3.5–5 days** (consistent with the `FUTURE_SCOPE/B` estimate).
Anchor `a7ac8e97` may rotate when SC2–SC5 land interception/allocation code;
preserve `.pre_phaseC` per discipline.

## §8 — Recommendation

Proceed to SC2 with **interception point (c)** — model-load path via the T4.6
VMM bridge — and **mechanism: CUDA VMM POSIX-FD shareable handles**, the
verified T4.6.4 cross-process path, with the **kmod owning the arena
lifetime**. The build is largely *integration of existing, verified T4.6
infrastructure* (VMM allocator, the bridge, the export/import path, the
kmod pool) — not new substrate mechanism — which is why the estimate is days,
not weeks. Adjudicate the interception point and mechanism, then proceed.
