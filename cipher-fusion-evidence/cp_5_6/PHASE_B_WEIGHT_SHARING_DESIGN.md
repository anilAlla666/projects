# Phase B — Track 2: Cross-Tenant Weight Sharing — Design Memo

**Date:** 2026-05-18. **Type:** design, no substrate code. Substrate
`a7ac8e97`.

**Role — honest restatement.** The task framing calls weight-sharing the
"required supporting primitive" for cross-tenant batching. The Track 1 design
shows it is **not a functional prerequisite**: the process that runs the
batched B=N step reads weights once for all N rows because it is *one kernel
in one process* — it uses the one weight copy it already holds, regardless of
whether other processes share pages. Weight-sharing is instead a **capacity
primitive**: it lets the *non-executing* tenants drop their redundant weight
copies. That is genuinely valuable — capacity → tenant density → compounds
the concurrency factor — but it is composition-orthogonal to the batching
lever, not a blocking dependency. This memo designs it on those honest terms.

## 1. What it does, precisely

Today (Diagnostic 1, measured): N tenants = N processes, each runs stock
`from_pretrained(...).cuda()` and `cudaMalloc`s its **own full 2.2 GB copy**
of TinyLlama weights — four disjoint physical copies for 4 tenants (per-process
GPU memory 4×2720 MiB; disjoint `data_ptr`s; default `cudaMalloc` allocator;
CIPHER hooks never see weight allocation). Weight-sharing makes tenants 1..N
map **one** physical copy: weight HBM `N×2.2 GB → 1×2.2 GB`.

It is **not** a tok/W lift (Diagnostic 1: B=1 decode is overhead-bound, HBM
4–6 % utilised; ideal weight-sharing ceiling ≈1.05×). It is a capacity lift,
and capacity buys tenant count.

## 2. Architecture — interception point

Three candidate layers, in increasing invasiveness:

- **(A) PyTorch post-load rebind (recommended).** Tenant 0 loads the model
  normally. The substrate's Python shim (same install path as
  `cipher_spec_decode`) then, on tenants 1..N, **rebinds** each
  `nn.Parameter`'s storage to externally-owned device memory imported from
  tenant 0. No kmod work, no CUDA-allocator interception; reuses the existing
  substrate Python hook. The fiddly part is rebinding HF parameter storage to
  memory PyTorch's allocator did not create — done via
  `torch.cuda.caching_allocator` bypass: build tensors with
  `torch.frombuffer`-style device pointers, or load the model on a `meta`
  device and assign imported storages. Bounded, ~the bulk of the 3–5 days.
- **(B) `libcipher_rt` CUDA-allocator interception.** Hook `cudaMalloc` in the
  LD_PRELOAD substrate; when a tenant in a model-group allocates a region
  matching a registered weight arena, return a mapping of the shared arena
  instead. Transparent to PyTorch but fragile — must identify *which*
  allocations are weights vs activations/KV (size/lifetime heuristics).
- **(C) kmod-level mmap interception.** Rejected — device memory is not
  file-backed mmap; the kmod has no clean hook for `cudaMalloc`'d HBM. cuIpc /
  CUDA VMM are the supported cross-process device-memory paths.

Recommendation: **(A)**. Lowest risk, reuses substrate machinery, no driver
work. (B) is a fallback if rebinding HF storage proves intractable.

## 3. Cross-process sharing mechanism

Two CUDA-supported paths:

- **cuIpc handles** (`cuIpcGetMemHandle` / `cuIpcOpenMemHandle`). Exactly the
  mechanism CIPHER already ships in the **T4.6 cross-process page pool**
  (`cipher_kv_bridge.cpp`, `cipher_rt_kv_alloc.c`), proven real==sim 1.000 for
  cross-tenant KV-dedup (CP 4.6). Per-allocation handles; the weight set is
  many tensors → either one big contiguous weight arena exported as a single
  handle (preferred — load weights into one slab, one handle) or per-tensor
  handles (simpler load path, more handles).
- **CUDA VMM** (`cuMemCreate` + `cuMemExportToShareableHandle` +
  `cuMemMap`). More flexible (virtual-address control, partial maps) but a
  heavier API surface. Not needed if the weight arena is a single static slab.

Recommendation: **cuIpc, single contiguous weight arena**, directly extending
the T4.6 page-pool code. Tenant 0 allocates a 2.2 GB slab, loads weights into
it, `cuIpcGetMemHandle`; peers `cuIpcOpenMemHandle` and bind (§2A).

## 4. Fallback when tenants run different models

Weight-sharing applies only within a **model-group** — tenants running the
identical model + dtype. `libcipher_v2` already tracks tenant identity; extend
the registration with a model fingerprint (repo path + dtype + revision hash).
Tenants with matching fingerprints join a group and share; a tenant with a
unique fingerprint runs standalone with its own copy (today's behaviour — no
regression). Mixed-model pods degrade gracefully to per-group sharing.

## 5. Build estimate

**~3–5 days.** Breakdown: cuIpc weight-arena export/import on top of the T4.6
pool (~1 day, mostly reuse); the HF parameter-storage rebind path (§2A) and
making `from_pretrained` land weights in the shared slab (~2–3 days — the real
work, HF/PyTorch storage internals); correctness validation — a shared-weight
tenant must pass the Phase A teacher-forced / logit-KL gate bit-for-bit vs its
own-copy run (~1 day). No kmod changes.

## 6. Composition with Track 1 — and build order

The batched-decode executor (Track 1 §2.2) runs the B=N step on one weight
copy. Weight-sharing's contribution to that picture:

- **Form (A) dedicated executor:** the executor is one process with one copy
  — weight-sharing is *redundant* for the executor itself. It still helps if
  tenant client processes also hold model copies for any non-batched path;
  if clients are thin (no model), Track 2 is not needed at all.
- **Form (B) elected-tenant executor:** the executing tenant uses its copy;
  weight-sharing lets the other N−1 tenants drop theirs → `N×2.2 GB → 1×2.2 GB`,
  freeing HBM for KV / more tenants.

**Build order is not forced.** Track 1 is the critical path to the ≥3.6 ×
TPW target and does not depend on Track 2. Recommended sequencing:

1. **Track 1 prototype first** (1–2 weeks) — it is the lever; it carries the
   schedule risk (the plumbing-gap question); it must be de-risked first.
2. **Track 2 in parallel or immediately after** (3–5 days) — small, and it
   raises the tenant ceiling that Track 1's lift is multiplied against. Best
   landed before Track 1's N=16 production phase, where 16 weight copies
   (~35 GB for TinyLlama, far more for 7B-class) would otherwise cap density.

So: weight-sharing is worth building — as the capacity primitive it is — and
slots in alongside Track 1 without gating it. The TPW target is earned by
Track 1; Track 2 lets a GPU hold enough tenants for Track 1's N to be large.

## 7. Artefacts referenced

Diagnostic: `PHASE_B_DIAG1_WEIGHT_SHARING.md`, `phase_b_diag1/`. cuIpc prior
art: `cipher_kv_bridge.cpp`, `cipher_rt_kv_alloc.c`, `t4_6_3_dedup_report.md`.
Track 1: `PHASE_B_CROSS_TENANT_BATCHING_DESIGN.md`.
