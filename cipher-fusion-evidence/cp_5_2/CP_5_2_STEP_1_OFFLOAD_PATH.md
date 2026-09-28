# CP 5.2 — Step 1: KV offload hierarchy — SCOPE MEMO

**Date:** 2026-05-17. **Status:** Step 1 of CP 5.2. Investigation complete —
scoped here for adjudication. **Step 1 is paper**: it maps the path, verifies
the design memo's assumptions against current pod state, surfaces
known-unknowns, and recommends a composition option. Step 2 is adjudication;
Step 3 is build. No code is written until the recommendation is approved.

**Anchors held:** kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2
`86618c30`, `cipher_kv_bridge` `8d6ffe3f`. Step 1 ships no code.

**Context in:** CP 5.1 closed — vLLM 0.20.2's per-layer KV buffers are owned by
CIPHER VMM (Option A hook on `_allocate_kv_cache_tensors`), correctness gate
PASS on TinyLlama + Mistral-7B. CP 5.2 adds the offload/tiering layer above
that owned KV.

---

## §(a) — The offload path, as mapped + design-memo assumptions verified

Six findings. F1–F4 answer the four assumptions the adjudicator asked be
checked against pod state; F5 is the single most consequential structural fact
for CP 5.2 and gets its own subsection; F6 is a granularity constraint that
falls out of F1–F5.

### F1 — CIPHER VMM page-tracking is device-resident-only

`cipher_rt_kv_alloc.c`: `struct page_slot` = `{ uint8_t state; CUmemGeneric‑
AllocationHandle handle; }`, with `enum page_state` = `FREE | RESERVED |
MAPPED` only. `map_one()` (`:145`) hardcodes the physical page as device
memory — `prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE`, and
`cuMemSetAccess` likewise `CU_MEM_LOCATION_TYPE_DEVICE`. **There is no
host-memory-backing concept and no residency-tier state** — a page is either
unbacked or device-resident, full stop. *Verdict:* the allocator as it stands
cannot represent a host- or NVMe-resident page. Whether that needs to change
depends on the option chosen (see §c): the **recommended Option A does not
require it** — it snapshots KV bytes to a host buffer *outside* the allocator
and leaves every CIPHER slab device-resident and fully mapped. Only the
heavier Option B (physical page demotion) would force a `page_slot` /
`map_one` extension.

### F2 — Granularity is 2 MiB; the tag is per-slab; the lazy path is dormant

Page granularity = **2 MiB** — `cuMemGetAllocationGranularity(...,
CU_MEM_ALLOC_GRANULARITY_MINIMUM)` on H100/CUDA 13 (init log confirms
`page=2048 KiB`). The lazy-map path (`cipher_rt_kv_slab_ensure`) exists but is
**dormant on the KV path**: the CP 5.1 hook calls `vmm_zeros` →
`cipher_rt_kv_slab_create(tag, nbytes, nbytes, …)` with `initial_bytes ==
max_bytes`, so each per-layer slab is fully mapped at creation. The page tag
(`struct cipher_rt_kv_page_tag` — tenant/seq/layer/head_kv/role/content_hash)
is stored **per-slab** (`s->tag`), not per-page; `cipher_rt_kv_page_info()`
derives a page's tag by locating its owning slab. *Verdict on tag-preservation
under migration:* under the recommended Option A there is **no page
migration** — bytes are copied, the VA and slab are untouched, so the tag is
trivially intact. Under Option B, CUDA VMM remap keeps a page within its
slab's VA span, so the slab→tag association also survives for free. **Tag
preservation is not at risk under any option.**

### F3 — vLLM PagedAttention does NOT fault cleanly on a non-resident page

CUDA VMM (`cuMemMap`/`cuMemSetAccess`) is **explicit-mapping, not
demand-paged** — unlike UVM (`cudaMallocManaged`). A page is in exactly one of
three states for a GPU kernel: (a) device-mapped — fast; (b) host-mapped
(`CU_MEM_LOCATION_TYPE_HOST` physical, still `cuMemMap`+`SetAccess`-ed into the
GPU VA) — readable by the kernel over PCIe, ~50× slower than HBM; (c) unmapped
— **any kernel access is an illegal-memory-access crash; there is no
fault-and-migrate**. vLLM's attention kernels (FlashInfer/cuDNN on H100) read
KV straight from the device pointer and have no fault handler. *Verdict:* KV
may be moved off HBM **only for sequences not in the running decode batch** —
the running batch's KV must stay device-resident. This is the structural
reason offload is bound to the scheduler (see F5). NVMe has no GPU-mappable
location type at all → an NVMe tier is necessarily copy-based, never mapped.

### F4 — Hot/cold tracking belongs in userspace (vLLM scheduler), not the kmod

The kmod (`cipher_kvdedup.c`, ABI `cipher_kvdedup.h`) owns **only the
cross-tenant dedup refcount table** — `INIT/PUT/CONFIRM/FREE/STATS` ioctls. It
has **zero GPU-memory-access observation**: no page-access hook, no PTE
accessed-bit, nothing that could tell it which KV pages are hot. The
allocator's `page_slot` likewise carries no access counter or LRU timestamp,
and CUDA VMM exposes no per-page accessed bit to userspace. The **only**
component that knows which blocks are hot is vLLM's Python scheduler /
`KVCacheManager` — it owns the running set and drives preemption. *Verdict:*
the tiering manager lives in **libcipher_rt userspace + a vLLM scheduler
hook**. The kmod is not in the offload path. **The kmod ABI is untouched by
CP 5.2 → anchor `e2f50452` does not rotate.**

### F5 — vLLM v1 preemption is recompute-only: there is no KV swap path

This is the load-bearing fact. vLLM **v1 has preemption but no swapping.**
`Scheduler._preempt_request()` (`v1/core/sched/scheduler.py:965`) does:

```python
self.kv_cache_manager.free(request)      # blocks → freelist
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0          # ← KV is DISCARDED
```

`num_computed_tokens = 0` means a preempted request, when re-scheduled
(`PREEMPTED` → `scheduled_resumed_reqs`, `:827`), **re-runs prefill from
scratch** — its KV cache is recomputed, not restored. v1 deliberately dropped
v0's CPU-swap space; there is **no swap-out / swap-in of KV to a secondary
tier** anywhere in v1.

Two corollaries that shape the whole CP:

1. **CP 5.2's offload IS the missing swap tier.** Instead of discarding KV on
   preempt and paying a full re-prefill on resume, CIPHER snapshots the KV to
   a host tier and restores it on resume with `num_computed_tokens` preserved.
   The capacity-extension headline (memo §4 #2) is exactly this: preemption
   stops costing a re-prefill, so more tenants can be parked/resumed at fixed
   context.

2. **`KVCacheManager.free()` is NOT a demotion hook.** Verified:
   `KVCacheManager.free` → `coordinator.free` → `BlockPool.free_blocks`, which
   is pure Python — `block.ref_cnt -= 1` and append zero-ref blocks to
   `free_block_queue`. **No GPU-memory operation.** The CIPHER VMM slab page
   stays mapped; only logical block *indices* return to a freelist. So CP 5.2
   cannot "intercept free and demote" — there is nothing physical freed at
   `free()`. The hook must be **snapshot-before-free / restore-before-alloc**:
   copy the request's KV bytes out of the still-mapped per-layer slabs at
   preempt time, copy them back into freshly-allocated blocks at resume time.

### F6 — Block↔page granularity mismatch (consequence of F2 + F5.2)

A vLLM "block" is a fixed token span (default 16 tokens). For Mistral-7B that
is ~64 KiB *per layer* (`2 × 8 KV heads × 128 head_dim × 2 B × 16`). A CIPHER
VMM page is 2 MiB → **one page ≈ 32 vLLM blocks**. `BlockPool.get_new_blocks`
pops blocks off a single shared `free_block_queue` in request-arrival order —
**not** per-sequence-contiguously — so one 2 MiB page typically holds blocks
belonging to many different sequences interleaved. CUDA VMM **cannot
`cuMemUnmap` below the 2 MiB granularity**, so "demote just sequence S's
pages" is mechanically impossible whenever S's blocks are page-shared with a
running sequence. *This bites only Option B* (physical page demotion). The
recommended Option A **dissolves it**: Option A never unmaps a VMM page — it
copies block bytes (per-block gather-copy, 64 KiB units) to host and leaves
the slab mapped — so the 2 MiB unmap-granularity constraint never applies.

### Design-memo §3 phrasing — flagged stale (not edited)

Memo §3 says "Hook the decode engine's KV-block allocator (vLLM
paged-attention block manager is the reference target)." CP 5.1 §(a)
established — and F5.2 reconfirms — that **vLLM v1 has no per-block GPU
allocator**: `KVCacheManager` / `BlockPool` are pure-Python integer
bookkeeping over the per-layer slabs allocated once at init. The real seam for
offload is the **scheduler's preempt/resume path** plus `BlockPool`'s freelist
policy, not a "block allocator." Flagged here for the adjudication trail (same
treatment as CP 5.1's stale `CUDA_INJECTION64_PATH` reference); not silently
edited.

## §(b) — Prior art that carries in

- **CP 5.1 KV ownership.** CIPHER VMM already owns vLLM's per-layer KV buffers
  and the slab stays fully mapped. CP 5.2 builds strictly above this; the
  CP 5.1 hook and its "slab fully mapped" invariant are preserved by
  Option A.
- **Phase 4.6 dedup primitive (`cp_4_6_5_6`).** Content-hash dedup
  (`cipher_rt_kv_dedup_*`, xxhash64 + full-page memcmp-verify, kmod refcount
  table, cuIpc POSIX-fd cross-process handles) — 100-proc validated. It
  carries in for the memo §1 "shared-prefix dedup on real traffic" bullet —
  but see §(d): dedup-on-real-traffic is a *separable axis* from offload.
- **`cipher_kv_bridge`** (`8d6ffe3f`) — the `vmm_zeros` / `page_info` /
  `get_stats` Python bridge. CP 5.2's snapshot/restore primitives would extend
  this module (new build artifact, non-anchor — as in CP 5.1).

## §(c) — Composition options

**Option A — snapshot-on-preempt / restore-on-resume (RECOMMENDED).**
Hook vLLM v1's scheduler. At `_preempt_request`, *before* `kv_cache_manager.
free(request)` recycles the blocks, gather-copy the request's KV block bytes
out of the still-mapped per-layer CIPHER slabs into a host-DRAM (warm) store
keyed by `request_id`. On `PREEMPTED`→resume, copy the bytes back into the
freshly-allocated blocks and preserve `num_computed_tokens` so the request
skips recompute.
- *Coverage / correctness:* the CIPHER slab stays fully mapped (CP 5.1
  invariant intact); no VMM unmap, so F6's 2 MiB granularity constraint never
  bites. Correctness bar = byte-identical output vs the recompute path.
- *Capacity extension:* host-DRAM snapshot store *is* the capacity beyond the
  HBM KV buffer; tenant-count-at-fixed-context rises because preemption stops
  costing a full re-prefill. NVMe = the same shape with a deeper store.
- *Cost:* scheduler preempt + resume hooks (a larger surface than CP 5.1's
  single monkey-patch); the per-block gather-copy GPU→host is the §6 central
  bandwidth risk and must be measured early; the resume-restore must complete
  before the model runner reads those blocks.

**Option B — physical VMM page demotion.** Actually `cuMemUnmap` cold CIPHER
pages and free HBM. *Rejected for CP 5.2 baseline / deferred.* CUDA VMM cannot
unmap below 2 MiB (F6); a preempted sequence's blocks are freelist-scattered
across pages shared with running sequences, so it needs a GPU→GPU compaction
into page-pure regions first, and it breaks the CP 5.1 "slab fully mapped"
invariant. Its only gain over Option A is reclaiming the *physical* HBM the
parked slab pages hold — revisit only if Option A's gate shows the
still-mapped slab is the binding capacity limit.

**Option C — host-mapped warm tier (pages stay GPU-mapped over PCIe).**
*Rejected.* By F3 a host-mapped page an active attention kernel touches is
read at PCIe rate *every decode step* → unbounded, unpredictable decode-latency
regression; and NVMe cannot be GPU-mapped at all. Listed to record that
mechanism (a) was considered.

**Option D — autonomous CIPHER tier manager (demote cold pages without the
scheduler).** *Rejected.* By F3+F4 CIPHER cannot know which pages a running
kernel may touch without the scheduler's running set; demoting a live page is
an illegal-access crash. It necessarily reduces to "hook the scheduler" =
Option A.

## §(d) — Recommendation for adjudication

**Recommended: Option A — snapshot-on-preempt / restore-on-resume**, scoped as
a **two-tier HBM ↔ host-DRAM** hierarchy for the CP 5.2 baseline, with the
**NVMe cold tier deferred** to the memo §5 upper-calendar-bound (NVMe is the
same copy-based shape, deeper). Option A is the lowest-risk path that delivers
a real capacity extension: it turns vLLM v1's recompute-on-preemption into
tier-restore, keeps the CP 5.1 fully-mapped-slab invariant, sidesteps the
2 MiB unmap-granularity problem entirely, and leaves both the kmod
(anchor `e2f50452`) and `cipher_rt_kv_alloc.c`'s `page_slot` unchanged — all
new work lands in libcipher_rt + a new userspace tier manager + a vLLM
scheduler hook.

**Separable axis flagged for the adjudicator — dedup vs offload.** Memo §1
bundles "KV offload tiering" with "shared-prefix dedup on real traffic." They
are independent: vLLM v1 *already* has native in-HBM automatic prefix caching
(`BlockPool.cached_block_hash_to_block`), so CIPHER's content-hash dedup adds
value specifically for **cross-engine-process** sharing (vLLM's prefix cache
is per-engine) and for **content-identical-without-shared-prefix-position**
matches. Whether dedup-on-real-traffic rides CP 5.2 or splits to a sub-CP is
an adjudication call — this memo recommends shipping **offload narrow first**
(Option A) and treating dedup wiring as a follow-on, but defers the decision.

**Known-unknowns to carry into Step 3 (not Step-1 blockers):**

1. **Per-block gather-copy bandwidth (the §6 central risk).** A snapshot
   gathers a sequence's blocks — scattered across freelist positions in *every
   layer's* slab — into a contiguous host buffer. Measure GPU→host (and
   host→GPU restore) bandwidth headroom against the decode-step budget early;
   the pre-registered latency bound is set in the Step-3 build memo.
2. **Sequence-block locality in `BlockPool`.** Two ways to make snapshot
   efficient: (i) opportunistic gather over stock `BlockPool` (works today,
   pays a scattered-copy cost), or (ii) a per-sequence page-affinity freelist
   policy hooked into `BlockPool` so a sequence's blocks land contiguously
   (structural, deeper hook, enables a single bulk copy). Step-3 build choice;
   surfaced so the adjudication sees the tradeoff.
3. **Resume-restore ordering.** The restore copy must complete before the
   model runner's attention kernels read the resumed sequence's blocks. The
   exact safe interception point on the `PREEMPTED`→running transition (and
   whether `num_computed_tokens` can be set non-zero cleanly without
   tripping a v1 invariant) is a Step-3 build detail.
4. **Eviction / admission policy & thrash (memo §6).** Which running request
   to snapshot under pressure, when to resume — a poor policy oscillates
   sequences across tiers. Policy + a pre-registered thrash bound are Step-3
   build memo work.
5. **NVMe tier** — deferred; copy-based, no GPU mapping (F3). Cover or
   explicitly keep deferred when Step 3 scopes.

**Gate to proceed:** adjudicate Option A (and the dedup-vs-offload split). On
approval, Step 3 builds the snapshot/restore tier manager + the vLLM scheduler
hook, two-tier HBM↔DRAM, gated on byte-identical output substrate-on vs off
(memo §4 #1) and a measured capacity extension within a pre-registered
latency bound (memo §4 #2). On rejection, re-scope here. No code is written
until §(d) is approved.
