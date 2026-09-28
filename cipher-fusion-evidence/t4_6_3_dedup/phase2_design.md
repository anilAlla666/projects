# T4.6.3 Phase 2 — content-hash dedup — DESIGN MEMO

Pre-build memo. No C written until this is approved. Based on an
end-to-end read of the current S2b `cipher_rt_kv_alloc.c` (355 lines).

## S2b assumptions dedup BREAKS — surfaced loud, not parenthetical

| # | Current assumption (file:line) | Why dedup breaks it |
|---|---|---|
| **A1** | `slab_free` / `shutdown` unconditionally `cuMemRelease(slot->handle)` (`:275`, `:344`) | A shared handle released while another tenant still maps it → that tenant's page becomes invalid. **Central change: handles become refcounted; release only at refcount 0.** |
| **A2** | `map_pages` `cuMemCreate`s a fresh handle per VA page (`:139`) | A dedup-HIT page must `cuMemMap` an *existing* handle — no `cuMemCreate`. |
| **A3 — DANGER** | `map_pages` `cuMemsetD8`-zeroes every freshly-mapped range (`:164`) | A dedup-HIT page already holds the shared tenant's content. **Zeroing it destroys deduped content.** Zero is MISS-only. This is the loudest, easiest-to-get-wrong item. |
| **A4** | `cipher_rt_kv_page_info` returns one tag per page (`:309`) | A shared page has multiple logical owners; one tag is ambiguous. Phase 2 (synthesized harness): acceptable if documented — tag = first/owning slab. |
| **A5** | `stats.pages_resident` counts VA pages 1:1 with physical (`:146`,`:278`) | A dedup-HIT maps a VA page but consumes **zero** new physical memory. Need separate counters: `physical_pages_resident` (unique handles) vs `virtual_pages_mapped`. The dedup win *is* virtual − physical. |
| **A6** | `page_slot` has no "shared" notion (`:23`) | The slot must record whether its handle is owned-fresh or shared, so `slab_free` routes release through the refcount table. |
| **A7** | content is unknown at `slab_create`/`map_pages` time | KV content exists only *after* it is written. Dedup is therefore a **post-write** operation, not a map-time one — which is exactly why Phase 2 tests on synthesized known-content pages and why live-decode write-path wiring is T4.6.4+. |

## Concurrency — the overarching decision: single mutex, NOT lock-free

Keep S2b's single global `g.mu`; extend its coverage to the new
content-hash table + refcounts. Rationale: slab create/free is a
**per-request** event, not per-token — a cold path; contention is a
non-issue. Under one mutex, find-or-insert + refcount inc/dec + map +
release are one serialized critical section. This **dissolves** the ABA
question and the concurrent-map race by construction — they are
lock-free hazards and there is no lock-free code. Lock-free refcounting
+ a lock-free hash table would add real ABA hazards for zero benefit on
a cold path. Decision: **lock, deliberately.**

## The five points

1. **Refcount semantics.** New global content-hash table: per bucket a
   chain of `{ uint64_t xxh; CUmemGenericAllocationHandle handle;
   uint32_t refcount; CUdeviceptr verify_va; }`. `refcount` is a plain
   `uint32_t`, **not atomic** — every access is under `g.mu`. No atomic
   ops, by the concurrency decision above.

2. **Free-on-zero.** `slab_free`, under `g.mu`, per MAPPED page: always
   `cuMemUnmap` the VA (per-VA mapping); look up the handle's table
   entry; `--refcount`; if it hits 0 → `cuMemRelease(handle)` + remove
   the entry — on the calling thread, still under `g.mu`. Hitting 0 and
   removing the entry are atomic w.r.t. the mutex, so no 0-refcount
   window is observable. `cuMemRelease` under the mutex is safe (driver
   call, no re-entry into CIPHER).

3. **Hash collision policy — xxhash64 key + true byte-compare on every
   hit** (per your recommendation). A collision degrades to a missed
   dedup, never to corruption; dedup correctness is independent of hash
   strength.

   **memcmp mechanism (explicit).** The compare runs **on the host**.
   On a hash-bucket match: `cuMemMap` the stored handle into a single
   2 MiB scratch VA reserved once at allocator init; `cuMemcpyDtoH` both
   the candidate page (new VA) and the stored page (scratch VA) into two
   2 MiB pinned host buffers allocated once at init; `memcmp` the full
   2 MiB; `cuMemUnmap` the scratch VA. Equal → real dedup; unequal →
   xxhash64 collision → MISS + chain a new bucket entry. Per-compare
   latency ≈ **0.4 ms** (cuMemMap ~1 µs + 2× DtoH-to-pinned ~80 µs each
   + full `memcmp` ≤200 µs + cuMemUnmap ~17 µs); off-hot-path
   (per-request slab op, not per-token). Constant overhead — one scratch
   VA + two pinned host buffers (~6 MiB), not per-page.

   **No CUDA kernel.** The verify is `cuMemcpyDtoH` + libc `memcmp`
   (~25 LOC of C). The "new-kernel LOC budget" and "kernel-pre-ship
   test" items are therefore **N/A** — there is no kernel.

   **Strict-byte-identical constraint (stated so future readers do not
   drift to the cheap-but-wrong option).** Indicator (b) is strict
   byte-identical. The verify MUST be a true full-2 MiB byte compare.
   Re-hashing the page with a second/stronger hash and comparing the
   hashes is **disqualified** — that is double-hashing, not
   byte-comparison; a second-hash collision (vanishingly rare but
   nonzero) would admit a corrupt dedup. `memcmp` over the actual bytes
   is the only accepted mechanism.

4. **Failure mode — `cuMemMap` fails after refcount bump.** Eliminated
   by ordering: on a dedup-hit do (1) memcmp-verify, (2) `cuMemMap` the
   shared handle at the new VA, (3) **`++refcount` only on map success.**
   The bump is the last step → no rollback path exists or is needed. On
   `cuMemMap` failure that page falls back to a fresh (MISS) allocation.

5. **Concurrent-map race.** Under `g.mu` the find-or-insert is one
   critical section: hash → bucket → walk chain with memcmp → bump
   existing or insert new. Two tenants hash-hitting the same handle
   serialize on `g.mu`; the second sees `refcount ≥ 1` and bumps. No
   lock-free find-or-insert ⇒ no ABA.

## Zero-on-map split (A3 — implementation, single conditional in `map_pages`)

`map_pages` is the function that decides miss-vs-hit per page — it is
the one that either `cuMemCreate`s a fresh handle or `cuMemMap`s a
deduped shared handle. The zero therefore stays a single conditional
**local to `map_pages`**, never a parameter passed from a caller — the
security-critical invariant lives in one function:

- **Miss** (fresh `cuMemCreate`d handle) → `cuMemsetD8(page, 0, 2 MiB)`.
  Preserves op #1's invariant: a 2 MiB page may be physical memory just
  freed by another tenant; zeroing prevents a cross-tenant leak and
  satisfies `StaticLayer` zero-init.
- **Hit** (existing shared handle mapped in) → **skip the zero.** The
  page already holds the shared content; zeroing destroys it (A3).

`cuMemSetAccess` stays batched over the whole range — every VA mapping
needs access granted, hit or miss. Only `cuMemsetD8` moves into the
per-page loop, gated on the miss/hit flag `map_pages` itself set.

## Phase 2 scope — no drift

Build the content-hash dedup mechanism in `cipher_rt_kv_alloc.c` (new
post-write dedup entry point + the refcount table + A1–A7 fixes). A
Phase-2 harness synthesizes KV pages carrying the Mooncake
block-equality structure with real bytes, drives the allocator, and
measures `dedup_ratio_real / dedup_ratio_sim` + the three binding
indicators (counters match sim within n=5 bounds; **strict**
byte-identical SHA-256 readback; refcount integrity → all freed, no
leak/double-free). Wiring dedup into the live decode write-path is
**T4.6.4+, not Phase 2.**

Gate: indicators pass per flavor; allocator unit test 14/14 unchanged;
anchors `55ab8c0c` / `86618c30` unchanged; taint 12288.

**Awaiting approval of this memo before writing C.**
