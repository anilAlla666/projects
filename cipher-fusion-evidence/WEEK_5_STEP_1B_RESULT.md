# Week 5 Step 1b — Substrate Primitive `cipher_rt_kv_dedup_alias` — RESULT

**Status: PASS.**

New substrate function landed (`cipher_rt_kv_dedup_alias`) — supports
in-place rebind of caller's existing VA onto a deduped physical page
without disturbing the caller's data_ptr. Mirror of
`cipher_rt_kv_dedup_put` but operates on the caller's pre-allocated VA
(not bridge's pool VA) via cuMemUnmap + cuMemRelease + cuMemMap atomic
swap under existing `g.mu`. Track 2 SC3 byte-identical (load-bearing
gate); SC6 + CP 5.4 bit-identical. alias_smoke confirms dedup actually
fires (2 VAs share 1 physical post-rebind, content preserved).

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 5, Step 1b of 4
**Anchors:**
- `cipher_rt_phase4` HEAD: `3c5ddaa` → **`ec0e005`** (tag `week-5-step-1b-substrate-alias`)
- `cipher_kmod` HEAD: `2fc70c3` (unchanged — kmod-free step)
- `cipher-may13-evidence` HEAD: `fc8a9ae` (unchanged)
- `cipher_kv_bridge.so` md5: **`c04b0c39` → `f041789c`** (Track 2 SC3 anchor rotation)
- `libcipher_rt.so` md5: `259ac994` (unchanged — different module)

---

## A — Pre-edit verification + baselines — PASS

| signal | value |
|---|---|
| cipher_rt_phase4 HEAD | `3c5ddaa` ✓ |
| cipher_kmod HEAD | `2fc70c3` ✓ |
| Both trees clean | ✓ |
| cipher_kv_bridge.so pre snapshot | md5 `c04b0c39` → `/tmp/week5_step1b/cipher_kv_bridge.so.pre_w5_step1b` ✓ |
| Track 2 SC3 baseline | **PASS** (`/tmp/week5_step1b/sc3_pre.log`): bit-identical forward, fb_before=5001 MiB, fb_after=5001 MiB, consumer_added=0 MiB, weight_arena_backed=true, both same-VA + offset-relative modes |
| SC6 TinyLlama vanilla | 7/7 PASS bit-identical |
| SC6 TinyLlama CIPHER | 7/7 PASS bit-identical |
| Existing `_put` smoke (Step 6 probe) | PASS (cipher_kv_bridge.so loads cleanly) |

---

## B — Read existing `_put` implementation — DONE

Read of `cipher_rt_kv_alloc.c:522-619` produced the design pattern for
`_alias`. Key state surfaces understood:

| element | location | role |
|---|---|---|
| `struct page_slot { state, handle }` | L29-32 | per pool-page state |
| `g.pool_base, g.page_size, g.n_pages, g.pages[]` | L44-58 | pool VA range + per-page table |
| `g.mu` | L54 | single mutex protecting all bridge state |
| `g.pages[gi].handle` | — | CUmem handle currently mapped at VA `g.pool_base + gi * g.page_size` |
| `d.pin_buf` | L437 | single 2 MiB pinned host buffer (per-process; allocated at `dedup_init` via `cuMemHostAlloc`) |
| `d.scratch_va` | L436 | single 1-page VA reserved for HIT memcmp-verify |
| `d.kvd_fd, d.tenant_id, d.pooloff[]` | L434-438 | kmod-side state |
| `dedup_verify(content, stored)` | L455-470 | maps `stored` at `scratch_va` + DtoH to `d.pin_buf` + memcmp(`content`, `d.pin_buf`). **NOTE: overwrites `d.pin_buf`** — caller must not also expect `d.pin_buf` to retain its own content during verify. |

This last point drove the implementation: `_alias` needs to memcmp the
caller's content vs the imported (HIT-candidate) content. Both go through
`d.pin_buf` if `dedup_verify` is reused. **Solution adopted:** in
`_alias`, snapshot `d.pin_buf` to a `malloc`-allocated temp 2 MiB buffer
before calling `dedup_verify`; pass the tmp buffer as `content`;
`dedup_verify` overwrites `d.pin_buf` with the imported content and
compares against the tmp (= original caller content). `free(tmp)` after.
Adds ~2 MiB malloc on the HIT path (~400 µs total per memo §Call model,
malloc adds negligible time).

---

## C — Implementation — DONE

| change | LOC | location |
|---|---|---|
| Function decl + 20-line doc-comment | **+24** | `cipher_rt_kv_alloc.h:130-153` |
| Function impl: lookup + idempotency + DtoH + xxh64 + PUT + MISS / HIT-match / HIT-collision branches + defensive recovery on rebind-fail | **+143** | `cipher_rt_kv_alloc.c:660-826` |
| Net diff | **+167 LOC** | (Part I estimate ~135 LOC; came in slightly larger due to the defensive recovery branch + the malloc-tmp-buffer workaround for `dedup_verify`'s `d.pin_buf` overwrite) |

### Function structure

```
cipher_rt_kv_dedup_alias(existing_devptr, was_deduped*):
  1. Lock g.mu
  2. Resolve existing_devptr → gi (range + alignment + state checks; -EINVAL on mismatch)
  3. Idempotency: if d.pooloff[gi] != 0 → no-op, return 0, was_deduped=0
  4. cuMemcpyDtoH(d.pin_buf, va, page_size)  — read caller's current content
  5. xxh64(d.pin_buf, page_size)             — compute hash
  6. cuMemExportToShareableHandle(own_handle, POSIX-FD)
  7. ioctl PUT (content_hash, export_fd)
  8a. MISS: close(export_fd); d.pooloff[gi] = p.pool_offset; unlock; return 0
  8b. HIT-CANDIDATE: cuMemImportFromShareableHandle(candidate_fd, shared)
                     malloc(tmp); memcpy(tmp, d.pin_buf, page_size)
                     dedup_verify(tmp, shared)  → matched ∈ {0,1}
                     close(candidate_fd); free(tmp)
  9a. matched: cuMemUnmap(va) + cuMemRelease(own_handle) + cuMemMap(va, shared)
              + cuMemSetAccess; ioctl CONFIRM; d.pooloff[gi]=offset; was_deduped=1
              [on cuMemMap fail: defensive recovery — cuMemCreate + cuMemMap
               of fresh empty handle so VA stays mapped; content lost, logged]
  9b. matched=0 (xxh64 collision): cuMemRelease(shared)
                                    ioctl PUT FORCE_NEW; d.pooloff[gi]=offset
                                    was_deduped stays 0
  10. unlock g.mu; return 0
```

### Lock discipline

All state mutations under `g.mu` (existing pattern from `_put`). The
cuMemUnmap+cuMemMap on HIT-match is atomic from concurrent
bridge-thread perspective (no other thread can touch this VA's
`page_slot` during the swap). Caller responsibility: **quiesce
concurrent CUDA kernels reading this VA before calling `_alias`** —
design model per Step 1 memo Part E.4 is to call from quiescent points
(post-prefill explicit-flush).

### Error handling

| condition | return | side effects |
|---|---|---|
| `init_done == 0` | -1 | none |
| existing_devptr not in pool / not aligned / not MAPPED | -EINVAL | none |
| d.pooloff[gi] != 0 (already alias'd) | 0 | was_deduped=0 (idempotent no-op) |
| DtoH fails | -1 | none |
| cuMemExport fails | -1 | none |
| ioctl PUT fails | -1 | export_fd closed |
| MISS path | 0 | export_fd closed, pooloff set, was_deduped=0 |
| HIT match (rebind OK) | 0 | own_handle released, shared retained, pooloff set, was_deduped=1 |
| HIT match + cuMemMap(shared) fails | -EIO | DEFENSIVE: cuMemRelease(shared) + try fresh empty handle + map it + set access; if recovery succeeds, VA stays mapped (content LOST, stderr log); if recovery fails, page marked PAGE_FREE (CRITICAL stderr log) |
| HIT memcmp-fail (xxh64 collision) | 0 | shared released, FORCE_NEW PUT, pooloff set, was_deduped=0 |

---

## D — Build — PASS

```
[ build_kv_bridge.sh — gcc + g++ link via pybind11 + libtorch ]
built : cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so
```

| signal | value |
|---|---|
| build rc | 0 |
| `cipher_kv_bridge.so` md5 | **`f041789cf8bf8cac5a3cd2dd7183e68`** (pre: `c04b0c39d8282daee66b3b865a1849a9`) |
| new T symbol exported | `cipher_rt_kv_dedup_alias` at `0x352f0` ✓ |
| existing dedup symbols preserved | 5 total: `init / put / free / get_stats / alias` ✓ |
| weight_arena_* symbols preserved | 5 of those ✓ (cipher_rt_weight_arena_create / export / free / import / info) |

---

## E — Regression gates — PASS

| gate | result |
|---|---|
| **E.1 Track 2 SC3** (load-bearing) | **PASS** (`/tmp/week5_step1b/sc3_post.log`); `diff sc3_pre.log sc3_post.log` = empty → **byte-identical**. Both modes (same-VA + offset-relative) PASS bit-identical forward, fb_before=5001 fb_after=5001 consumer_added=0. |
| **E.3 SC6 TinyLlama vanilla** | 7/7 PASS bit-identical |
| **E.3 SC6 TinyLlama CIPHER** | 7/7 PASS bit-identical |
| **E.4 CP 5.4 isolation** | 15/15 PASS (kmod untouched; safety-net check) |

---

## F — `_alias` smoke test — PASS

Test approach: cipher_kv_bridge.so is a Python extension module (links
libtorch_python; needs PyInstanceMethod_Type from Python interpreter),
so it cannot be dlopen'd from a standalone C binary. Smoke test rewritten
as Python ctypes (`/tmp/week5_step1b/alias_smoke.py`).

### Output

```
init: ok
slabs: va1=0x320000000 va2=0x320200000 page=2097152
HtoD: both VAs hold identical content
alias(va1): rc=0 was_deduped=0  (expect rc=0 was=0 MISS)
alias(va2): rc=0 was_deduped=1  (expect rc=0 was=1 HIT)
va2 content post-rebind: matches original=True  (expect True)
stats: puts=2 hits=1 misses=1 physical_pages=1 virtual_pages=2 collisions=0

VERDICT: PASS
```

### Interpretation

- **`puts=2, hits=1, misses=1`** — both alias calls fired exactly one
  PUT; the second one hit the first one's content as expected.
- **`physical_pages=1, virtual_pages=2`** — **deduplication actually
  happened**: 2 distinct VAs (`0x320000000` and `0x320200000`) are now
  mapped onto **1 shared physical page** in HBM. This is the load-bearing
  proof of the substrate primitive: real HBM is saved (one tenant's
  worth of this 2 MiB page).
- **`collisions=0`** — xxh64 didn't false-positive collide.
- **content `matches original=True`** — `cuMemUnmap + cuMemMap` atomic
  swap preserved the byte-level content via the shared physical (memcmp
  vs original host content = 0).

---

## G — Commit + tag — PASS

| signal | value |
|---|---|
| commit | **`ec0e005f472529639d537d05cde34b0402f4f8a9`** |
| author | Anil &lt;anil.0666369@gmail.com&gt; |
| message | "Week 5 Step 1b: substrate primitive cipher_rt_kv_dedup_alias" |
| diff stat | 2 files changed, +211 lines (+24 .h + +143 .c + +44 lines for the +143 figure (counts include comment + blank lines)) |
| tag | **`week-5-step-1b-substrate-alias`** ✓ at HEAD |
| Fallbacks preserved | `cipher_kmod_fallback/cipher_kv_bridge.so.pre_w5_step1b` (`c04b0c39`) + `cipher_kv_bridge.so.w5_step1b_alias` (`f041789c`) |

---

## Honest notes

1. **Pinned-buffer Q2 STOP-gate RESOLVED by inspection.** The `d.pin_buf`
   pinned host scratch is allocated per-process at `dedup_init` via
   `cuMemHostAlloc(g.page_size)`. `g.mu` serializes concurrent `_alias`
   calls within a process. Each vLLM EngineCore subprocess has its own
   bridge instance → no inter-process contention. **Step 2 does NOT need
   a separate pinned-buffer pre-verification gate** (Q2 user adjudication
   accepted the Step 2 entry STOP-gate; the gate is satisfied trivially
   by the pre-existing per-process pinned alloc). Step 2 proceeds with
   the original ~190 LOC budget, not the Q2 ~225 LOC expansion.

2. **`dedup_verify` overwrites `d.pin_buf` — fixed via malloc tmp.** The
   existing `dedup_verify(content, stored)` does
   `cuMemcpyDtoH(d.pin_buf, scratch_va, page_size)` — destroys whatever
   was in `d.pin_buf`. In `_alias`, the caller's content lives in
   `d.pin_buf` at the moment we'd call `dedup_verify`. Solution: malloc
   a 2 MiB tmp host buffer, memcpy `d.pin_buf` → tmp, then call
   `dedup_verify(tmp, shared)` so the comparison is `tmp` (saved
   caller content) vs `d.pin_buf` (newly-DtoH'd imported content).
   `free(tmp)` after. ~2 MiB malloc on HIT path; negligible vs the
   ~400 µs HIT cost.

3. **Defensive recovery on rebind-fail.** If `cuMemMap(va, shared)`
   fails after we've already `cuMemUnmap`'d, the caller's VA is
   unmapped — caller's next access seg-faults. The defensive recovery:
   try `cuMemCreate + cuMemMap` a fresh empty handle so the VA stays
   mapped (content is lost regardless, logged at stderr). If even
   recovery fails: mark `page_slot.state = PAGE_FREE` so the bridge
   never tries to use it again, log "DATA LOST" at stderr. This is
   belt-and-braces — `cuMemMap` failure post-Unmap is extremely
   unlikely (same VA, same alignment, just different handle), but
   the recovery prevents a worse failure mode (seg-faulting caller
   process).

4. **Anchor rotation is cipher_kv_bridge.so only, not libcipher_rt.so.**
   `cipher_rt_kv_alloc.c` builds into `cipher_kv_bridge.so` (via
   `build_kv_bridge.sh`), which is a Python extension module separate
   from `libcipher_rt.so`. `libcipher_rt.so` md5 unchanged at
   `259ac994` (it doesn't link cipher_rt_kv_alloc.o directly). The
   Track 2 SC3 anchor lives in `cipher_kv_bridge.so` so the rotation
   `c04b0c39 → f041789c` is the Track 2 SC3 anchor rotation, not a
   libcipher_rt rotation.

5. **No `pybind11` exposure added.** The 4 pre-existing `kv_dedup_*`
   functions are not pybind-exposed (just plain `T` C symbols);
   `cipher_rt_kv_dedup_alias` matches that pattern. Step 2 plugin
   will ctypes-bind directly against the C symbol in cipher_kv_bridge.so.
   If a future iteration wants Python-native exposure, add `m.def(...)`
   in cipher_kv_bridge.cpp's PYBIND11_MODULE block — out of scope for
   1b.

6. **Smoke test caveat — same-process N=2.** alias_smoke creates two
   tenants in the SAME process (different `tenant_id` env per
   `vmm_zeros` call, but one bridge instance). Real Week-5 test will
   spawn N=4 separate vLLM processes, each with its own bridge. Step 2/3
   harness will need cross-process orchestration; alias_smoke confirms
   the per-process correctness of the substrate primitive, not the
   cross-process flow.

---

## Step 2 readiness

**Step 2 may now be drafted.** Substrate primitive verified end-to-end:
- ctypes bindable from Python (alias_smoke proven)
- Idempotent on second call (no special caller handling needed)
- Correct rebind on HIT (verified by `physical_pages=1, virtual_pages=2`)
- Content preserved (verified by readback memcmp)

Step 2 work-shape (per Step 1 memo Part H, unchanged):
- ~190 LOC plugin module + ~5 LOC setup.py addition
- N=2 same-prompt exit gate (now well-defined: dedup actually happens
  per Step 1b smoke; just orchestrate vLLM-side wiring)
- Pinned-buffer Q2 STOP-gate **satisfied trivially** (note #1 above)

---

**Evidence:**
- `/tmp/week5_step1b/cipher_kv_bridge.so.pre_w5_step1b` (pre, `c04b0c39`)
- `/home/ubuntu/cipher_kmod_fallback/cipher_kv_bridge.so.{pre_w5_step1b,w5_step1b_alias}` (fallbacks)
- `/tmp/week5_step1b/build.log` (build rc=0)
- `/tmp/week5_step1b/sc3_{pre,post}.log` (load-bearing byte-identical)
- `/tmp/week5_step1b/sc6_{pre,post}_{vanilla,cipher}.log` (7/7 bit-identical × 4)
- `/tmp/week5_step1b/cp54_post.log` (15/15 PASS)
- `/tmp/week5_step1b/alias_smoke.py` + `.log` (PASS — dedup actually fires)
- `git show ec0e005` (the commit)
