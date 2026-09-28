# BUILD — distinct-model pager residency state machine (G-O2/G-O8/G-O9), STEP 1 proper

**Date:** 2026-05-31. **Type:** BUILD (production source, default-OFF, additive, OFF byte-identical; deployed
`1f305ce6` UNCHANGED, staging `.so` `29686678`→`c1cf3696`). First committed production code of the pager arc.
**Result: the residency state machine + the eviction-during-use coherence invariant are PROVEN under contention
(700 evictions racing 11,147 serve-reads → 0 corruption, 0 faults), and a negative control proves the test can
detect a violation. NOT yet serving real models — torch/vLLM weight-allocation routing is the next op.**

---

## STEP 0 — probe-2 caveat closed
`easy-install.pth` only adds the plugin *path* (no import); a clean re-run of the torch-pluggable-allocator-
behind-cuMemMap probe (`pager_torch_probe.py`, bare torch, no vLLM) is **noise-free**: W.data_ptr=0x304000000,
mapped 1→0→1, `torch.matmul` after evict+pagein+restore == before, **max_abs_diff 0.00**, no cipher-plugin
output, no segfault. The earlier noise was other commands in that block, not the probe.

## The module (`cipher_rt_pager.c` / `.h`, cipher_rt_phase4, default-OFF, additive)
Per-region: reserved VA + pinned-RAM warm copy + HBM handle + `state {EVICTED/PAGING_IN/RESIDENT/PAGING_OUT}` +
`ref` (both guarded by a per-region mutex) + instrumentation (cold_miss / pagein_cnt / evict_cnt; global
pages-in-flight = PCIe queue depth). VMM mirrors the proven `cipher_rt_kv_alloc.c:127-208` pattern
(cuMemCreate/cuMemMap/cuMemSetAccess). Symbols in the `.so` (`nm -D`: cipher_pager_init/register/page_in/
page_out/serve_begin/serve_end/state/get_stats); **NOT wired into `cipher_v2_init_body` → OFF byte-identical**.

**Eviction-safety mechanism (the hard part, advisor-corrected from my first design):**
- A lock-free acq_rel protocol here is a **StoreLoad/Dekker race** (serve stores ref+loads state; page_out
  stores state+loads ref). → **per-region mutex** serializes {state, ref}, held O(1), **never across GPU work**.
- serve hot path: `serve_begin` (lock; check RESIDENT; ref++; unlock) → [caller issues GPU read of va] →
  `serve_end` (lock; ref--; unlock). **No sync on the serve path.**
- `page_out` (the evictor, off the serve path): CAS RESIDENT→PAGING_OUT (new serves now MISS) → drain `ref==0`
  (all serves finished issuing) → **`cuCtxSynchronize()`** (all issued GPU reads complete; `cuMemUnmap` does NOT
  implicitly sync) → unmap+release → EVICTED. (Per-stream completion events to replace the global sync = STEP 2;
  the NSLOTS event-ring was dropped as a correctness hole under fan-out collision.)

## Correctness gates (`test_pager.c` — Mem #11, BEFORE any density/throughput claim)
- **GATE 1 — bit-identical after page-out→page-in: PASS** (checksum of va == warm copy, across a full cycle;
  extends the proven sequential primitive).
- **GATE 2 — eviction-during-use under contention: PASS.** 8 serve threads (real 16 MiB DtoH GPU read + 3-offset
  checksum on per-thread streams, serve_end *before* the sync so the read is in-flight during eviction) racing a
  page-out/in churner (3 ms resident dwell): **hits=11,147 miss=113,072 corrupt=0 memerr=0, evicts=700,
  pageins=700** → race_exercised (hits>0 ∧ miss>0) ∧ coherent (0 corrupt ∧ 0 faults) → **never served
  half-evicted under contention.**
- **NEGATIVE CONTROL (`-DCIPHER_PAGER_NO_GPU_SYNC`, eviction sync removed): DETECTED.** First in-flight read
  during an unmap **faulted** (`memerr=1`, then the poisoned context cascaded to all-MISS) → proves (a) the test
  *can* fail, and (b) the `cuCtxSynchronize`-gated unmap is **load-bearing**, not decorative. It faulted on hit #1,
  so the race window is WIDE → the positive's 0 faults across 700×11k is a meaningful pass, not window-too-narrow.

**Precise scope of what the test proved (advisor):**
- The negative control removed the **GPU-sync**, not the mutex → GATE2 + the control prove the **eviction
  GPU-sync (cuCtxSynchronize) under contention**. The {state,ref} **StoreLoad race-freedom is correct
  BY CONSTRUCTION (mutual exclusion), not separately race-tested** (with the mutex in, there is no window to
  widen). Both halves are sound; only the GPU-async half was empirically exercised.
- The control fired the **fault (`memerr`) detector**, not the cksum (`corrupt`) detector — unmap-under-read
  manifests as a hard CUDA fault here, not silent stale data. The cksum path is present but unexercised as a
  true-positive (the dominant failure mode is caught).
- **Read-only-region assumption:** the warm copy never mutates and page_in always restores the same bytes —
  correct *because model weights are read-only*. The pager does not handle dirty/mutated regions (and need not,
  for weights). Explicit scope assumption.
- **OFF byte-identical SHOWN (not just argued):** the Marlin graph-gate probe against the new `.so` (`c1cf3696`,
  pager compiled in but unused) passes identically (capture succeeds, handled+4, MARLIN-INT4) → adding the pager
  changed no existing behavior.

## Scope (precise — not more than it is)
Proven: residency state machine + eviction-during-use invariant, on **representative regions** under contention.
**NOT** wired to vLLM weights. NEXT OP: **per-allocation routing** (so only weight allocs land behind the pager,
not KV/activation/scratch — `change_current_allocator` is process-global) + **vLLM-allocator coexistence** + N
distinct models each behind per-model regions; then STEP 2 (per-stream-event eviction + predictive prefetch vs
the aggregate-PCIe-under-correlated-bursts binding constraint), STEP 3 (multi-tenant isolation/billing).

## Anchors / discipline
Deployed `/usr/lib/cipher` `1f305ce6` UNCHANGED; staging `.so` `29686678`→`c1cf3696`. Source added:
`cipher_rt_pager.c`/`.h` + `test_pager.c` + Makefile (OBJS + rule); D.8 strays (cublas_shim/fairness) excluded.
Default-OFF, additive, OFF byte-identical (not in init; smoke-confirmed). Correctness passed before commit;
git-tagged. **Inherited debt (not introduced here):** the committed Makefile already lists `cipher_rt_fairness.o`
in OBJS while `cipher_rt_fairness.c` is an untracked D.8 stray → a clean checkout of this tag won't build without
the strays present. Pre-existing; flagged, not fixed in this commit.
