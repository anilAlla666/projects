# BUILD STEP 1 (probe-first) — the distinct-model pager PRIMITIVE is proven feasible on cu13 (both probes HOLD)

**Date:** 2026-05-31. **Type:** throwaway probes (NO production source change; deployed `1f305ce6` / staging
`29686678` / Marlin gate `graph-gate-step1` all unchanged). Per the probe-first discipline (advisor) the module
build was redirected to first prove the load-bearing primitive. **Result: the weight-behind-pageable-cuMemMap-VA
primitive HOLDS — including the make-or-break torch-ownership path. The distinct-model tiered-residence pager is
feasible; the residency state-machine module is now the worth-building next op.**

---

## The load-bearing question (advisor): can model WEIGHTS be paged via cuMemMap, for a kernel CIPHER doesn't own?
The pager assumes weights can sit behind a CIPHER-controlled, evictable/remappable VA, read by vLLM/torch's own
GEMMs. `cipher_rt_kv_alloc.c` only proves this for KV that CIPHER *owns* — weights are torch-owned, a different
problem. Two throwaway probes (no production source):

**PROBE 1 — a non-CIPHER cuBLAS GEMM reads through my cuMemMap VA + survives remap (`pager_va_probe.py`):**
Reserve VA → `cuMemCreate`+`cuMemMap`+`cuMemSetAccess` → copy known weight W in → `cublasGemmEx` identity matmul
(C=I·W=W, exact in fp16) with B=my-VA. Then `cuMemUnmap`+`cuMemRelease` (evict) → fresh HBM remapped to the SAME
VA + recopy (page-in) → GEMM again.
- round 1 (mapped): C == W, **max_abs_diff 0.00**.
- round 2 (after evict + page-in): C == W, **max_abs_diff 0.00**.
- ⇒ a kernel CIPHER doesn't control reads the weight through CIPHER's pageable VA; unmap→remap→recopy restores
  it bit-identically.

**PROBE 2 — the make-or-break: a torch-OWNED weight tensor behind a CIPHER-pageable VA (`pager_torch_probe.py`
+ `pager_probe_alloc.c`):** a tiny cuMemMap-backed `torch.cuda.memory.CUDAPluggableAllocator` (the documented
"own the allocation path" mechanism) → `W = torch.randn(...).cuda()` has its **storage = my cuMemMap VA**
(`W.data_ptr()=0x304000000`, mapped=1). `torch.matmul(x, W.T)`; then `cipher_evict(va)` (mapped→0, unmapped) →
`cipher_pagein(va)` (mapped→1, fresh HBM, same VA) → restore from RAM warm → matmul again.
- after evict+page-in+restore: `torch.matmul` == before, **max_abs_diff 0.00**.
- ⇒ **torch's own weight tensor can live behind a CIPHER-pageable VA, be evicted (HBM freed) and paged back, and
  be read correctly by a real torch matmul.** This is the real vLLM-integration path (own the weight-alloc via
  the pluggable allocator — a torch config, not a framework source patch; substrate-line per Mem #24).

## What this de-risks vs what is still open
- **DE-RISKED (the foundation, precisely):** weights *can* be paged (HBM hot / RAM warm) via cuMemMap; CIPHER can
  own the weight VA (clean torch process) via the pluggable allocator; **round-trip restore (sequential
  evict→pagein→use) is bit-identical.** That is necessary and was the open question — but it is the *easy half*.
- **STILL OPEN — and these are the hard half, not line-items:**
  1. **Eviction-during-use is UNPROVEN.** Both probes did sync→evict→pagein→use (the case never in doubt). The
     coherence invariant that matters — evict *while a serve is in flight* (a full forward / graph replay over
     tens of ms; a per-serve refcount-spin/sync can't sit on the hot path) — is **silent in these probes by
     construction**. The module must prove **completion-event-gated unmap** or **evict-cold-by-construction**.
     This is the real engineering, still ahead.
  2. **Global-allocator caveat may invert the integration model.** `change_current_allocator` is process-global
     and must precede any CUDA alloc → **every** tensor (weights/KV/activations/scratch) goes behind cuMemMap,
     not just the weights we want to page. So (a) selective weight-paging needs **per-allocation routing** on
     top; (b) **vLLM may set/assume its own allocator/mempool** → a global override can collide. So "own the
     weight-alloc via pluggable allocator" is proven **as a mechanism in a clean torch process, NOT as a drop-in
     under vLLM.** Pager module + allocation-routing + vLLM-allocator-coexistence.
  3. **The residency state machine** (EVICTED/PAGING_IN/RESIDENT/PAGING_OUT; never serve half-evicted) + N
     distinct models each behind per-model VA regions, evict/page-in on bursty arrival.
  4. **Binding constraint (sized earlier):** aggregate-PCIe-under-correlated-bursts (51 GB/s serialized) + INT4
     quality — attacked by predictive prefetch in STEP 2.
- **Caveat on probe 2:** run under the CIPHER vLLM-plugin env (filtered the known atexit-segfault noise); the
  data_ptr/mapped/diff sequence is internally consistent, but worth a clean (no-CIPHER-plugin) re-run to be certain.

## NET / next-op decision
Foundation **proven feasible** (the pager is real, not hypothetical) — but "foundation proven" ≠ "most of the
work done": the foundation was the *precondition*; the **coherence model under concurrency (eviction-during-use),
the global-allocator routing, and vLLM-allocator coexistence are the actual engineering, still ahead.** Next op =
build the residency state-machine module (STEP 1 proper) on this primitive, with **event-gated/evict-cold eviction
as the coherence model (the proof these probes did NOT cover)** + instrumentation from day one — a **multi-turn
build**, not one turn. Then PCIe (STEP 2) + multi-tenant (STEP 3). No production source changed this turn; probes
are throwaway (`pager_va_probe.py`, `pager_probe_alloc.c`, `pager_torch_probe.py`).
