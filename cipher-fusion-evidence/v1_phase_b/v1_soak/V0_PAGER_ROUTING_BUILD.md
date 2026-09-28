# BUILD — pager per-allocation routing (ORCHESTRATE G-O2, pager STEP 1.5)

**Date:** 2026-05-31. **Type:** BUILD (production source, default-OFF, additive, OFF byte-identical; deployed
`1f305ce6` UNCHANGED, staging `.so` `c1cf3696`→`7b5f4311`). Builds on `pager-step1-residency-eviction-coherence`
(77e23cc). **Result: a REAL model's weights are now routed behind a per-model pageable pager region — 100% of
params+buffers in-region, evict+pagein byte-identical and forward-correct, and TWO distinct models in disjoint
regions with per-region isolation (evict A while B resident → B untouched). The state machine now pages real
models. All 11 correctness gates PASS. Not yet wired into vLLM's internal load path — that is the next op.**

---

## STEP 0 — the routing signal (probe-first; the core unknown was "how does the allocator know a malloc is a weight?")
**Finding 1 (the collision the spec warned about, confirmed):** a *global* `change_current_allocator` BREAKS real
frameworks — `transformers`' `caching_allocator_warmup` calls `torch.cuda.memory_reserved/memory_stats`, which
`CUDAPluggableAllocator` does not implement (`getDeviceStats` → RuntimeError). vLLM calls the same APIs for KV
profiling. So a global override is **not viable** (and NOT scoped down to "clean-torch-only").

**Finding 2 (the coexistence-correct mechanism):** torch's **scoped `MemPool` + `use_mem_pool`** routes allocations
by *context* without replacing the global allocator — the default allocator (and its `memory_stats`) stays intact
for KV/activation. **The load-phase window IS the `with use_mem_pool(pager_pool)` block.** This is signal (a)
phase-marker *implemented via* (c) explicit ownership, and it is the general coexistence answer (any framework
that calls `memory_stats` keeps working). Probe (`pager_route_probe.c/.py`, TinyLlama): **IN-WINDOW 2190 MiB /
113 allocs = 1.044× param bytes; POST-WINDOW (forward) 0.0 MiB / 0 allocs; `memory_stats` works = True** →
`MEMPOOL PHASE-MARKER ROBUST`. Size-signature (b) rejected (KV cache is also large → not separable from weights).

## STEP 1 — the routing (cipher_rt_pager.c/.h, additive, default-OFF)
New API (`cipher_rt_pager.h`): `begin_load(model_key, reserve_bytes)` reserves+`pg_map`s a per-model region and
arms routing (`g_loading`); `cipher_pager_malloc/free` are the torch MemPool allocator — weight allocs **bump
from the region's cuMemMap VA** (512 B aligned), so they are pageable; under-reservation surfaces as a NULL/OOM
(never a silent `cudaMalloc` passthrough that would leave a weight non-pageable); `end_load` captures the warm
copy of `[0,bump)` and `PAGING_IN→RESIDENT`; `find(model_key)` is the N-model registry lookup; `live_cksum` is a
full chunked-D2H restore proof. `page_in` restores `warm_bytes` (the populated prefix). **`g_loading` is a single
global → loads are SERIAL (STEP-1 boundary; concurrent `begin_load`s would mis-route — fine for the N-model
sequential-load case, flagged for the future).** Mirrors `cipher_rt_kv_alloc.c` VMM.

## CORRECTNESS GATES (`pager_route_serve.py`; Mem #11, before any density claim) — 11/11 PASS
Advisor-hardened (each gap in the first design closed):
- **G-route — 100% residency (not the aggregate ratio):** iterate EVERY `named_parameters()` + `named_buffers()`,
  assert each `.data_ptr()` ∈ `[base_va, base_va+used_bytes)`. **A: 203/203 in-region, escaped=[]** (nothing leaked
  to the default allocator — not even rotary buffers). used=2190 MiB.
- **G-reserve:** load fit, no NULL mid-load (used 2190 MiB ≤ reserve).
- **G-det:** forward deterministic baseline `O1a==O1b` FIRST (so the O2 compare is vs zero, not vs variance).
- **G-cksum (PRIMARY, determinism-INDEPENDENT):** `live_cksum` before `page_out` == after `page_in`
  (`0x87032f0178f3deea` → identical) → the full real-model weight set round-trips byte-identical, independent of
  forward determinism. This is GATE1 lifted to a real model's entire weight set.
- **G-fwd (integration):** forward bit-identical after evict+pagein, **max_abs_diff = 0.000e+00**.
- **G-reg (two distinct models, per-region isolation):** TinyLlama (A) + Llama-3.2-1B (B), disjoint regions
  (`0x302000000` vs `0x398000000`); B 148/148 in its OWN region; registry `find()` keys→regions;
  **evict A while B RESIDENT → B forward unchanged (max_abs_diff 0)**; A restores → A forward correct
  (max_abs_diff 0). The registry/per-region isolation is demonstrated, not just compiled.

## G-reclaim — eviction actually FREES HBM (the density thesis; advisor catch, `pager_reclaim.py`)
The 11 gates prove eviction is byte-*correct* but would all pass even if `page_out` were a no-op. The whole
"N models on one H100" lever rests on `cuMemUnmap`+`cuMemRelease` returning the bytes to the **physical** free
pool. Verified with driver-level `mem_get_info()` (`cuMemGetInfo`, immune to torch caching): free **78070 →
80442 MiB on `page_out` = +2372 MiB = the full region**, and back to 78070 on `page_in`. **G-reclaim PASS /
G-reacquire PASS** → eviction physically reclaims, density is real, not a byte-correct no-op.

**Honest density-math caveats for the NEXT op (advisor; not regressions):** (1) the warm copy is **pinned host
RAM 1:1 with model size** — 100 models = 100× model bytes in scarce pinned RAM; the production version must page
from on-disk/mmap'd weights, not a resident host mirror. (2) `cuMemCreate` of the full aligned reserve
over-allocates ~10% physical HBM (2372 vs 2190 MiB here) — fine for correctness, matters when packing for density.

## OFF byte-identical + teardown caveat
**OFF byte-identical SHOWN:** Marlin graph-gate against the new `.so` (`7b5f4311`) passes identically (capture
True, handled+4, MARLIN-INT4) → routing additions are dead-until-called. **Teardown caveat (diagnosed, not the
pager):** ctypes-loading the full production `.so` into a host process attaches the cipher_v2 CUPTI subscriber via
the cuInit hook (the CLASSIFY logs); its teardown segfaults at interpreter exit. **CONTROL1 reproduces the 139
with NO pager and NO model** → it is a harness artifact of the non-injection (ctypes) load path, firing AFTER all
gates complete. `os._exit` past the finalizers gives a clean exit 0. The deployed injection path is unaffected.

## Anchors / discipline / next op
Deployed `1f305ce6` UNCHANGED; staging `.so` `c1cf3696`→`7b5f4311`. Source: `cipher_rt_pager.c`/`.h` (D.8 strays
excluded). Default-OFF, additive, OFF byte-identical. Substrate-line: torch pluggable-allocator config + driver
boundary (`use_mem_pool` / cuMemMap), **no framework source patch**. **NEXT OP: wire the routing into vLLM's
internal weight-load** (vLLM loads weights itself — wrap its model-load in `use_mem_pool(pager_pool)`; KV/profiling
already use the default allocator, so coexistence holds) + N-model serving harness; then STEP 2 (predictive
prefetch / per-stream-event eviction vs aggregate-PCIe-under-correlated-bursts). The transformers `.to()`-inside-
pool path proved the mechanism on real models; vLLM is the production integration.
**First probe of the vLLM op (advisor):** the open question is NOT coexistence (MemPool settled that) — it is
whether vLLM's weight loader allocates through a torch allocator that `use_mem_pool` actually CAPTURES (its newer
`CuMemAllocator`/sleep-mode path may bypass it). Probe that one fact before designing the wrap.
