# BUILD — multi-model co-residence residency manager (ORCHESTRATE G-O2, pager STEP 2-arch)

**Date:** 2026-05-31. **Type:** BUILD (production source, default-OFF, additive, OFF byte-identical; deployed
`1f305ce6` UNCHANGED, staging `.so` `7b5f4311`→`d6dfd5a2`). Builds on `pager-step1.5-routing-real-model` (2db11c3).
**Result: N distinct models co-resident in ONE process/HBM, each pageable via the proven per-region state machine,
demand-paged under concurrent serve with the eviction-during-use coherence PROVEN to hold ACROSS N regions
concurrently. 3 distinct 7-8B models served bit-identical under residency pressure. All correctness gates PASS.**

**HONEST SCOPE (advisor, binding): this proves the MECHANISM is CORRECT — it does NOT prove it BEATS orchestrating
N sleep-mode vLLM instances. The delta is UNMEASURED here.** Co-residence saves N×(CUDA context+libs), NOT weights
or CPU-backup (both approaches pay those; CIPHER's warm copy is pinned = scarcer). The real lever is (i) partial-
layer residency (next op); (ii) co-residence is the weakest leg and is sized below, not asserted.

---

## The manager (`cipher_rt_pager.c`/`.h`, additive, default-OFF: budget 0 = inactive)
`cipher_pager_mgr_init(budget)` sets a resident-HBM budget and evicts LRU down to it; `cipher_pager_serve_demand(id)`
is the demand-paging serve — HIT fast-path (region lock only) or, on MISS, under the manager lock: `mgr_make_room`
(evict LRU `ref==0` victims, skipping in-use models) → `page_in` → `serve_begin`. Per-region `last_used` (LRU);
`mgr_resident_bytes`/`mgr_wasted` instrumentation. Builds on the proven primitives unchanged.

**Lock discipline (advisor-reviewed): `g_mgr_mu` → region `mu` (HIT fast-path takes only the region lock); `g_reg_mu`
taken sequentially inside page_in, never nested → no cycle.** **Livelock fix (advisor catch #1): the
page_in→serve_begin is ATOMIC under `g_mgr_mu` so the demanded model holds a ref BEFORE the manager lock drops** —
otherwise a concurrent victim-selection could evict it in the gap (re-MISS → thrash). Known correct-but-serializing
(logged, not a bug): `make_room` holds `g_mgr_mu` across page_in's ~70ms H2D + cuCtxSync; resident HITs still proceed
lock-free; STEP-2 per-stream events fix it.

## CORRECTNESS — C-level N-region contention proof (`test_residency.c`; Mem #11, before any density claim)
The new invariant is N-region concurrent coherence + deadlock/livelock-freedom (single-region was proven in STEP 1).
- **GATE-A PASS:** 6 regions, budget = 3 → **13,040 evictions racing 37,930 serve_demands across 8 threads → 0
  corrupt, 0 cross-region, 0 fault, 0 wasted page_ins, resident == 12 MiB budget.** The per-region eviction-during-
  use invariant holds across N regions concurrently (disjoint VAs + per-region locks + conservative context sync).
- **NEGATIVE CONTROL — livelock (`-DCIPHER_RESIDENCY_BREAK_LIVELOCK`, release mgr lock before taking ref):
  DETECTED** — wasted page_ins = 219 (the demanded model gets evicted before its serve) → the test catches livelock.
- **NEGATIVE CONTROL — deadlock (`-DCIPHER_RESIDENCY_BREAK_LOCKORDER`, reversed lock order): DETECTED** — AB-BA
  deadlock → the watchdog fired (`_exit 3`). Proves both the lock-ordering claim AND that the test can detect a
  violation. (Single-region coherence's `-DCIPHER_PAGER_NO_GPU_SYNC` control still stands from STEP 1.)

## CORRECTNESS — real-model gate (`pager_residency_serve.py`; 3 DISTINCT 7-8B models)
Mistral-7B + Qwen2-7B + Llama-3.1-8B in ONE process, each behind its own pager region (the proven MemPool routing):
- **G-cores:** each **100% residency** (291/291, 339/339, 291/291 params in-region), **disjoint VAs**
  (`0x302000000` / `0x6bc000000` / `0xaa8000000`).
- **G-pressure:** total resident 47.1 GiB, budget 33 GiB → `mgr_init` evicts LRU → **2/3 resident** (more registered
  than fit).
- **G-correct:** every served forward **bit-identical to its solo reference** (determinism baseline `O1a==O1b`
  first) under BOTH sequential churn (9 served, 0 mismatch) and **concurrent churn (8 served, 0 mismatch, wasted=0)**
  — eviction/pagein cycles on real 14-15 GiB weight sets, no half-evicted serve, no cross-model corruption.
- **G-hbm:** resident 31.4 GiB ≤ budget 33 GiB after churn. **Physical-HBM truth (advisor — not the manager's
  counter):** `mem_get_info` real used = 34.0 GiB ≈ counter(32.0) + ctx(0.6) → no VMM handle leak (a leaked 15 GiB
  model would blow past the 4 GiB tolerance). The C test confirms it at scale: **freeHBM drift = 4 MiB across 13,180
  evict/pagein cycles** → `cuMemRelease` actually frees, no compounding leak.
- **G-overhead (the (ii) sizing, MEASURED not asserted):** per-process CUDA context + torch libs = **633 MiB**; N
  separate processes replicate it → co-residence saves **~(N-1)×633 ≈ 1.27 GiB** HBM at N=3. Bounded; weights and
  CPU-backup are NOT saved.

## OFF byte-identical + regression
Marlin graph-gate vs the new `.so` (`d6dfd5a2`) passes identically (manager dead-until-called). `test_pager`
GATE1/GATE2 still PASS (the `last_used` struct field didn't regress the primitive).

## Anchors / discipline / next op
Deployed `1f305ce6` UNCHANGED; staging `7b5f4311`→`d6dfd5a2`. Source: `cipher_rt_pager.c`/`.h` + `test_residency.c`
(D.8 strays excluded). Default-OFF, additive. Substrate-line: cuMemMap/driver + MemPool routing, no framework
source patch. **NEXT OP:** (i) sub-model / **partial-layer residency** (the real lever — fewer bytes moved, which
vLLM's all-or-nothing tags can't do; this co-residence substrate is its foundation), then PCIe predictive prefetch;
AND — before scaling the density claim — **measure the delta vs N orchestrated vLLM-sleep instances on the RIGHT
metric (memory-overhead + during-use tail latency, NOT tok/s, since PCIe ~51 GB/s is the shared cold-miss
bottleneck).** [[cipher-vllm-weight-alloc-probe]], [[cipher-pager-routing-build]], [[cipher-density-axis-cargo]].
