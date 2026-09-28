# THREE READ-ONLY ASSESSMENTS — does the pager arc beat N-sleep / is 100 reachable / distance to ORCHESTRATE

**Date:** 2026-06-01. **Type:** READ-ONLY (measurement + cited source; NO build, NO commit; deployed `1f305ce6`,
staging `d6dfd5a2`, all tags stand). Measured numbers marked **[M]**; reasoned **[R]**.

---

## ASSESSMENT 1 — THE DELTA: real, but CONDITIONAL and COUPLED to #2 (NOT a clean reversal; ~3× not 19×)

**Baseline measured [M]** (`vllm_sleep_probe.py`, Mistral-7B, enable_sleep_mode): `sleep(level=1)` = **5640 ms**
(re-offloads ~14 GB weights D2H at ~2.5 GB/s pageable), `wake_up()` = **416 ms**, correct across cycle,
resident-while-asleep = 1.56 GiB. **vLLM's awake CPU-RAM footprint = 0** — it allocates the `cpu_backup_tensor`
on sleep and frees it on wake (cumem.py).

**CIPHER measured [M]** (`cipher_evict_timing.py`, Mistral-7B behind a region): `page_out` (evict) = **6 ms**
(unmap-only, cipher_rt_pager.c:126-154 — NO D2H; the warm mirror is captured ONCE at end_load, weights are
read-only), `page_in` (restore) = **308 ms** (pinned H2D ~44 GB/s). **CIPHER's awake CPU-RAM footprint = 14 GB
pinned, PERMANENTLY, per model.**

**The honest decomposition (advisor — do NOT report 19× as the headline):**
- As-shipped swap (evict victim + restore demanded): CIPHER ~314 ms vs N-sleep ~6056 ms = **19×** — but most of
  that is vLLM's *implementation* choice (pageable backup at 2.5 GB/s; pinned would be ~0.56 s).
- **vs an OPTIMIZED N-sleep** (pinned persistent backup): ~970 ms → **~3×**.
- **On the USER-FACING critical path:** activation latency = the demanded model's `page_in` (308 vs 416 ms =
  **~1.3×**). Eviction of the victim can be done **proactively/in the background** by any competent orchestrator
  (keep HBM headroom) → vLLM's 5.6 s sleep never touches user latency → the 19× evaporates there.
- **The DURABLE, fundamental advantage:** CIPHER's evict consumes **~0 PCIe** (pure unmap); vLLM's sleep consumes
  a **full D2H**. This matters precisely because [[cipher-density-axis-cargo]] found **PCIe is the binding
  constraint at scale** — copy-free swap-out frees that bandwidth for the demanded-model `page_in`s.

**#1 and #2 are ONE coupled trade-off (advisor catch):** CIPHER's 6 ms evict is *bought by* the persistent 1:1
pinned host mirror that Assessment 2 shows does not scale past ~15 models. The fast-evict win IS the spend of #2's
scarce resource. Cannot bank #1 independently of #2.

**VERDICT: real-delta-exists — copy-free eviction (read-only-weights → static mirror → evict = unmap), saving PCIe
on swap-out — and it is from the ALREADY-BUILT mechanism, NOT unbuilt (i). BUT it is conditional (coupled to the
#2 mirror story) and worth ~3× vs an optimized baseline, not 19×; on user-latency it is ~1.3×.** This is "CIPHER is
a better swap allocator (no D2H on evict)" — an engineering lead with a real PCIe-bandwidth basis — NOT "a
capability vLLM structurally lacks." The arc is justified **to the next de-risk**, not to "100 proven." **The right
next measurement is the delta at disk-bound 100-density (where BOTH are disk-bound on restore), NOT N=3-in-RAM.**

## ASSESSMENT 2 — PINNED-RAM CEILING: mmap-warm compatible; escapes the PINNED limit, not RAM-capacity

**Compatibility [R, cited]:** the page_in restore is `cudaMemcpy(r->va, r->warm, warm_bytes, H2D)`
(cipher_rt_pager.c:121); the warm buffer is `cudaMallocHost` pinned (line 99; end_load capture line 245-246).
`cudaMemcpy` accepts ANY host pointer → swapping `r->warm` for an **mmap'd file pointer** is a ~2-line change at
lines 99-100 / 245-246. **The eviction-during-use coherence is GPU-side (unmap + cuCtxSynchronize, page_out) and
is UNAFFECTED by the warm source** → mmap-warm preserves the proven coherence AND the copy-free evict.

**Cost [M + R]:** disk/page-cache read (vda virtual disk) = **7.5 GB/s [M]** (`dd`, page-cache-served; the earlier
2.0 GB/s was python-read overhead). Restore tiers: pinned H2D 308 ms [M] → pageable mmap-from-page-cache ~0.6-1 s
[R, CUDA stages pageable through a bounce buffer, ~½ pinned] → cold mmap-from-disk ~**2 s/7B** [R, 14 GB / 7.5 GB/s
+ fault]. So mmap-warm makes restore ~2-6× slower and pushes the binding constraint toward disk for the cold tail.

**THE CEILING [R]:** three tiers of warm source — **pinned** (scarce, fast 44 GB/s), **page-cache RAM** (1:1 with
model bytes, medium ~0.6-1 s, still RAM-capacity-bound), **disk** (unbounded capacity, slow ~2 s/7B, shared
7.5 GB/s). **"100 at full fp16" is UNREACHABLE** — 100×14 GB = 1.4 TB ≫ 221 GB RAM, so the cold tail MUST spill to
disk and restore becomes disk-bound. Reachable warm-in-RAM ceiling ≈ 221 GB / 14 GB ≈ **15 fp16 models** (or ~60
INT4). Beyond that → disk-bound. **The binding resource shifts pinned-RAM → RAM-capacity → disk-bandwidth.** mmap-
warm relaxes the *pinned* ceiling (the build's current limit) but the real path to high N is **INT4 + (i) partial-
layer residency** (fewer bytes per model in the warm/resident tiers), not the host-mirror medium alone.

## ASSESSMENT 3 — DISTANCE TO ORCHESTRATE CLOSE: pager is G-O2 critical-path; G-O3 (the headline) is UNTOUCHED

**BANKED (cite):** G-O7 fire-in-graph = `graph-gate-step1-marlin-capture-safe` (Marlin fires in capture, replay
bit-identical). G-O5 dynamic-membership = the graph-membership DECISION (graph-sufficient via batch-buckets,
[[cipher-graph-membership-decision]]) — a decision, not a build. G-O2 multiplex (residency half) =
`pager-step1/1.5/step2arch` (per-region coherence + routing + N-model co-residence). G-O4 on-chip fusion =
closed-by-subsumption ([[cipher-fusion-not-vllm-cargo]] — vLLM already fuses). G-O8 correctness = the gate
discipline carried through every pager build.

**UNTOUCHED / partial:** **G-O3 statistical-mux MFU (85%) — NEVER measured assembled; this is the headline number
and it is the SCHEDULER-BRAIN, not the pager.** The pager decides *which models are resident*; G-O3 needs the
*bursty-arrival multiplexer* that packs requests across resident models into shared batches/graphs to fill MFU —
a different build. Also untouched/partial: G-O1 launch-elim assembled (graph gate addresses it but unmeasured as a
goal), G-O6 compose (the assembled multi-goal run), G-O9 per-tenant isolation/P99 under multiplexing (green-ctx
work exists, not assembled with pager+scheduler).

**VERDICT [R]:** the pager is **critical-path for G-O2 (density)** but **NOT sufficient for G-O3** — "pager done" ≠
"ORCHESTRATE close." Build-count to close ≈ **4-5**: (i) partial-layer residency [~1, pager], the bursty-arrival
**scheduler-brain** [~2, UNTOUCHED — the heart of G-O3], the assembled G-O2+G-O3+G-O9 MFU/P99 measurement [~1].
The pager arc is ~half; the scheduler that fills MFU is the untouched other half.

---
## NET (for Anil's next-build decision)
1. **Delta is real but conditional** — copy-free eviction saves PCIe on swap-out (fundamental, already built),
   worth ~3× vs an optimized N-sleep, ~1.3× on user latency; coupled to the #2 mirror cost. Not a moat; a real
   engineering lead with a PCIe basis. **Next de-risk: measure the delta at disk-bound 100-density, not N=3-in-RAM.**
2. **100 at full fp16 is physically unreachable** (1.4 TB host mirror); mmap-warm relaxes pinned→RAM/disk; the
   real density lever is **INT4 + (i) partial-layer**, not the mirror medium.
3. **The headline ORCHESTRATE number (G-O3 85% MFU) is the scheduler-brain, untouched** — the pager enables it but
   does not deliver it. If the goal is ORCHESTRATE close, the next high-leverage build may be the scheduler, not
   more pager.
