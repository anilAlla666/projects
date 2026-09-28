# Marlin cache GC-on-free — SCOPE + DESIGN MEMO (backfill substrate step)

**Date:** 2026-05-30. **Type:** READ-ONLY diagnosis + design memo. **No build, no commit, anchors
unchanged.** STOP for Anil's approval of (a) the design and (b) the additive-vs-full-regression call
before any code. Every claim cited `file:line`, not memory.

**Context:** first buildable single-GPU step to CP-5.5. The Marlin weight cache does not GC kits on
model free → under load/free/reload churn a freed model's stale cache entry is reused → **wrong output
(Qwen2 all-zeros) + co-resident illegal-memory-access crash** (`D7_CLOSE_REPORT.md:35-44`). Goal-1 (100
bursty agents) **is** this churn pattern. This is a fix on earlier-shipped W.2/D.7 substrate, so per
Mem #13 it is a **backfill substrate step** (design→approve→build→full regression), NOT a quick fix.

---

## PART 1 — DIAGNOSIS (cited)

### 1. The bug — two stale-map layers, not one

The failure is **stale-but-valid wrong-data reuse + dim-mismatch OOB** (not literal use-after-free — the
buffers are never freed today, they leak). The irony to keep in view: **the fix is what introduces the
literal-dangling risk** (see §PART 2 safe-reclaim).

**Production serves bf16** (`D7_CLOSE_REPORT.md:3` "bf16 … throughout"; the Qwen2-all-zeros failure was
bf16), so the dominant path is the **bf16 route through TWO cache layers**:

- **Layer 1 — `g_bf16_to_fp16_weight`** (`cipher_rt_marlin_engine.cpp:1280`): an
  `unordered_map<const void*, void*>` mapping the **app's bf16 weight pointer → CIPHER's fp16 surrogate
  buffer**. Stored at `:1271` inside `quantize_repack_bf16`; read by `bf16_surrogate()` (`:1282-1286`).
  **This map is keyed by the raw app `bf16_w` and has NO erase path anywhere.**
- **Layer 2 — `g_weights`** (`:571`): `unordered_map<MarlinWeightKey, WeightSlot>` keyed by
  `(model_id, surrogate_ptr)` holding the quantized kit (`marlin_B`, `marlin_S`, cached `K/N/G`,
  `:525-533`). **Also no erase path** (the only `.erase` in the file is `g_wptr_model.erase` in
  `bind_model`, `:606`).

**Lifecycle trace (bf16, the failing path):**
1. Model A's bf16 weight lives at address **X**. First eligible Marlin call (`actuator.c:198-216`):
   `bf16_surrogate(X)` misses → `quantize_repack_bf16` casts bf16→fp16 into a CIPHER-allocated surrogate
   `d_fp16_A` (`engine:1245`), stores `g_bf16_to_fp16_weight[X] = d_fp16_A` (`:1271`), then quantizes →
   `g_weights[(0, d_fp16_A)]` = A's kit (`:1043-1046`).
2. App frees model A (`d7_rh1_kl.py:31` `del m; gc.collect(); torch.cuda.empty_cache()`). **CIPHER is not
   told.** `d_fp16_A`, A's kit, `g_bf16_to_fp16_weight[X]`, and `g_weights[(0,d_fp16_A)]` all **linger
   (leak), still valid memory.**
3. Model B's bf16 weight loads at the **reused address X** (allocator reuse). Actuator calls
   `bf16_surrogate(X)` → **HITS the stale entry → returns `d_fp16_A`** (A's casted weights, never re-cast
   for B). Because `fp16_surrogate != nullptr`, `actuator.c:199 if(!fp16_surrogate)` is **false** → the
   re-cast/re-quant is **skipped** → `lookup(d_fp16_A)` returns A's kit → `dispatch_bf16` runs **B's GEMM
   on A's quantized weights**.
   - **Same shape (A.K/N == B.K/N):** silent wrong-data → **Qwen2 all-zeros**.
   - **Different shape:** the kernel launches with B's live `(M,N,K)` (`actuator.c:225-228`,
     `bf16_M/N/K` from the live call) against a kit packed for A's dims → out-of-bounds device read →
     **illegal-memory-access crash**.

**Why dim-reval on `g_weights` alone does NOT fix the production path:** once `bf16_surrogate(X)` hands
back the stale `d_fp16_A`, the `g_weights[(0, d_fp16_A)]` pair is **internally consistent** (A's surrogate
↔ A's kit, dims match A) — a `g_weights`-level dim check passes and A's weights are still used for B. **The
staleness must be caught at Layer 1 (`g_bf16_to_fp16_weight`), where the app pointer X is the key.**

**fp16 path (secondary, same root):** `ensure_weight_quantized_repacked` (`:1008-1049`) honors `s->ready`
at `:1013` **without comparing the cached `s->K/s->N` to the live `(K,N)`**; `lookup` (`:1087-1102`)
returns cached `marlin_B/marlin_S` + cached dims with no identity recheck. Same stale-reuse class, one
layer (no surrogate hop), so dim-reval here is sufficient for the fp16 case.

### 2. The keying — eviction is missing on BOTH maps; the entry lingers, the buffer leaks

The D.7 re-key (`:537-559`) keyed `g_weights` on `(model_id, w_ptr)` to stop a **different-family** model
at a reused `w_ptr` from hitting another family's kit — that fixed *cross-family* contamination at clean
co-residence (`D7_CLOSE_REPORT.md:21-23`). It does **not** address the free lifecycle: on free, **neither
map evicts the entry, and neither frees the device buffer** (the surrogate `d_fp16` and the kit
`marlin_B/marlin_S`). So this is **both faces at once**: *entry not evicted → dangling-reuse corruption*
**and** *buffer not freed → leak*. The re-key only narrows the same-key collision to: **same family
(same `model_id`) at a reused `w_ptr`**, or **any unbound churn** (`model_id=0`, see §3).

### 3. The churn pattern (cited)

Goal-1 = 100 bursty agents = models **load → run → evict → reload** as agents arrive/idle
(`cipher-marlin-cache-churn-gap` memory; `D7_CLOSE_REPORT.md:41`). The D.7 churn harness reproduced
**Qwen2 all-zeros + co-resident illegal-memory-access** (`D7_CLOSE_REPORT.md:39-40`,
"observed in the first, churn-based harness"). The clean static-residence test (load-all-once, no churn)
is correct, KL=0 per family (`:8-21`) — confirming the bug is **purely the free/reload lifecycle**, not
the re-key.

**Bound-vs-unbound, resolved from the harness:** the residence path **binds on load** —
`d7_rh1_kl.py:21` `bind_model(p.data_ptr(), i)` per model — but tears down via **raw torch
`del m; gc.collect(); torch.cuda.empty_cache()`** (`:31`) with **NO unbind/evict call**. So even the
*bound* production path leaks and goes stale on free: **`bind_model` registers an identity on load but the
lifecycle has no teardown side.** Two sub-cases for the same-key collision:
- **bound churn** (residence owner): same `model_id` re-binds at the reused `w_ptr` → same key → stale
  kit reused (the production case).
- **unbound churn** (raw torch, `model_id=0` throughout): key `(0, w_ptr)` collides across models → stale
  reuse (the repro-harness case).

### 4. The odd-dim→vanilla guard — SEPARATE mechanism, but linked to the crash

The "odd-dim→vanilla" guard is the **shape-decline gate** in `actuator.c`: `marlin_N<1024 || marlin_K<1024`
(`:258`), `(marlin_K & 127)!=0 || (marlin_N & 63)!=0` (`:262`) → `PASSTHROUGH` to vanilla cuBLAS (and the
bf16 mirror at `:191-193`). It routes **geometry-incompatible weights** to vanilla; it is **not** part of
the free lifecycle and is a **separate mechanism** from the GC bug.

**But it is linked to the crash:** the shape gate runs **only on the live call before `is_ready`**. A
**stale `ready` slot bypasses it** — the actuator sees `is_ready==true` and dispatches on the cached kit's
dims (`actuator.c:271`, `engine:1013`), never re-checking that the *current* weight's geometry still
matches. So a model whose real shape would have been declined, or differs from the cached `K/N`, reaches
the kernel via the stale slot → OOB. **The fix should therefore also re-assert the shape/dim invariant on
the cached path** (dim-reval), not only at first-call. The guard itself needs no change; the GC fix
restores the invariant it assumes.

---

## PART 2 — GC-ON-FREE DESIGN (design only; build is a separate approved step)

**Goal:** the Marlin cache correctly handles a weight being freed and later reloaded — entry eviction +
device-buffer lifecycle on **both** map layers so no stale surrogate/kit is ever reused and no buffer
leaks; **byte-identical when not churning**; safe under concurrent same-kit launchers.

### The two real design decisions (per the diagnosis)

**(a) Event-driven eviction (PRIMARY) + validate-on-use (defense-in-depth).** Gated on whether production
tears down through a CIPHER-controlled signal — §3 shows it does **not** today (binds on load, raw-torch
free, no teardown call). So:

1. **PRIMARY — complete the `bind_model` lifecycle with a teardown-evict API** (event-driven; closes BOTH
   the crash and same-shape corruption with **no fingerprint**). Add:
   `cipher_rt_marlin_engine_evict_weight(const void *app_w_ptr)` (and/or
   `..._evict_model(model_id)`), which under `g_bf16_weight_mu` + `g_weight_mu`:
   - resolves `app_w_ptr` → surrogate via `g_bf16_to_fp16_weight` (if bf16), **erases** the Layer-1 entry,
   - **erases** the `g_weights` slot(s) for that surrogate (and the app_w_ptr for the fp16 path),
   - **safely reclaims** the surrogate + kit buffers (see (b)),
   - **erases** any `g_wptr_model` binding.
   The residence layer calls it on model free — the exact site already exists (`d7_rh1_kl.py:31`, right at
   `del m; empty_cache()`). This is the natural completion of bind(load)→evict(free) and is the
   production-realistic primary. Idempotent; a no-op for unknown pointers.

2. **DEFENSE-IN-DEPTH — validate-on-use at BOTH map layers** for the unbound / no-evict-call (raw-torch)
   path:
   - **Layer 2 (`g_weights`) — dim-reval, mirroring CIPHER's own FP8 precedent**
     `cipher_rt_fp8_engine.cpp:251`
     (`if(s.ready && (s.K!=K || s.N!=N)){ free; s=WeightSlot{}; }`): before honoring `s->ready` at
     `engine:1013`/`:1037` and in `lookup` (`:1094`), compare cached `(K,N)` vs the live `(K,N)`; on
     mismatch, reclaim the kit and re-quantize. **Closes the crash + reshape-leak.** Residual: same-`(K,N)`
     same-key silent corruption survives this check alone.
   - **Layer 1 (`g_bf16_to_fp16_weight`) — the same dim-reval at the surrogate hop:** the surrogate must
     carry the bf16 weight's `(K,N)` so a reused `bf16_w` with a different shape re-casts instead of
     returning the stale surrogate. (Layer-1 keyed by app pointer is where production staleness lives.)

**(b) Safe-reclaim mechanism (load-bearing safety, NOT eng-debt).** Today the bug is silent corruption
(buffers never freed). **The instant we free-on-evict, a kit shared across concurrent
same-`(model_id, w_ptr)` launchers — exactly the Track-2 weight-sharing regime, where a kit IS shared
across threads — can be freed mid-launch → a real crash the current code does not have.** Reconcile this
with the "GPU mem flat across cycles" PASS criterion as follows:
   - Evicted surrogate/kit buffers go on a **retired list**, not freed inline.
   - The retired list is drained (real `cudaFree`) **only at a quiescent barrier** — refcount==0 for the
     kit **and** a `cudaStreamSynchronize`/device-sync point — never asynchronously while a launch may be
     in flight. The churn test (below) includes that barrier between cycles, so memory returns **flat each
     cycle** without an async free racing a launch.
   - v1 reclaim = **deferred-free-at-barrier** (simple, correct); a precise per-kit **refcount** is the
     v-next hardening if continuous (barrier-free) reclaim is later required.

**Fingerprint — scoped, NOT default.** A content fingerprint (strided device hash of the weight, rechecked
on the ready path) is the only thing that closes **unbound same-`(K,N)` raw-torch churn** without an evict
call. With the event-driven evict API as primary, this is **speculative scope**: include it **only if Anil
requires unbound same-shape raw-torch coverage**. Flagged in eng-debt, not built by default.

### Additive-first vs cache-modification — the determination

**FULL REGRESSION REQUIRED.** Per the backfill protocol: the bug is **in the cache lifecycle**
(`ensure_weight_quantized_repacked`, `lookup`, `bf16_surrogate`, the two maps), so the fix **must modify
W.2/D.7-shipped cache code** — it cannot be purely additive. The new evict API is additive (new exported
symbol), but the validate-on-use dim-reval and the retired-list reclaim **touch shipped hot-path code**.
Therefore this is treated as a **FRESH substrate step with the full regression protocol**, not a quick
fix:
- **Files that change:** `cipher_rt_marlin_engine.cpp` (both maps: evict API + dim-reval + retired-list
  reclaim) and possibly `cipher_rt_marlin.h` (the new API decl). **No actuator-gate logic change**
  expected (the odd-dim guard is untouched). **No kmod change** (libcipher-only, as D.7 was —
  `D7_CLOSE_REPORT.md:51`).
- **The fix is "additive in behavior"** — see the no-op guarantee below — **but modifies shipped code**, so
  it gets the full regression, stated explicitly here so it is not waved through as additive.

**Stage-0 passthrough / "changes nothing when not churning" preservation:**
- **CIPHER OFF (no injection):** no actuator → byte-identical (the FP8 §1 gate model).
- **Engaged, no churn** (the D.7 clean static-residence case): every `w_ptr` maps to one model for the
  process lifetime → dims always match, no evict ever fires, retired list stays empty → the validate path
  is a **no-op pass-through** → **byte-identical kits, KL=0 vs the current build.** This is the additivity
  guarantee that §2 regression must prove.
- **Engaged, churn:** the new path fires (evict + re-cast/re-quant) → **correct** output instead of
  corruption/crash.

### Marvel standard — full free/reload signal space

The design + close must cover, not just the happy single-cycle:
- **single free → reload** (core case): evict-on-free (or dim-reval fallback) → re-quant → correct.
- **reload-after-free at reused address, same shape** (the Qwen2-all-zeros case): event-driven evict
  closes it; validate-on-use needs the fingerprint (scoped) for the unbound variant.
- **reload-after-free, different shape** (the crash case): dim-reval → re-quant → no OOB.
- **concurrent free** (free on thread X while inference on thread Y holds the same shared kit — the
  Track-2 regime): the retired-list + quiescent-barrier reclaim is the safety design for exactly this; the
  test must exercise concurrent launchers on a shared kit during eviction.
- **free-during-inference**: eviction must never free a buffer with launches in flight (barrier/refcount).
- **churn under N tenants**: the 100-agent steady state; the test runs cyclic churn under concurrency.
- **in-place same-shape weight update** (fine-tune at same `w_ptr`): fingerprint (if enabled) re-quants;
  without it, the event-driven evict API must be called — documented.

### THE CHURN TEST (the G1-gating close gate — design)

A load/free/reload churn harness reproducing the D.7 failure, run on the candidate `.so`:
- **Models:** ≥2 Marlin-engaged bf16 families. **Must include both** (i) a **different-shape** reload
  (e.g. Mistral-7B ↔ Qwen2-7B vs a differently-shaped family) to exercise dim-reval/crash-path, and
  (ii) a **same-shape** reload at a reused address (e.g. Qwen2 freed → reload, or two same-shape families)
  to exercise the same-`(K,N)` corruption path — **this is the case that proves the event-driven evict /
  fingerprint, not the dim-check.**
- **Cycle:** load A → bind (production-realistic) → run forward (Marlin engages, cache populated) →
  **evict-on-free** (residence calls the new API) **then** `del; empty_cache()` → load B at the reused
  address → run B → compare to a **clean solo-B reference** (separate-process). Repeat ≥10 cycles, and a
  concurrent variant with N launchers sharing a kit during an eviction.
- **Two modes:** **bound churn (primary, production-realistic)** with the evict API called; **unbound
  raw-torch churn (defense-in-depth)** without the evict call, exercising validate-on-use.
- **PASS iff** (all three):
  1. **Correctness:** every reloaded model **KL=0 / max_logit_diff=0** vs its clean solo ref — **no
     all-zeros, no wrong-data** (the §1 D.7 KL methodology, `d7_rh1_kl.py`).
  2. **Safety:** **no illegal-memory-access, no crash, no `dmesg` anomaly** across all cycles incl. the
     concurrent-eviction variant.
  3. **No leak:** CIPHER's own surrogate+kit GPU footprint **flat across cycles** (drained at the
     quiescent barrier) — bounded by the live model set, not cumulative.

This is the negative-case validation the close needs, **alongside** the standard full regression:
every-prior-model **KL=0** (additivity, byte-identical — the no-churn guarantee), **W7-11 microbenches**
as available (note `D7_CLOSE_REPORT.md:53-56` / `D9_FP8_ACTUATOR_CLOSE_REPORT.md:61` — several are
`/tmp`-scratch, not reproducible as named binaries; the runnable `test_step3_*` set + additivity-KL are
the direct evidence), and the **30-min N=128 soak**.

### Eng-debt forecast (honest residue, NOT done here)

- **Fingerprint collision** (if enabled): astronomically unlikely with a strided 256B sample + size, but
  documented; not built unless unbound same-shape coverage is required.
- **Pure-eviction leak** (model freed and **nothing reloads** at that address, and no evict call): the
  validate-on-use path only reclaims **on next reuse**; a `cudaFree` interposer (the general solution) is
  the only thing that reclaims at the moment of free with no reload. That interposer is a **new
  process-wide interception surface on every free** (GOT-patch of late-loaded `cudaFree`, like the cuBLAS
  injection) — **disproportionate** when the event-driven evict API + on-reuse reclaim closes the observed
  failure. **Flagged as v-next hardening**, explicitly not the v1 fix.
- **Continuous (barrier-free) reclaim:** v1 uses deferred-free-at-barrier; precise per-kit **refcount** is
  v-next if a barrier between churn bursts is ever unacceptable.

### Confidence calibration (Marvel std)

- **HIGH** that event-driven evict (completing bind/unbind) + Layer-1/Layer-2 dim-reval closes the crash
  and the bound same-shape corruption — direct precedent (`fp8_engine.cpp:251`) and a clear, localized
  surface.
- **MEDIUM-HIGH** that the close is achievable single-GPU on this pod (D.7's KL/soak harnesses exist and
  the churn harness is a small extension).
- **The one place additive-only is NOT fully achievable** is the **concurrent-free safe-reclaim** — it
  needs a small new mechanism (retired-list / refcount), which is still **engine-internal** (no new
  interception surface, no framework coupling). This is precisely why **cache modification is unavoidable →
  full regression**.

### SUBSTRATE-LINE CHECK (Mem #24) — CONFIRMED

Every change lives in `cipher_rt_marlin_engine.cpp` (the cache layer, at the `cublasGemmEx`-intercept
depth) + the new API decl in `cipher_rt_marlin.h`. The fingerprint (if enabled) reads the weight device
buffer CIPHER **already holds the pointer to** (`call->A`). The event-driven evict call is made by
**CIPHER's own residence/owner layer** (the same layer that already calls `bind_model`) — **not** a
vLLM/torch scheduler hook. **No framework coupling, no monkeypatch.**

---

## STOP — Anil's call

Approve before any code:
1. **The design** — PRIMARY event-driven evict API (completing the `bind_model` lifecycle) + validate-on-use
   dim-reval at both map layers + retired-list/quiescent-barrier safe-reclaim.
2. **The additive-vs-full-regression determination** — **FULL REGRESSION** (cache hot-path code is
   modified; the bug is in the lifecycle, so it cannot be purely additive).
3. **The fingerprint scope** — **excluded by default**; include only if unbound same-shape raw-torch churn
   coverage is required.

**No build, no commit, anchors unchanged.** Build is a separate approved step. Commit (when approved) as
Anil, no co-author trailer.
