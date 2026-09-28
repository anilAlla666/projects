# Marlin cache GC-on-free — BUILD REPORT (backfill substrate step)

**Date:** 2026-05-30. **Design:** `MARLIN_GC_ON_FREE_DESIGN_MEMO.md` (approved, full scope).
**Type:** backfill fix on shipped W.2/D.7 Marlin cache (Mem #13 — cache-modification, full
regression). **Build under test:** `cipher_rt_phase4` working tree on HEAD `2909eb3`
(`d9-fp8-staging-close`); `libcipher_rt.so` md5 **`944f5706…`** (staging — NOT deployed, NOT tagged
yet; tag is the separate close step after gates pass). **Anchors UNCHANGED** (Mem #16): deployed
`/usr/lib/cipher/libcipher_rt.so` stays `1f305ce6`; `d7-rh1-close` stays `ed130e7`.

**STOP point:** this reports the build + churn gate + regression + soak. Per instruction it is NOT an
auto-close — the tag/rotation is the close step Anil authorizes after reviewing these gates.

---

## 0. HEADLINE

The Marlin weight-cache GC-on-free fix is **built and the correctness gate is green**: under
load/free/reload churn the previously-fatal stale-kit reuse (Qwen2 all-zeros / illegal-memory-access,
`D7_CLOSE_REPORT.md:35-44`) is closed. Implemented in full scope — **both cache layers**, **dim
re-validation + content fingerprint + event-driven evict + retire/reclaim safe-reclaim** — entirely at
the CIPHER engine layer (Mem #24). An adversarial pre-build review + an empirical concurrent UAF probe
each surfaced a real defect that was fixed before the gate (see §3).

---

## 1. THE FIX AS BUILT (cited to `cipher_rt_marlin_engine.cpp`)

All eviction paths **RETIRE only** (move buffers to `g_retired`); buffers are `cudaFree`'d **solely** in
`reclaim_retired()` after a two-part quiescent barrier. Lock order `g_bf16_weight_mu → g_weight_mu →
g_retire_mu`; `g_fp_mu` is an independent leaf.

1. **Layer-1 (`g_bf16_to_fp16_weight`, the dominant bf16 production surface)** — value type changed
   `void* → Bf16Entry {surrogate, fp, K, N}`. `bf16_surrogate(bf16_w, K, N)` now validates: stored
   `(K,N)` vs live `(K,N)` (dim-reval) + a content fingerprint of the **app** `bf16_w` (same-shape
   reuse). On staleness → `evict_weight` (both layers) + return null → the actuator re-quantizes the
   current model.
2. **Layer-2 (`g_weights`)** — `WeightSlot` gains `fp`. `is_ready(w_ptr, K, N)` dim-revals + fingerprints
   the fp16-direct path; `ensure_weight_quantized_repacked` dim-revals at both ready-checks; mirrors
   CIPHER's own FP8 precedent `cipher_rt_fp8_engine.cpp:251`.
3. **Event-driven evict** — `cipher_rt_marlin_engine_evict_weight(app_w_ptr)` removes the Layer-1 entry
   and all Layer-2 kit slots (by app ptr and by surrogate), retiring every buffer + erasing the
   `g_wptr_model` binding. Completes the `bind_model` lifecycle (bind-on-load → evict-on-free).
4. **Content fingerprint** — `weight_fingerprint()` samples a 1 KB head off a **dedicated non-blocking
   stream + pinned staging** (not the legacy default stream), retried once, folded with `(K,N,total)`;
   env `CIPHER_MARLIN_GC_FP` (default ON).
5. **Safe-reclaim (the user's "refcount==0 + stream-sync")** — `lookup()` takes a **dispatch borrow**
   (`g_dispatch_borrow++`, under `g_weight_mu`) when it hands a kit out; the **actuator releases it after
   `dispatch()`/`dispatch_bf16()` returns, on BOTH success and error branches**
   (`cipher_rt_marlin_engine_release_dispatch_borrow()`). `reclaim_retired()` waits for **borrow==0** (all
   pre-eviction holders drained — a retired kit is unreachable by any new lookup) **then**
   `cudaDeviceSynchronize` (launched kernels done) **then** frees. No in-flight launch can read a freed
   buffer. **(Release is in the always-run caller, NOT a RAII guard inside `dispatch()` — see §3.1: a
   dispatch-only release leaks the borrow on `dispatch_bf16`'s early-error returns.)**
6. **Observability** — `cipher_rt_marlin_engine_gc_fp_failures()` counts fingerprint-sample failures
   (0 in a healthy run); a one-time log on first failure (no longer silent).

**Default-OFF / no-churn invariants (preserved):** `CIPHER_MARLIN` unset → actuator returns before any
GC code (byte-identical OFF). Engaged-no-churn → every pointer maps to one model for the process
lifetime → dims match, fingerprint matches, no evict ever fires, retired list empty → byte-identical
kits/output vs the pre-GC build.

**Files changed (this commit):** `cipher_rt_marlin_engine.cpp`, `cipher_rt_marlin_actuator.c` (extern
decls + 3 call sites threading live `(K,N)` to `is_ready`/`bf16_surrogate` — a validation-threading
change, NOT a gate change; decline logic byte-unchanged), `cipher_rt_marlin.h` (3 new decls). **No
W7-12 substrate TU touched. No kmod change** (libcipher-only, as D.7). A pre-existing **uncommitted,
default-OFF D.8 FAIRNESS/SHIELD wire** (`cipher_rt_cublas_shim.c` + `cipher_rt_fairness.{c,h}`, HARD-
STOPPED, anchor-not-rotated) is present in the working tree, inert in all GC gates (no
`CIPHER_FAIRNESS`/`CIPHER_SHIELD` set), and is **NOT part of this commit**.

---

## 2. ADVERSARIAL PRE-BUILD REVIEW (4 lenses, findings fixed before gating)

A 4-lens adversarial review (concurrency / GC-correctness / fingerprint-perf / ABI-regression) +
per-finding verify pass confirmed the core design sound (clean lock order, no double-free, churn re-quant
fires same-call not vanilla, byte-identical OFF, ABI consistent) and surfaced:

- **(MAJOR) fingerprint fails-OPEN** — a `fp==0` sample failure silently/permanently disabled the content
  check. **Fixed:** retry the DtoH 1×, count + one-time log, no silent mask (falls back to dim-reval).
- **(perf) per-call synchronous DtoH** on the legacy default stream would barrier every tenant's GEMM.
  **Fixed:** moved to a dedicated non-blocking stream + pinned staging (off the default stream).
- **(NIT) head-only 256B collision** → single 1 KB head sample (one sync, better resistance).

## 3. EMPIRICAL CONCURRENT UAF — FOUND AND CLOSED

The dedicated concurrent probe (`marlin_gc_concurrent.py`: one preloaded model, thread-A generates =
live launches, thread-B hammers `evict_weight`+`reclaim`) **crashed with an illegal-memory-access** on
the first build — the exact lookup→launch use-after-free window. This is the case the design names
("free-on-evict creates a UAF on kits shared across concurrent same-key launchers; reclaim drained ONLY
at a quiescent barrier (refcount==0 + stream-sync)"). The first build had the stream-sync but not the
**refcount==0** half. **Fixed** with the dispatch-borrow barrier (§1.5). Re-probe: **PASS — 0 errors, 0
garbage, 353,452 evict+reclaim iterations racing live launches, fp_failures=0.**

### 3.1 SECOND review (post-fix) — borrow-leak on the bf16 error path, FOUND AND CLOSED

A follow-up adversarial review of the borrow barrier caught a real defect the gates structurally could
not (no `cudaMalloc` failed during them): the borrow was released by a **RAII guard inside `dispatch()`**,
but `dispatch_bf16()` has **early-error returns BEFORE it reaches `dispatch()`** (cast-kernel-missing,
`rt_malloc`-fail under pressure). Any such error → the `lookup()` borrow is never released → permanent
leak → `g_dispatch_borrow` stuck ≥1 → `reclaim`'s borrow==0 wait never succeeds → degrades to
devsync-only (the pre-borrow build that crashed). The trigger regime — malloc-fail under memory pressure
during churn — is exactly this fix's target. **Fixed:** the release moved to the **actuator** (the
caller that always runs after a lookup-success), on both success and error branches; the RAII was dropped
from `dispatch()`. **Proven** with a `dispatch_borrow()` self-check (must read 0 post-run) plus a
`CIPHER_MARLIN_GC_FAULT_BF16` injection that forces ~1/3 of `dispatch_bf16` calls onto the early-error
return: §4 reports `dispatch_borrow==0` after both a normal run and the fault-injected run. (binary
`b16cfe59`→`944f5706`).

---

## 4. CLOSE GATES — ALL PASS

**Final binary `944f5706`** (= `b16cfe59` + the §3.1 leak-proof borrow release). The borrow-release
relocation is **output- and memory-neutral on every success path** (it changes only the bf16 error-path
release + reclaim free-timing), so the output/mem gates below (4.1 synthetic, 4.3 full 12-cycle churn,
4.4b additivity, 4.6 30-min soak) were established on `b16cfe59` and **transfer**; the HARD STOP and all
changed/new paths were **re-confirmed on `944f5706`** (4.4a byte-identical-OFF 4/4, 4.2 concurrent, 4.7
borrow-balance normal+fault, and a 6-cycle bound churn KL=0 / 0 MiB leak / post-churn `dispatch_borrow=0`).

| # | Gate | Result | Evidence |
|---|---|---|---|
| 4.1 | Synthetic deterministic mechanism (fp ON) | **PASS** | `gate_synthetic.json` — T1 dim-evict, T2 same-shape fingerprint-evict, T3 evict+reclaim (freed 9 buffers = 3 evictions × {surrogate,B,S}), T4 no-churn 50/50 stable; **fp-OFF witness** = same-shape reuse uncaught (proves the fingerprint is load-bearing) |
| 4.2 | Concurrent safe-reclaim (UAF probe) | **PASS** | `gate_concurrent.json` — 0 errors, 0 garbage, **353,452 evict+reclaim iters racing live launches**, fp_failures=0. (First build crashed here → borrow barrier added → closed.) |
| 4.3 | Real-model churn — BOUND (primary) | **PASS** | `gate_churn.json` — **12/12 cycles KL=0**, **0.0 MiB leak**, no OOM. 12 consecutive 7-8B loads on 80 GB ⇒ evict+reclaim bounds the fp16 surrogate. Qwen2 (orig symptom) + same-shape Mistral↔Llama-3.1-8B + diff-shape. |
| 4.3 | Real-model churn — UNBOUND (fingerprint) | **PASS** | 8/8 KL=0 (same-shape tinyllama↔llama32_1b q/o + diff-shape ffn); 11 GB surrogate leak RECORDED (documented v-next; no evict call) |
| 4.4a | **Byte-identical OFF (Mem #11 HARD STOP)** | **PASS 4/4** | `marlin_gc_off_byte_identical.json` — max_abs_diff 0.0, KL 0.0 (tinyllama/llama32_1b/mistral7b/llama31_8b). The every-prior-model KL=0 gate. |
| 4.4b | Additivity — steady-state Marlin | **PASS (byte-identical)** | pre-GC-HEAD-Marlin-ON vs my-GC-Marlin-ON, tinyllama no-churn = **max_abs_diff 0** ⇒ GC inert in steady-state. (The 0/4 vs `227d7973` was a wrong-baseline artifact — that `.so` is not clean-HEAD.) |
| 4.4b | Additivity — **A/B churn witness** | **fix confirmed** | clean-HEAD (pre-GC) build **crashed illegal-memory-access at model 2** of the offreg load/free/reload loop (1/4); my GC build completed **4/4** — direct before/after proof the fix closes the D.7 crash. |
| 4.4c | W7-12 microbenches `test_step3_{b0,b1}` | **PASS** | `gate_regression.log` — B0 P99 cadence + N=128 smoke (1.28M writes coherent); B1 cold-start + compose (0 drop, 32000 drained). Additive: no W7-12 TU touched. |
| 4.5 | Fingerprint hot-path cost (ON vs OFF) | **+5.7%** | `gate_fpcost_{on,off}.json` — single-tenant Mistral decode 2633 vs 2491 ms median. Far under the review's 50-100% estimate (off-default-stream pinned async sample). `CIPHER_MARLIN_GC_FP=0` disables it. |
| 4.6 | 30-min churn soak | **PASS** | `gate_soak.json` — **55 cycles / 1807 s, 0 corruption, 0 all-zero, 0 fp-failures, no crash, free-memory FLAT (4 MiB spread over 30 checkpoints)**. (Raw `PASS=False` was an `abs(net_leak)` formula artifact — net_leak −4644 MiB = MORE free at end than the mid-cycle-0 baseline sample, i.e. no leak; corrected criterion = checkpoint spread.) |
| 4.7 | **Dispatch-borrow balance** (normal + fault-injected) | **PASS — borrow=0** | `reval_borrow_{normal,fault}.json` — `dispatch_borrow()==0` after a normal run AND after `CIPHER_MARLIN_GC_FAULT_BF16=1` forced ~1/3 of `dispatch_bf16` onto the early-error return (proves the §3.1 error-path release; no leak). 6-cycle bound churn: post-run `dispatch_borrow=0`. |

**Verdict: every close gate PASS** (the HARD STOP byte-identical-OFF, the churn KL=0/flat-mem, the concurrent UAF closure, the soak, the steady-state additivity, and a live A/B proving the crash is fixed). The fingerprint cost (+5.7%) and the UNBOUND surrogate leak are the two honest, documented residues — both addressed by, respectively, the env knob and the V.0 evict-on-free wiring.

---

## 5. HONEST FRAMING / ENG-DEBT (Marvel standard)

- **`evict_weight` is the production-primary mechanism for MEMORY, the fingerprint for CORRECTNESS — and
  evict is not yet wired to a production residence caller.** The bf16 path keeps a full-size fp16
  surrogate per model (≈ the model's fp16 footprint); under churn WITHOUT an evict call those surrogates
  leak (UNBOUND soak measured ~11 GB over 8 small-model cycles, OOM with 7-8B models) — the documented
  v-next "pure-eviction leak". The BOUND path (evict + reclaim) holds 12 consecutive 7-8B churn cycles at
  **0 MiB leak**. **The fingerprint keeps output correct without an evict call (UNBOUND 8/8 KL=0) but
  cannot bound memory.** ⇒ **Wiring `evict_weight` into the vLLM-serve residence layer on model-free is
  the V.0/CP-5.5 integration step**; the churn harness's BOUND mode calls it exactly as that layer will.
- **Fingerprint cost** — default ON (Anil's explicit "include it"); §4.5 reports the measured single-
  tenant overhead. The env `CIPHER_MARLIN_GC_FP=0` disables it (dim-reval + evict remain) for deployments
  that rely on the event-driven evict exclusively.
- **Residual safe-reclaim bound** — the borrow barrier closes the lookup→launch window for the actuator
  paths (lookup always paired with one dispatch). The `reclaim` borrow-wait is bounded (hang-prevention);
  the device-sync is the always-applied second guard.
- **Qwen2** (the original-symptom model) is included in the churn rotation; the same-shape stress uses the
  local Mistral↔Llama-3.1-8B pair (identical 4096/14336 projections), a stronger same-shape test than a
  single Qwen2 reload.

## 6. DISCIPLINE

Mem #11 (KL=0 HARD STOP) — §4.4 byte-identical OFF + churn KL=0. Mem #13 (additive/full-regression) —
no W7-12 TU touched; full regression run. Mem #16 (anchors unchanged) — staging only, no rotation, no
tag in this build. Mem #24 (substrate-line) — all engine-internal; evict called by CIPHER's residence
layer, not a framework hook. Commit (when authorized) as Anil, no co-author.
