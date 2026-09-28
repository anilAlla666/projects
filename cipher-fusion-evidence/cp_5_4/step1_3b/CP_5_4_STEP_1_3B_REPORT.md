# CP 5.4 — Step 1.3b' (POOL / executor binding) — CLOSURE REPORT

**Date:** 2026-05-19. **Verdict: Step 1.3b' COMPLETE — Phase A + Phase B
(B1/B2/B3) all PASS.** The Phase B batch executor is now a CP 5.4 POOL tenant
bound to a kmod-aligned green context, with verified disjoint POOL+PARTITION
coexistence. No anchor rotated — Option 2 needed only executor-side changes.

---

## Phase A — design (Option 2 adjudicated)

`CP_5_4_STEP_1_3B_DESIGN_MEMO.md` + Appendix A. The executor stays on
`libcipher_v2.so`; a thin Python CP54 client gives it a POOL allocation and a
`torch.cuda.GreenContext`. Appendix-A probe proved `GreenContext.create(8N)`
picks the low-prefix groups 0..N-1 deterministically; the kmod already places
the POOL on that low prefix → aligned, disjoint. Option 4 (wrap an external
`CUgreenCtx`) was investigated and rejected (torch's `GreenContext` is
count-only by C++ design).

## Phase B — implementation + tests

**Source changes (executor-side only):**
- **`cp54_pool.py`** (new) — `PoolBinding`: `ALLOCATE(POOL)` via `/dev/cipher`
  (ioctl nr 13); `build_green_ctx()` creates `torch.cuda.GreenContext.create(
  grp_count*8)`, binds it (`set_context` + a green-ctx stream), and
  **self-verifies** the physical SM set with a `%smid` `load_inline` kernel —
  raising if it is not exactly the kmod low-prefix groups (the safety net for
  the PyTorch-internal low-prefix-pick dependency); `check_resize()`
  re-`ALLOCATE`s each round and rebuilds the green context if the pool count
  changed; `free()` for clean exit.
- **`cipher_batch_executor_gen.py`** — 5 minimal insertions (no reindent):
  POOL bind after `.cuda()`; warmup / 5 rounds / teacher-forced forward run
  under the green-ctx stream; `check_resize()` at each round top; `free()` at
  exit. Preserved as `.pre_cp5_4_step1_3b` (`39056662`, two locations).

**B1 — build/sanity:** `py_compile` clean on both files; standalone
`PoolBinding` exercise — `ALLOCATE(POOL)`→15 groups, green ctx self-verify PASS
(SMs = kmod groups 0..14), `FREE` clean.

**B2 — executor POOL lifecycle:** one `run_arm3.sh` rep — executor registered
POOL (`grp_mask=0x7fff`, 15 groups / 120 SMs), green ctx self-verify PASS, all
5 rounds ran (TFGATE KL ≤ 5.5e-5), `FREE(POOL)` on clean exit; ledger
`pool=15` during, `free=15` after.

**B3 — integration tests:**

| test | result |
|---|---|
| **A** — POOL alone | **PASS** (= B2: 120-SM green ctx, self-verified, forward correct) |
| **B** — POOL + PARTITION coexistence | **PASS** — see below |
| **C** — POOL reaper | **PASS** — `kill -9` executor → do_exit reaper reclaimed all 15 POOL groups (`free=15`, no leak) |
| **D** — Phase B regression (3 reps) | **PASS** (mean-vs-mean gate) — see below |

**Test B (load-bearing) — disjoint coexistence verified.** Executor bound POOL
at 15 groups / 120 SMs; a PARTITION tenant (libcipher_rt, `sm_count=16`)
allocated `grp_mask=0x6000` → **groups 13,14**. The executor's `check_resize()`
detected it: `POOL resize 15 -> 13 groups`, rebuilt the green context at
**104 SMs**, self-verify PASS — **SMs = kmod groups 0..12**. Ledger during
coexistence: `n_partitions=1 pool_grp_count=13 free_grp_count=0`. Pool groups
{0..12} ∩ partition groups {13,14} = ∅ — **physically disjoint**, the CP 5.4
core invariant, confirmed under Option 2.

## Test D — regression magnitude (flagged for adjudication)

3 reps, POOL-bound executor (120-SM / 15-group green ctx) vs the Phase 0
baseline (491.8 tok/s, 3.54 tok/W — measured on the full 132 SMs):

| rep | agg tok/s | tok/W |
|---|---|---|
| 1 | 475.11 | 3.452 |
| 2 | 472.49 | 3.413 |
| 3 | 493.98 | 3.551 |
| **mean** | **480.53** | **3.472** |

**Gate (mean-vs-mean, ±3%): PASS** — tok/s 480.53 ∈ [477.0, 506.5];
tok/W 3.472 ∈ [3.434, 3.646]. TFGATE KL ≤ 5.5e-5; dmesg clean.

**Transparency / magnitude (per the Step 1.3b' Test-D stop-condition note —
"adjudicate the regression magnitude"):** the 120-SM green-ctx confinement
costs a **measurable ~2.3% mean throughput** vs the full-132-SM baseline. It is
**marginal**: reps 1 and 2 (475.1, 472.5) fall individually *below* the −3%
per-rep floor (477.0); the 3-rep mean clears the band by 3.5 tok/s on the
strength of rep 3. The regression is **−2.3%, not >3%**, so the stop-condition
does not trigger and the specified mean-vs-mean gate passes — but the cost is
real and the mean sits near the floor. A ~9% SM reduction (132→120) landing as
~2.3% throughput loss is consistent with B=8 decode being only partly
SM-bound. **Flagged for closure adjudication:** accept as-is, or re-measure
with more reps / a recalibrated pool baseline (120-SM, not 132-SM, as Step
1.4's curve memo §8 already prescribes — "anchor the curve to the top pool
point, not Phase B's full-GPU number").

## Anchors — all unchanged

Option 2 required **no kmod, libcipher_rt, or libcipher_v2 change** — only the
executor (Python). Verified post-run: kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`. Executor preserved at
`cipher_batch_executor_gen.py.pre_cp5_4_step1_3b` (`39056662`) in
`cp_5_6/phase_b/session1/` and `cipher_rt_fallback/`.

## Scope deferred

- **Multi-partition free-order fragmentation** (Phase A dependency 3) — ≥2
  partitions freeing out of LIFO order can fragment the pool off the low
  prefix, where torch's count-only pick no longer aligns. Out of Step 1.3b'
  scope; for Step 1.6 (mixed deployment) — needs kmod compaction or an
  accepted-fragmentation policy.

## Adjudication ask

Step 1.3b' is complete — POOL/executor binding works, disjoint coexistence
verified, reaper-safe. Per campaign discipline, **stopping here.** Two items
for the user: (1) accept Test D's −2.3% magnitude, or call for re-measurement;
(2) authorize **Step 1.4** — the confined-pool batch-lift curve (120/104/88/64
SMs), which is now unblocked. Or direct otherwise.
