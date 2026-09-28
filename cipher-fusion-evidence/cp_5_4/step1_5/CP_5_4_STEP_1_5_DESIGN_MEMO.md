# CP 5.4 — Step 1.5 (green-context churn cost) — DESIGN MEMO

**Date:** 2026-05-19. **Type:** measurement design — STOP for adjudication
before any run (the same Phase-A discipline Step 1.4 used). No GPU run yet, no
source modified. Anchors unchanged: kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`.

---

## §0 — Where this sits

Step 1.4 closed on the flat 120 → 40-SM confined-pool lift curve. Step 1.5 is
the next CP 5.4 sub-step.

**Numbering reconciliation.** `CP_5_4_SCOPE_V1.md` §8 lists sub-steps 1.1–1.7;
`CP_5_4_STEP_1_3_DESIGN_MEMO.md` (§ sub-step table) expanded the tail to
**1.5 churn / 1.6 mixed-deployment / 1.7 failure-modes / 1.8 closure**. The
Step 1.4 report's "remaining 1.5–1.8" follows that expanded list — there is no
count slip; §8 simply predates the explicit 1.8-closure split. This memo is
Step 1.5 (churn cost).

## §1 — The §8 spec, as written

> 1.5 | Green-context churn cost — teardown/create microbench; is Marlin
> cubin recompiled per green-ctx | 0.5 d

Two deliverables: **(A)** a teardown/create microbench, **(B)** resolve whether
the Marlin cubin is recompiled per green context.

## §2 — Discrepancy 1: "churn" in the deployed substrate is POOL-only

§8's "teardown/create microbench" reads as an abstract green-ctx API cost. But
in the actual CP 5.4 substrate green contexts are **not** uniformly churned:

- **PARTITION / SHARED tenants** — `cipher_rt_green_ctx.c` header, "Refresh
  policy": the green context is created **once at first `cuStreamCreate` and
  never refreshed** (destroying it mid-run would strand PyTorch's warmup
  streams). A PARTITION tenant does exactly one green-ctx create over its whole
  lifetime and zero teardowns until process exit. **Churn ≈ 0.**
- **The POOL** — `cp54_pool.py:check_resize()` re-`ALLOCATE`s every round and,
  *iff the granted pool group count changed*, calls `build_green_ctx()` which
  **destroys and recreates** the `torch.cuda.GreenContext`. This is the only
  repeated create/teardown in the substrate, and it fires exactly on
  **partition arrival/departure** (a PARTITION allocate shrinks the pool; a
  FREE grows it on the next tick).

So the arbitration-relevant churn cost is specifically **the latency of one
pool-resize event** — the wall-time the batch pool stalls when a partition
tenant arrives or leaves. The microbench should price *that*, not just a bare
`cuGreenCtxCreate/Destroy` loop. Bare-API cost is still measured, as the floor.

**Recommendation:** scope the microbench as **two tiers** —
(i) bare driver-API `cuGreenCtxCreate` + `cuCtxFromGreenCtx` + verify +
`cuGreenCtxDestroy`, swept over group count; (ii) the real `cp54_pool.py`
resize path (`check_resize()` → `build_green_ctx()` incl. the `%smid`
self-verify kernel), measured as the per-event stall the executor sees, with
the **pool-resize event as the headline number**. (Confirm.)

## §3 — Discrepancy 2: the Step 1.4 sub-40-SM floor-probe flag

Step 1.4's report flagged: *"the K=5 reps are noisy (~5% spread)… a soft signal
of approaching but not crossing the §8 knee; Step 1.5 may want to probe past
40 SMs."* That is a **lift-curve** scope item, not a churn item — it does not
belong in Step 1.5's §8 scope.

**Recommendation:** **defer it explicitly.** The sub-40-SM floor probe is a
lift-curve extension; it belongs either as a Step 1.4 addendum or folded into
Step 1.6's mixed-deployment runs (≥2 partitions out naturally drives the pool
below 40 SMs). Step 1.5 stays churn-only. (Confirm — or direct a single
sub-40-SM point folded into 1.5B if the user wants it priced now.)

## §4 — The Marlin sub-question — expected answer + confirmation design

Code reading gives a strong prior, so deliverable (B) is a **confirmation
test**, not an open investigation:

- `ensure_marlin_compiled()` (`cipher_rt_marlin_engine.cpp:316`) is guarded by
  a one-shot `std::atomic g_marlin_state` — the NVRTC compile + `cuModuleLoadData`
  runs **once per process**, period.
- It forces context bring-up with `cudaFree(0)` before the load →
  `g_marlin_module` is resident in the **primary context**.
- The Marlin GEMM launch (`cipher_rt_marlin_engine.cpp:831`) runs on the
  caller's `stream`; CP 5.3 STEP 2's partition-awareness is **grid-sizing only**
  (`grid = green_sm`), and per the `cipher-marlin-primary-ctx-pin` campaign
  record the Marlin GEMM is structurally primary-context-pinned. The green
  context is *read* (`cipher_rt_green_ctx_sm_count()`) but Marlin never *runs
  in* one.

**Expected answer: NO — the Marlin cubin is compiled exactly once per process
and is never recompiled or reloaded on green-ctx churn**, because Marlin's
module lives in the primary context and green-ctx create/destroy never touches
it.

**Confirmation test (1.5C).** In one process, after Marlin is warm
(`g_marlin_state == 2`), drive ≥ 20 pool-resize cycles (`build_green_ctx()`
destroy+recreate), interleaving a real Marlin INT4 GEMM after each, and assert:

1. `g_marlin_state` stays `2` throughout (no transition back to 0/1).
2. The `MARLIN: NVRTC compile starting` log line appears **exactly once**
   (never reoccurs).
3. Every post-churn GEMM is numerically correct vs a pre-churn reference.
4. **Latency** — every post-churn Marlin GEMM wall-time is within noise of the
   warm pre-churn baseline (≤ 1.1×). This catches a *silent* `cuModuleLoadData`
   reload into a new context — no NVRTC, no log line, but real ms — which
   asserts 1–3 would all miss.

**Launch-context discipline (avoids a false negative).** `g_marlin_module`
lives in the **primary** context. The post-churn GEMM in this test must be
issued with the primary context current (the `PrimaryCtxGuard` code path the
runtime substrate already uses) — otherwise a context-mismatch launch failure,
unrelated to cubin invalidation, would be misread as a finding.

If — contrary to the prior — a recompile or reload *is* observed, that is a
real finding (per-resize NVRTC cost is seconds) and Step 1.5 stop-and-adjudicates.

## §5 — Proposed Step 1.5 plan (pending adjudication)

1. **1.5A** (this memo) — adjudicate Discrepancies 1 & 2 and the §4 framing.
2. **1.5B — churn microbench.**
   - *Tier (i)* bare driver-API create/teardown cost vs group count
     {1,3,5,7,9,11,13,15}, ≥ 100 iterations each → mean / p50 / p99 µs.
   - *Tier (ii)* the `cp54_pool.py` resize path — instrument
     `check_resize()` / `build_green_ctx()` to time one resize event
     (`ALLOCATE` ioctl + `GreenContext` destroy/create + `%smid` self-verify),
     measured across the transitions a real partition arrival/departure
     produces (e.g. 15→13→11→13→15 groups). **The timed window must include a
     `torch.cuda.synchronize()` before the `GreenContext` destroy** — pending
     green-stream work would otherwise either UB or implicitly stall the
     destroy, under-attributing the per-event stall (the same cross-stream
     hazard class Step 1.4C paid for). Report the per-event executor stall.
3. **1.5C — Marlin cubin confirmation** (§4 test, asserts 1–4, primary-context
   launch).
4. **1.5D — analysis / report** — is pool-resize churn cheap enough to ignore
   in the arbitration policy, or must the policy rate-limit / amortize
   resizes? State the threshold. Evidence packaged as
   `cp_5_4_step1_5_evidence.tar.gz` (+ `.md5`), per phase discipline.

Source touches expected: a throwaway microbench harness + timing
instrumentation in `cp54_pool.py` (already a Step 1.3b' test file, not an
anchor). **No anchor rotation** — measurement step.

## §6 — Anchors

Unchanged by this memo and expected unchanged by all of Step 1.5: kmod
`8d777dfb`, libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`, cipher_kv_bridge
`fca6843d`.

## §7 — Adjudication ask

**STOPPING HERE for adjudication.** No run, no source modified. Decisions:

1. **Discrepancy 1** — accept the two-tier microbench (bare-API floor + real
   pool-resize stall), with the pool-resize event as the headline number.
2. **Discrepancy 2** — accept deferring the sub-40-SM floor probe out of
   Step 1.5 (→ Step 1.4 addendum or Step 1.6), or direct it folded into 1.5B.
3. **§4** — accept the Marlin question as a confirmation test (asserts 1–4)
   against the code-derived "NO recompile" prior, stop-and-adjudicate only if
   a recompile/reload is actually observed.

No code until this memo is adjudicated.
